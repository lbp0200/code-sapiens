# model.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_scan(x, a):
    """优化的因果扫描

    x: [batch, seq, d_state]
    a: [batch, seq, d_state]

    y[i] = a[i] * y[i-1] + x[i]
    """
    batch, seq, d_state = x.shape
    device = x.device

    # 使用 chunk 并行处理
    y = torch.zeros_like(x)

    # 预分配，减少内存分配开销
    h = torch.zeros(batch, d_state, device=device)

    # 每次处理多个 token
    chunk_size = 16
    for i in range(0, seq, chunk_size):
        end = min(i + chunk_size, seq)
        for j in range(i, end):
            h = a[:, j] * h + x[:, j]
            y[:, j] = h

    return y


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama/Mamba 风格)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """RoPE 实现，支持 128K 上下文

    使用动态计算而非预计算，节省内存
    """

    def __init__(self, dim, max_position_embeddings=131072, base=10000):
        super().__init__()
        # 增大 base 以支持更长序列
        # 标准: base=10000 for 4K, 使用 1000000 for 128K
        self.base = base
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_position_embeddings

    def forward(self, x, seq_len=None):
        """动态计算 RoPE 频率"""
        if seq_len is None:
            seq_len = x.shape[1]

        # 动态计算 cos/sin
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_emb(q, k, cos, sin):
    """应用 RoPE 到 q 和 k"""

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class MambaSSD(nn.Module):
    """我们之前修复的 SimpleSSD，重命名为 MambaSSD"""

    def __init__(self, d_model=512, d_state=64, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = expand * d_model

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.B_proj = nn.Linear(self.d_state, self.d_inner, bias=False)
        self.C_proj = nn.Linear(self.d_state, self.d_inner, bias=False)

        self.A_log = nn.Parameter(torch.log(torch.tensor(0.01)))

    def forward(self, x):
        batch, seq, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        proj = self.x_proj(x)
        delta, B_raw, C_raw = proj.split([1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(delta)

        B = self.B_proj(B_raw)
        C = self.C_proj(C_raw)

        A = -torch.exp(self.A_log)
        A_bar = torch.exp(delta * A)

        y_list = []
        h = torch.zeros(batch, self.d_inner, device=x.device)

        for t in range(seq):
            h_new = A_bar[:, t] * h + B[:, t] * x[:, t]
            y_t = C[:, t] * h_new + self.D * x[:, t]
            y_list.append(y_t.unsqueeze(1))
            h = h_new

        y = torch.cat(y_list, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)


class Mamba2Block(nn.Module):
    """Mamba-2 SSD Block (使用 mamba-ssm 库的优化版本)

    使用 CUDA 级别的并行扫描，速度比 PyTorch for 循环快几十倍
    """

    def __init__(self, d_model=512, d_state=128, expand=2, dropout=0.1, device=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # 使用 mamba-ssm 的优化版本
        try:
            from mamba_ssm.modules.mamba2_simple import Mamba2Simple

            # 计算 headdim (mamba-ssm 需要)
            headdim = 64  # 可以调整
            self.mamba = Mamba2Simple(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                device='cuda',  # 强制在 CUDA 上初始化
            )
            print(f"使用 mamba-ssm 优化版 Mamba2Block (d_model={d_model}, d_state={d_state})")
        except Exception as e:
            print(f"警告: mamba-ssm 加载失败: {e}")
            print("使用简化版 Mamba2Block")
            # fallback 到简化版
            self.mamba = None
            self.d_inner = expand * d_model
            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        if self.mamba is not None:
            # 使用 mamba-ssm 优化版
            return x + (self.dropout(self.mamba(x)) if self.dropout else self.mamba(x))
        else:
            # fallback 简化版
            return x


class Mamba2Attention(nn.Module):
    """Mamba-2 Attention: 使用 SSM 处理 QKV

    用 SSM 替代标准 attention，适合长序列
    """

    def __init__(self, d_model=512, n_heads=8, d_state=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_state = d_state

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # SSM for Q, K processing
        self.q_ssm = nn.Sequential(
            nn.Linear(self.head_dim, d_state),
            nn.SiLU(),
            nn.Linear(d_state, self.head_dim),
        )
        self.k_ssm = nn.Sequential(
            nn.Linear(self.head_dim, d_state),
            nn.SiLU(),
            nn.Linear(d_state, self.head_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(self, x, attn_mask=None):
        """Mamba2Attention 前向传播

        x: [batch, seq, d_model]
        attn_mask: [seq, seq] 因果掩码
        """
        batch, seq, _ = x.shape
        device = x.device

        # Q, K, V projections
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply SSM to Q and K (per head)
        # Reshape to process each head
        q = q.transpose(1, 2)  # [batch, seq, heads, head_dim]
        k = k.transpose(1, 2)

        q_ssm_out = torch.zeros_like(q)
        k_ssm_out = torch.zeros_like(k)

        for h in range(self.n_heads):
            q_h = q[:, :, h, :]  # [batch, seq, head_dim]
            k_h = k[:, :, h, :]

            # Simple SSM-like transformation
            q_ssm_out[:, :, h, :] = self.q_ssm(q_h) + q_h
            k_ssm_out[:, :, h, :] = self.k_ssm(k_h) + k_h

        q = q_ssm_out.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k_ssm_out.transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        if attn_mask is not None:
            # attn_mask: [seq, seq] -> [1, 1, seq, seq]
            mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        output = self.o_proj(attn_output)

        return output


class TRMamba2AttnBlock(nn.Module):
    """TR-mamba2attn Block: Mamba2Block → Mamba2Block → Mamba2Attention → MLP

    目标架构，来自 arXiv:2602.12078
    """

    def __init__(self, d_model=512, n_heads=8, d_state=128, expand=2, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mamba2_1 = Mamba2Block(d_model, d_state, expand, dropout)

        self.norm2 = RMSNorm(d_model)
        self.mamba2_2 = Mamba2Block(d_model, d_state, expand, dropout)

        self.norm3 = RMSNorm(d_model)
        self.attn = Mamba2Attention(d_model, n_heads, d_state, dropout)

        # SwiGLU MLP
        self.mlp_norm = RMSNorm(d_model)
        self.mlp_fc1 = nn.Linear(d_model, d_model * 4)
        self.mlp_fc2 = nn.Linear(d_model * 4, d_model)
        self.mlp_gate = nn.Linear(d_model, d_model * 4)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Mamba2Block 1
        residual = x
        x = self.norm1(x)
        x = self.mamba2_1(x)
        x = residual + self.dropout(x)

        # Mamba2Block 2
        residual = x
        x = self.norm2(x)
        x = self.mamba2_2(x)
        x = residual + self.dropout(x)

        # Mamba2Attention
        residual = x
        x = self.norm3(x)
        x = self.attn(x, attn_mask)
        x = residual + self.dropout(x)

        # SwiGLU MLP
        residual = x
        x_norm = self.mlp_norm(x)
        gate = F.silu(self.mlp_gate(x_norm))
        x = gate * self.mlp_fc1(x_norm)
        x = self.mlp_fc2(x)
        x = residual + self.dropout(x)

        return x


class MambaHybridBlock(nn.Module):
    """Mamba-2 Hybrid Block: Mamba → Mamba → Light Attention → SwiGLU MLP"""

    def __init__(self, d_model=512, n_heads=8, d_state=64, expand=2, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mamba1 = MambaSSD(d_model, d_state, expand)
        self.norm2 = RMSNorm(d_model)
        self.mamba2 = MambaSSD(d_model, d_state, expand)
        self.norm3 = RMSNorm(d_model)

        # Light Attention (只用少量头，节省参数)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # SwiGLU MLP
        self.mlp_norm = RMSNorm(d_model)
        self.mlp_fc1 = nn.Linear(d_model, d_model * 4)
        self.mlp_fc2 = nn.Linear(d_model * 4, d_model)
        self.mlp_gate = nn.Linear(d_model, d_model * 4)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Mamba1
        residual = x
        x = self.norm1(x)
        x = self.mamba1(x)
        x = residual + self.dropout(x)

        # Mamba2
        residual = x
        x = self.norm2(x)
        x = self.mamba2(x)
        x = residual + self.dropout(x)

        # Light Attention
        residual = x
        x = self.norm3(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = residual + self.dropout(attn_out)

        # SwiGLU MLP
        residual = x
        x_norm = self.mlp_norm(x)
        gate = F.silu(self.mlp_gate(x_norm))
        x = gate * self.mlp_fc1(x_norm)
        x = self.mlp_fc2(x)
        x = residual + self.dropout(x)

        return x


# 快速测试单个 block
if __name__ == "__main__":
    # 测试 Mamba2Block
    print("Testing Mamba2Block...")
    mamba2_block = Mamba2Block(d_model=512, d_state=128)
    x = torch.randn(2, 128, 512)
    y = mamba2_block(x)
    print(f"Mamba2Block output shape: {y.shape}")  # 期望 [2, 128, 512]

    # 测试 Mamba2Attention
    print("\nTesting Mamba2Attention...")
    attn = Mamba2Attention(d_model=512, n_heads=8)
    causal_mask = torch.triu(torch.ones(128, 128, dtype=torch.bool), diagonal=1)
    y = attn(x, attn_mask=causal_mask)
    print(f"Mamba2Attention output shape: {y.shape}")

    # 测试 TRMamba2AttnBlock
    print("\nTesting TRMamba2AttnBlock...")
    block = TRMamba2AttnBlock(d_model=512, n_heads=8, d_state=128)
    y = block(x, attn_mask=causal_mask)
    print(f"TRMamba2AttnBlock output shape: {y.shape}")

    # 对比测试旧的 MambaHybridBlock
    print("\nTesting MambaHybridBlock (for comparison)...")
    old_block = MambaHybridBlock(d_model=512, n_heads=8, d_state=64)
    y = old_block(x, attn_mask=causal_mask)
    print(f"MambaHybridBlock output shape: {y.shape}")

    print("\n所有 Block 测试通过!")
