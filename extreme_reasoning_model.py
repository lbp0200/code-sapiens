# extreme_reasoning_model.py
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RMSNorm, RotaryEmbedding, MambaSSD, MambaHybridBlock, Mamba2Block, Mamba2Attention, TRMamba2AttnBlock


class ExtremeReasoningModel(nn.Module):
    def __init__(
        self,
        vocab_size=50257,  # GPT-2 风格 BPE vocab
        d_model=2560,      # 10G VRAM 安全范围
        n_layers=48,       # ~9B params
        n_heads=16,        # 增加到 16 (原 8)
        d_state=128,
        expand=2,
        max_seq_len=131072,  # 128K 上下文
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token + Position Embedding
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rotary_emb = RotaryEmbedding(
            d_model // n_heads, max_position_embeddings=max_seq_len
        )

        # Transformer-like 层堆叠 (使用 TRMamba2AttnBlock)
        self.layers = nn.ModuleList(
            [
                TRMamba2AttnBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_state=d_state,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Final norm + LM head
        self.norm_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 共享 embedding 权重（常见优化，节省 ~30% 参数）
        self.lm_head.weight = self.embed.weight

        # 参数统计
        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"总参数量: {total / 1e6:.2f}M (约 {total:,} 参数)")

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [batch, seq_len] LongTensor
        返回 logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embedding
        x = self.embed(input_ids)  # [b, l, d]

        # 生成 RoPE cos/sin
        cos, sin = self.rotary_emb(x, seq_len=seq_len)

        # 生成 causal mask（如果需要 attention 用）
        # For MambaHybridBlock, pass None - Mamba layers handle causality,
        # and we'll use 2D mask [seq, seq] which PyTorch broadcasts
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1,
            )  # [seq, seq] - 2D gets broadcast to all heads
        else:
            causal_mask = attention_mask

        # 通过各层
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask)

        # Final norm + head
        x = self.norm_final(x)
        logits = self.lm_head(x)  # [b, l, vocab]

        return logits

    def generate(
        self, prompt, tokenizer, max_new_tokens=100, temperature=0.8, top_k=50
    ):
        self.eval()
        with torch.no_grad():
            # tokenizer 作为参数传入
            input_ids = (
                torch.tensor(tokenizer.encode(prompt))
                .unsqueeze(0)
                .to(self.embed.weight.device)
            )

            for _ in range(max_new_tokens):
                logits = self(input_ids)[:, -1, :]  # 只取最后一个 token 的 logits
                logits = logits / temperature

                # top-k sampling（可选，提升生成质量，避免胡言乱语）
                if top_k is not None:
                    v, ix = torch.topk(logits, min(top_k, logits.size(-1)))
                    probs = F.softmax(v, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    next_token = ix.gather(-1, next_token)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_token], dim=1)

            return tokenizer.decode(input_ids[0].tolist())

    def quantize(self, dtype=torch.qint8):
        """动态量化模型 (INT8)

        使用 PyTorch 动态量化，权重转为 INT8
        显存减少约 75%，推理加速约 2-4x

        注意: embedding 层需要特殊处理，不量化
        """
        import torch.ao.quantization

        # 只量化 Linear 层，不量化 Embedding (embedding 量化需要特殊配置)
        torch.ao.quantization.quantize_dynamic(
            self,
            {nn.Linear},
            dtype=dtype,
            inplace=True
        )
        print("模型量化完成 (INT8, Linear层)")
        return self

    def get_quantized_model(self):
        """返回量化后的模型副本"""
        import copy
        quantized = copy.deepcopy(self)
        quantized.quantize()
        return quantized


# 测试完整模型
if __name__ == "__main__":
    model = ExtremeReasoningModel(
        vocab_size=50257,
        d_model=2560,
        n_layers=36,
        n_heads=16,
        d_state=128,
        expand=2,
        max_seq_len=131072,
    )

    # 随机输入测试 forward
    # input_ids = torch.randint(0, 50257, (2, 128))  # fake token ids
    # logits = model(input_ids)
    # print("Model logits shape:", logits.shape)  # [2, 128, 50257]
    # print("完整模型 forward 成功！")

    enc = tiktoken.get_encoding("gpt2")

    print(model.generate(
	    "To be, or not to be: that is the ",
	    tokenizer=enc,
	    max_new_tokens=100,
	    temperature=0.9,   # 稍高一点，增加多样性
	    top_k=40           # 加 top-k，避免太随机
	))

    # 参数量打印（应该接近 35-40M）
