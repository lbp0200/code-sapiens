# train.py
import argparse
import glob
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch import amp
import tiktoken
import math

from extreme_reasoning_model import ExtremeReasoningModel

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default=None, help="训练数据文件 (jsonl格式)")
parser.add_argument("--batch_size", type=int, default=4)  # 减小以节省显存
parser.add_argument("--gradient_accumulation", type=int, default=8, help="梯度累积步数")
parser.add_argument("--seq_len", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=1e-4)  # 降低学习率
parser.add_argument("--max_steps", type=int, default=10000)  # 大幅增加训练步数
parser.add_argument("--warmup_steps", type=int, default=200)  # 调整 warmup
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--data_repeat", type=int, default=50)  # 增加数据重复次数
parser.add_argument("--resume", action="store_true", help="从 checkpoint 恢复训练")
parser.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="checkpoint 文件路径")
parser.add_argument("--dry_run", action="store_true", help="仅测试几个 batch 后退出")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
if device == "cuda":
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Tokenizer (GPT-2 BPE)
enc = tiktoken.get_encoding("gpt2")

# 读取训练数据
import json

if args.data and args.data.endswith('.jsonl'):
    # JSONL 格式: question + reasoning + answer
    print(f"从 JSONL 文件加载: {args.data}")
    with open(args.data, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    problems = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        # 格式: 问题\n推理\n答案
        q = data.get('question', '')
        r = data.get('reasoning', '')
        a = data.get('answer', '')
        if q and a:
            problem = f"{q}\n\n{r}\n\n{a}"
            problems.append(problem)

    print(f"从 JSONL 加载 {len(problems)} 条数据")

else:
    # 从 train_data 文件夹读取所有 .md 文件
    data_dir = "train_data"
    md_files = sorted(glob.glob(os.path.join(data_dir, "*.md")))
    print(f"找到 {len(md_files)} 个训练数据文件:")

    text_parts = []
    for f in md_files:
        with open(f, "r", encoding="utf-8") as file:
            content = file.read().strip()
            if content:
                text_parts.append(content)
                print(f"  - {os.path.basename(f)}: {len(content)} chars")

    # 合并所有数据，重复以增加数据量
    text = "\n\n".join(text_parts) * args.data_repeat

    tokens = enc.encode(text)
    print(f"数据集 token 数量: {len(tokens)}")

    # 直接用问题作为独立样本，不做滑动窗口
    import re

    # 合并原始数据（不重复）
    raw_text = "\n\n".join(text_parts)
    # 按问题分割
    problems = re.split(r'\*\*\d+\.\*\*', raw_text)
    problems = [p.strip() for p in problems if p.strip()]

# 对每个问题编码为独立样本
problem_tokens_list = [enc.encode(p) for p in problems]
print(f"原始问题数: {len(problems)}")

# 过滤掉太长的问题
max_problem_len = args.seq_len
valid_problems = [t for t in problem_tokens_list if len(t) <= max_problem_len]
print(f"有效问题数 (len <= {max_problem_len}): {len(valid_problems)}")

# 重复以增加数据量
all_problems = valid_problems * args.data_repeat
print(f"总样本数 (重复 {args.data_repeat} 次): {len(all_problems)}")


class ProblemDataset(Dataset):
    """每个样本是一个完整问题，不做滑动窗口"""
    def __init__(self, problem_tokens_list, seq_len):
        self.problems = problem_tokens_list
        self.seq_len = seq_len

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        tokens = self.problems[idx]
        # padding 或截断到固定长度
        if len(tokens) < self.seq_len:
            tokens = tokens + [0] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[:self.seq_len]
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])  # input, target


# 划分训练/验证
n_train = int(len(all_problems) * 0.9)
train_problems = all_problems[:n_train]
val_problems = all_problems[n_train:]

train_dataset = ProblemDataset(train_problems, args.seq_len)
val_dataset = ProblemDataset(val_problems, args.seq_len)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,  # 多进程加载
    pin_memory=True,  # 加速 CPU->GPU 传输
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=2,
    pin_memory=True,
)

print(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

# 模型 (~35M 参数)
model = ExtremeReasoningModel(
    vocab_size=enc.n_vocab,
    d_model=512,    # 35M
    n_layers=8,    # 35M
    n_heads=8,
    d_state=64,
    expand=2,
    max_seq_len=args.seq_len + 100,
).to(device)

# 优化1: Gradient Checkpointing (时间换空间)
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()
elif hasattr(model, 'enable_gradient_checkpointing'):
    model.enable_gradient_checkpointing()
print("Gradient Checkpointing: 开启 (用时间换显存)")

# 混合精度训练 (BF16) - 已经在用 AMP

optimizer = AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=(0.9, 0.95),
    weight_decay=args.weight_decay
)

# 断点续训
if args.resume and os.path.exists(args.checkpoint):
    print(f"加载 checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    step = checkpoint.get('step', 0)
    print(f"从 step {step} 恢复训练")

# 学习率调度器: Warmup + Cosine Annealing
def lr_lambda(step):
    if step < args.warmup_steps:
        return step / args.warmup_steps
    else:
        progress = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = amp.GradScaler('cuda')  # AMP

model.train()
step = 0
best_val_loss = float('inf')

# 验证函数
def evaluate():
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = [t.to(device) for t in batch]
            with amp.autocast(device_type='cuda'):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                )
            val_loss += loss.item()
    model.train()
    return val_loss / max(1, len(val_loader))

# Dry run: 测试前几个 batch
if args.dry_run:
    print("=== Dry Run: 测试 5 个 batch ===")
    for i, batch in enumerate(train_loader):
        if i >= 5:
            break
        inputs, targets = [t.to(device) for t in batch]
        with amp.autocast(device_type='cuda'):
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        print(f"Batch {i}: Loss = {loss.item():.4f}")
    print("=== Dry Run 完成 ===")
    exit(0)

accumulation_steps = args.gradient_accumulation
print(f"梯度累积: {accumulation_steps} 步 (batch_size={args.batch_size}, 有效={args.batch_size * accumulation_steps})")

for batch in train_loader:
    inputs, targets = [t.to(device) for t in batch]

    with amp.autocast(device_type='cuda'):
        logits = model(inputs)  # [b, seq, vocab]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        loss = loss / accumulation_steps  # 归一化损失

    scaler.scale(loss).backward()

    if (step + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    step += 1

    if step % 10 == 0:
        gpu_mem = torch.cuda.memory_reserved() / 1e9 if device == "cuda" else 0
        lr = scheduler.get_last_lr()[0]
        print(
            f"Step {step}/{args.max_steps} | Loss: {loss.item():.4f} | LR: {lr:.2e} | GPU Mem: {gpu_mem:.2f} GB"
        )

    if step % 100 == 0:
        val_loss = evaluate()
        print(f"  -> Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  -> Best model saved! Val Loss: {val_loss:.4f}")

    if step >= args.max_steps:
        break

print("训练完成！")

# 保存完整 checkpoint（包含模型、优化器、调度器状态）
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'step': step,
    'best_val_loss': best_val_loss,
}
torch.save(checkpoint, args.checkpoint)
print(f"Checkpoint 已保存: {args.checkpoint}")
