# test_model.py
import torch
import tiktoken
from extreme_reasoning_model import ExtremeReasoningModel

# 加载 tokenizer
enc = tiktoken.get_encoding("gpt2")

# 加载模型 (使用训练时的配置)
model = ExtremeReasoningModel(
    vocab_size=enc.n_vocab,
    d_model=512,
    n_layers=12,
    n_heads=8,
    d_state=64,
    expand=2,
    max_seq_len=512,
)

# 加载权重
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()

print("模型加载完成！")
print("="*50)

# 测试问题
test_questions = [
    "Python如何逐行读取文本文件内容？",
    "JavaScript中如何使用map方法转换数组元素？",
    "SQL如何查询表中不重复的记录？",
]

for q in test_questions:
    print(f"\n问题: {q}")
    print("-"*50)
    result = model.generate(q, enc, max_new_tokens=200, temperature=0.8)
    # 只显示生成的部分
    prompt_len = len(q)
    answer = result[prompt_len:].strip()
    print(f"回答: {answer[:500]}...")
    print("="*50)
