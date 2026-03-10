#!/usr/bin/env python3
"""
将数据分成多个文件，每批约700条（seq_len=512限制）
"""
import json
import os
import math
import tiktoken

enc = tiktoken.get_encoding("gpt2")

def split_data(input_file, output_dir, max_per_file=700, max_tokens=512):
    """按数量分割数据，自动截断超长数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 读取所有数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"总数据: {len(data)} 条")

    # 截断超长数据
    truncated = 0
    for item in data:
        text = f"{item['question']}\n\n{item['reasoning']}\n\n{item['answer']}"
        tokens = enc.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = enc.decode(tokens)
            # 尝试解析回结构
            parts = text.split('\n\n')
            if len(parts) >= 3:
                item['question'] = parts[0]
                item['reasoning'] = parts[1]
                item['answer'] = '\n\n'.join(parts[2:])
            truncated += 1

    if truncated > 0:
        print(f"截断了 {truncated} 条超长数据")

    # 按长度排序（短的在前，充分利用 seq_len）
    data_with_len = []
    for item in data:
        text = f"{item['question']}\n\n{item['reasoning']}\n\n{item['answer']}"
        tokens = len(enc.encode(text))
        data_with_len.append((tokens, item))

    data_with_len.sort(key=lambda x: x[0])  # 按长度排序

    # 分成多份
    num_files = math.ceil(len(data) / max_per_file)
    print(f"分成 {num_files} 个文件，每批约 {max_per_file} 条")

    for i in range(num_files):
        start = i * max_per_file
        end = min(start + max_per_file, len(data))
        batch = data_with_len[start:end]

        output_file = os.path.join(output_dir, f"part{i+1}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, item in batch:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # 统计实际 token 长度
        sample_text = f"{batch[0][1]['question']}\n\n{batch[0][1]['reasoning']}\n\n{batch[0][1]['answer']}"
        sample_tokens = len(enc.encode(sample_text))
        print(f"  part{i+1}.jsonl: {len(batch)} 条 (sample: {sample_tokens} tokens)")

    print(f"\n完成！文件保存在 {output_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="train_data/final/data.jsonl")
    parser.add_argument("--output", default="train_data/batches")
    parser.add_argument("--max", type=int, default=700)
    args = parser.parse_args()

    split_data(args.input, args.output, args.max)
