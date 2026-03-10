#!/usr/bin/env python3
"""
增量训练脚本 - 自动轮询多批数据
每批训练指定步数，4批完成后自动停止
"""
import os
import sys
import glob
import subprocess

# 配置
BATCH_DIR = "train_data/batches"
STEPS_PER_BATCH = 10000  # 每批训练步数
SEQ_LEN = 512
BATCH_SIZE = 8

def get_batch_files():
    """获取所有批次文件，按序号排序"""
    files = sorted(glob.glob(f"{BATCH_DIR}/part*.jsonl"))
    return files

def run_training(data_file, max_steps):
    """运行一次训练"""
    cmd = [
        "python", "train.py",
        "--data", data_file,
        "--seq_len", str(SEQ_LEN),
        "--max_steps", str(max_steps),
        "--batch_size", str(BATCH_SIZE),
        "--resume"  # 断点续训
    ]
    print(f"\n{'='*50}")
    print(f"训练: {data_file}")
    print(f"步数: {max_steps}")
    print(f"{'='*50}\n")

    result = subprocess.run(cmd, cwd=os.getcwd())
    return result.returncode == 0

def main():
    batch_files = get_batch_files()

    if not batch_files:
        print(f"错误: 在 {BATCH_DIR} 中没有找到 part*.jsonl 文件")
        print("请先运行: python split_data.py")
        sys.exit(1)

    print(f"找到 {len(batch_files)} 批数据:")
    for f in batch_files:
        print(f"  - {f}")

    # 轮询训练
    for i, batch_file in enumerate(batch_files):
        print(f"\n{'#'*50}")
        print(f"第 {i+1}/{len(batch_files)} 批")
        print(f"{'#'*50}")

        success = run_training(batch_file, STEPS_PER_BATCH)

        if not success:
            print(f"训练失败: {batch_file}")
            break

    print("\n所有批次训练完成！")

if __name__ == "__main__":
    main()
