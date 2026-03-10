#!/usr/bin/env python3
"""追加生成Golang训练数据"""
import json
import random
import os
import sys

# 追加模式：根据现有数据量计算需要生成的数量
def get_existing_count(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except:
        return 0

def append_remaining(target=20000):
    output_file = "/home/lbp/projects/code-sapiens/train_data/tmp/data3.jsonl"
    
    existing = get_existing_count(output_file)
    need = target - existing
    
    if need <= 0:
        print(f"已有 {existing} 条数据，已达到目标")
        return
    
    print(f"现有 {existing} 条，需要追加 {need} 条")
    
    # 简单追加 - 直接导入主模块
    sys.path.insert(0, '/home/lbp/projects/code-sapiens')
    import generate_go_train_data as generator
    
    # 导入需要的常量和函数
    from generate_go_train_data import (
        QUESTION_BANK, CODE_ANSWERS, RAG_QUERIES, 
        generate_one_item, generate_batch, save_to_jsonl
    )
    
    # 导入 random 并设置种子
    random.seed()
    
    # 导入 CODE_ANSWERS 和 RAG_QUERIES
    generator.CODE_ANSWERS = CODE_ANSWERS
    generator.RAG_QUERIES = RAG_QUERIES
    
    # 生成所需数量
    batch = []
    for i in range(need):
        # 随机选择一个类别
        category_info = random.choice(generator.QUESTION_BANK)
        item = generator.generate_one_item(category_info, i)
        batch.append(item)
        
        if len(batch) >= 50:
            generator.save_to_jsonl(batch, output_file)
            print(f"已追加 {len(batch)} 条, 共 {existing + i + 1} 条")
            batch = []
    
    if batch:
        generator.save_to_jsonl(batch, output_file)
        print(f"完成！共 {get_existing_count(output_file)} 条数据")

if __name__ == "__main__":
    append_remaining()
