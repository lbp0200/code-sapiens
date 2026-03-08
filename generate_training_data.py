"""
生成训练数据的脚本
用法: python generate_training_data.py --api_key YOUR_KEY --output data.jsonl --count 100
"""
import json
import argparse
import os
import re

# 如果你有 OpenAI key，可以用官方库
# pip install openai
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


PROMPT_TEMPLATE = """你是一个数据生成器，生成用于训练"代码推理模型"的示例数据。

## 任务描述

生成JSON格式的训练数据，每条数据包含：
1. question: 用户的一个编程问题
2. reasoning: 模型解决这个问题时的思考过程（详细展示思考步骤）
3. tool_calls: 模型主动查询RAG的动作（数组，每个包含name和query）
4. tool_results: 模拟RAG返回的结果（代码片段或文档）
5. answer: 最终生成的完整代码

## 重要规则

1. reasoning要详细，展示完整的思考过程
2. tool_calls要自然，只有真正需要查资料时才调用RAG（最多2个）
3. tool_results要真实，模拟RAG可能返回的内容
4. answer要正确，最终代码要能正确运行

## 输出格式

每行一个JSON对象，不要其他内容。每条数据要完整，不要截断。

生成 {count} 条不同类型的编程问题数据："""


def generate_with_openai(api_key: str, count: int, model: str = "gpt-4o") -> list:
    """用OpenAI API生成数据"""
    if not HAS_OPENAI:
        raise ImportError("请安装 openai 库: pip install openai")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个专业的数据生成器，生成JSON格式的训练数据。"},
            {"role": "user", "content": PROMPT_TEMPLATE.format(count=count)}
        ],
        temperature=0.8,
    )

    content = response.choices[0].message.content

    # 解析JSON Lines
    lines = content.strip().split('\n')
    results = []
    for line in lines:
        line = line.strip()
        if line.startswith('```json'):
            line = line[7:]
        if line.startswith('```'):
            line = line[3:]
        if line.endswith('```'):
            line = line[:-3]
        line = line.strip()
        if line:
            try:
                data = json.loads(line)
                results.append(data)
            except json.JSONDecodeError:
                # 尝试提取JSON
                match = re.search(r'\{.*\}', line, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group())
                        results.append(data)
                    except:
                        pass

    return results


def generate_local(model: str, count: int) -> list:
    """
    用本地模型生成（如果你有本地LLM服务）
    需要你自己实现
    """
    raise NotImplementedError("本地模型生成需要你自己实现")


def save_jsonl(data: list, output_path: str):
    """保存为JSONL格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存 {len(data)} 条数据到 {output_path}")


def convert_to_training_format(jsonl_path: str, output_path: str):
    """
    把生成的JSONL转换成训练格式（tokenize后的数据）
    这里先只做格式验证
    """
    valid_count = 0
    errors = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                # 检查必要字段
                required = ['question', 'reasoning', 'tool_calls', 'tool_results', 'answer']
                for field in required:
                    if field not in data:
                        errors.append(f"第{i+1}行缺少字段: {field}")
                        raise ValueError(f"缺少字段: {field}")

                # 验证tool_calls格式
                for tc in data['tool_calls']:
                    if 'name' not in tc or 'query' not in tc:
                        errors.append(f"第{i+1}行tool_calls格式错误")
                        raise ValueError("tool_calls格式错误")

                valid_count += 1

            except Exception as e:
                if str(e) not in [str(x) for x in errors[-5:]]:
                    errors.append(f"第{i+1}行: {e}")

    print(f"验证完成: {valid_count} 条有效, {len(errors)} 个错误")
    if errors:
        for err in errors[:10]:
            print(f"  - {err}")

    return valid_count


def main():
    parser = argparse.ArgumentParser(description="生成训练数据")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""),
                        help="OpenAI API Key")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="使用的模型")
    parser.add_argument("--count", type=int, default=20,
                        help="生成数量")
    parser.add_argument("--output", type=str, default="train_data/code_reasoning.jsonl",
                        help="输出文件路径")

    args = parser.parse_args()

    if args.api_key:
        print(f"使用 {args.model} 生成 {args.count} 条数据...")
        data = generate_with_openai(args.api_key, args.count, args.model)
        save_jsonl(data, args.output)
    else:
        # 打印本地生成指南
        print("""
未提供 API Key，请选择以下方式生成数据：

1. 使用 OpenAI API:
   python generate_training_data.py --api_key YOUR_KEY --count 100

2. 使用其他LLM API（如 Anthropic）:
   请修改脚本，调用对应的 API

3. 手动编写数据:
   按照 format 生成 JSONL 格式的文件

生成的数据格式应该是:
{"question":"...","reasoning":"...","tool_calls":[{"name":"search_rag","query":"..."}],"tool_results":["..."],"answer":"..."}
        """)


if __name__ == "__main__":
    main()
