# 训练数据生成方案

## 数据格式设计

每条数据是一个JSON对象，包含：

```json
{
  "question": "用户的问题",
  "reasoning": "模型的思考过程",
  "tool_calls": [
    {"name": "search_rag", "query": "查询内容"}
  ],
  "tool_results": ["RAG返回的相关代码/文档"],
  "answer": "最终生成的代码"
}
```

## 训练时的输入输出

训练时，模型看到的是：
- **输入**：question + (可选的history)
- **目标输出**：reasoning + tool_calls + tool_results + answer

## 生成Prompt（给LLM用的）

见 `generate_prompt.txt`

## 生成脚本

见 `generate_training_data.py`
