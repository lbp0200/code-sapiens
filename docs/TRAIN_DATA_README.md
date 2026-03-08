# 训练数据说明

## 快速开始

### 1. 生成数据

```bash
# 使用 OpenAI API
export OPENAI_API_KEY=your_key_here
python generate_training_data.py --count 100 --output train_data/code_reasoning.jsonl
```

### 2. 验证数据

```bash
python generate_training_data.py --input train_data/code_reasoning.jsonl
```

### 3. 开始训练

```bash
python train.py --data train_data/code_reasoning.jsonl
```

## 数据格式示例

```json
{
  "question": "写一个Python快速排序函数",
  "reasoning": "用户需要排序功能。快速排序是一个经典的O(n log n)算法。我需要考虑：1. 基准元素选择 2. 分区操作 3. 递归调用。但我不太确定Pythonic的实现方式，让我查一下RAG。",
  "tool_calls": [
    {"name": "search_rag", "query": "python quick sort implementation"}
  ],
  "tool_results": [
    "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)"
  ],
  "answer": "def quick_sort(arr):\n    \"\"\"快速排序算法\"\"\"\n    if len(arr) <= 1:\n        return arr\n    \n    # 选择中间元素作为基准\n    pivot = arr[len(arr) // 2]\n    \n    # 分区\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    \n    # 递归排序并合并\n    return quick_sort(left) + middle + quick_sort(right)\n\n# 测试\nif __name__ == \"__main__\":\n    test_list = [3, 6, 8, 10, 1, 2, 1]\n    print(quick_sort(test_list))  # [1, 1, 2, 3, 6, 8, 10]"
}
```

## 训练格式

训练时需要把数据转换成 token IDs：

```
[Question] question [Reasoning] reasoning [ToolCall] tool_call1 [Result] tool_result1 [Answer] answer
```

具体格式见 `train.py` 的数据处理部分。
