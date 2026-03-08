# 项目总纲：代码推理模型 (Code Reasoning Model)

## 1. 项目目标

**一句话：训练一个1-3B参数的代码小模型，会自己推理、查资料、写代码。**

模型名称：CodeSapiens
作为OpenCode agent的"思考引擎"，通过vllm提供OpenAI兼容API。

---

## 2. 核心能力

| 能力 | 描述 |
|------|------|
| 推理 | 遇到问题先思考，分析需求 |
| 查资料 | 遇到不懂的，主动调用RAG查询 |
| 写代码 | 根据推理和查到的资料生成代码 |

---

## 3. 技术方案

### 3.1 模型架构

**论文依据**: arXiv:2602.12078 - Tiny Recursive Reasoning with Mamba-2 Attention Hybrid
- 作者: Wenlong Wang, Fergal Reid（康奈尔大学）
- 发布时间: 2026年2月

**TR-mamba2attn 结构**（每层）:
```
Mamba2Block → Mamba2Block → Mamba2Attention → MLP
```

| 参数 | 论文参考值 | 我们的目标(1-3B) |
|------|-----------|------------------|
| d_model | 384 | 512-768 |
| n_heads | 6 | 8-12 |
| d_state | 128 | 128 |
| head_dim | 64 | 64 |
| 展开因子 | 2 | 2 |
| 参数量 | 6.86M | ~100-300M |

**主要改动**:
- 当前 `MambaHybridBlock`: Mamba → Mamba → Light Attention → MLP
- 目标 `TR-mamba2attn`: Mamba2 → Mamba2 → Mamba2Attention → MLP
- 核心区别: Mamba2Attention 用SSM处理QKV，更适合长序列

**特点**: 递归推理能力，适合抽象逻辑任务

### 3.2 工具调用

- **方式**: 输出中显式调用 `<tool_call>search_rag(query)</tool_call>`
- **RAG**: 模型主动查询，自主决策时机
- **部署**: vllm提供OpenAI兼容API

### 3.3 训练数据

- **来源**: LLM合成生成
- **格式**: JSONL，每条包含 question + reasoning + tool_calls + tool_results + answer
- **生成脚本**: `generate_training_data.py`

### 3.4 硬件

- **训练**: CMP 90HX (10G显存) / RTX 3080
- **推理**: 消费级GPU即可

---

## 4. 工作流程

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  生成数据   │ ──▶ │   训练模型   │ ──▶ │  部署上线   │
└─────────────┘     └──────────────┘     └─────────────┘
     │                    │                    │
     ▼                    ▼                    ▼
generate_             train.py             vllm serve
training_data.py
```

---

## 5. 文件清单

| 文件 | 作用 |
|------|------|
| model.py | 核心模块（RMSNorm, RoPE, MambaSSD, HybridBlock） |
| extreme_reasoning_model.py | 完整语言模型 |
| generate_training_data.py | 训练数据生成脚本 |
| generate_prompt.txt | 数据生成Prompt |
| train.py | 训练脚本 |
| vllm部署 | 通过API提供服务 |

---

## 6. 下一步行动

### 阶段一：数据准备
1. **获取LLM API Key**（用于生成训练数据）
2. **运行 generate_training_data.py** 生成训练数据

### 阶段二：模型改造
3. **修改 model.py** - 实现 TR-mamba2attn 架构
   - 将 MambaSSD 升级为 Mamba2Block
   - 实现 Mamba2Attention（QKV用SSM处理）
4. **修改 extreme_reasoning_model.py** - 适配新块
5. **修改 train.py** - 支持新数据格式

### 阶段三：训练部署
6. **开始训练** - 在10G显存上
7. **部署测试** - 用vllm提供API

---

## 7. 关键技术决策（已确定）

- [x] 主控制器角色
- [x] 显式tool_call标签
- [x] 模型主动RAG查询
- [x] LLM合成训练数据
- [x] 1-3B参数规模
- [x] TR-mamba2attn架构
- [x] vllm部署
- [x] 消费级GPU训练

---

*创建日期: 2026-03-08*
