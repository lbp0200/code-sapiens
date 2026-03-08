# Code Sapiens - TR-mamba2attn Code Reasoning Model

> ⚠️ **Experimental** - This is an experimental project under active development.

## Overview

Code Sapiens is a code reasoning model based on the TR-mamba2attn architecture, designed to serve as the "thinking engine" for the OpenCode agent. The model reasons about problems, queries RAG when needed, and generates code.

## Architecture

TR-mamba2attn (based on [arXiv:2602.12078](https://arxiv.org/abs/2602.12078)):

```
Mamba2Block → Mamba2Block → Mamba2Attention → SwiGLU MLP
```

### Components

- **Mamba2Block**: State Space Duality (SSD) block implementing Mamba-2's algorithm
- **Mamba2Attention**: SSM-based attention for QKV processing
- **RMSNorm**: Root Mean Square Layer Normalization
- **RoPE**: Rotary Position Embedding (supports up to 128K)

## Specifications

| Parameter | Value |
|-----------|-------|
| Parameters | 9.1B |
| Context Length | 128K |
| Hidden Size (d_model) | 2560 |
| Layers | 48 |
| Attention Heads | 16 |
| State Dimension (d_state) | 128 |

## Hardware Requirements

Optimized for **CMP 90HX (10GB VRAM)** with INT4 AWQ quantization:

| Component | Memory |
|-----------|--------|
| Model Weights (INT4) | ~2.2GB |
| KV Cache (128K) | ~3.8GB |
| **Total** | **~6GB** |

## Installation

```bash
pip install torch
```

## Usage

```python
from extreme_reasoning_model import ExtremeReasoningModel

model = ExtremeReasoningModel(
    d_model=2560,
    n_layers=48,
    n_heads=16,
    d_state=128,
    max_seq_len=131072,
)

# Forward pass
logits = model(input_ids)
```

## Training

```bash
# Generate training data (requires LLM API key)
python generate_training_data.py

# Train the model
python train.py
```

## Deployment

After training, convert to AWQ format and serve with vLLM:

```bash
# Export model
torch.save(model.state_dict(), "model.pt")

# Quantize with vLLM
vllm convert --model model.pt --output_dir ./awq_model --quantization awq

# Serve
vllm serve ./awq_model --quantization awq --tensor-parallel-size 1
```

## Limitations

⚠️ **This is an experimental project:**

- Architecture is novel and untested at scale
- Training pipeline is still in development
- Model quality not validated
- May have bugs in SSM implementations

## License

MIT

## References

- [arXiv:2602.12078](https://arxiv.org/abs/2602.12078) - TR-mamba2attn
- [arXiv:2405.20960](https://arxiv.org/abs/2405.20960) - Mamba-2 SSD
- [Mamba: Linear-time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
