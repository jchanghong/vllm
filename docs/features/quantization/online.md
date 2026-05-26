# 在线量化

在线量化允许您获取一个 BF16/FP16 模型，并在加载时将其 Linear 和 MoE 权重量化到更低精度（例如 FP8），无需预量化的 checkpoint 或校准数据。权重在模型加载期间转换，激活值在每次前向传播中动态缩放。

## 快速开始

将方案名称传递给 `quantization` 参数：

```python
from vllm import LLM

# Per-tensor FP8 量化（每个权重张量一个 scale）
llm = LLM("meta-llama/Llama-3.1-8B", quantization="fp8_per_tensor")

# Per-block FP8 量化（128x128 block scale 用于权重，1x128 block scale 用于激活值）
llm = LLM("meta-llama/Llama-3.1-8B", quantization="fp8_per_block")

# MXFP8 量化用于权重和激活值
llm = LLM("meta-llama/Llama-3.1-8B", quantization="mxfp8")
```

或者通过 CLI：

```bash
vllm serve meta-llama/Llama-3.1-8B --quantization fp8_per_tensor
vllm serve meta-llama/Llama-3.1-8B --quantization fp8_per_block
vllm serve meta-llama/Llama-3.1-8B --quantization mxfp8
```

## 支持的方案

| 方案 | 权重配方 | 激活值配方 | 备注 |
| ------ | ------------- | ------------------ | ----- |
| `fp8_per_tensor` | fp8_e4m3 数据，fp32 per-tensor scale | fp8_e4m3 数据，fp32 per-tensor scale | 在某些 GPU（Ada、Hopper）上，线性激活值使用 per-token 缩放以获得更好性能 |
| `fp8_per_block` | fp8_e4m3 数据，fp32 per-128x128-block scale | fp8_e4m3 数据，fp32 per-1x128-block scale | |
| `mxfp8` | fp8_e4m3 数据，e8m0 per-1x32-block scale | fp8_e4m3 数据，e8m0 per-1x32-block scale | 需要 SM 100+（Blackwell 或更新）用于 w8a8，其他 GPU 使用 w8a16 回退 |

## 高级配置

如需精细控制，请使用 `quantization_config` 字典。

### 配置模式

```yaml
quantization_config:
  linear:
    weight: <名称>      # 参见 vllm/config/quantization.py 中的 QUANT_KEY_NAMES
    activation: <名称>
  moe:
    weight: <名称>
    activation: <名称>
  ignore: [<层名或正则表达式>, ...]
```

`linear` 和 `moe` 接受完整的 `{weight, activation}` 字典，或一个裸字符串。字符串首先解析为 `--quantization` 的简写（取其匹配的层类型槽位），然后作为权重名称在 `QUANT_KEY_NAMES` 中查找。未设置的字段将回退到 `--quantization` 简写的默认值，或者对于已量化的 checkpoint，回退到 checkpoint 声明的值。

CLI 接受 JSON 形式或点号键形式：

```bash
vllm serve <model> --quantization-config '{"moe":{"activation":"mxfp8"}}'
vllm serve <model> --quantization-config.moe.activation mxfp8
```

### 已量化 checkpoint 的激活值覆盖

对于 checkpoint 已量化的模型，`quantization_config` 允许您独立于内置权重选择激活格式。支持的覆盖范围因 checkpoint 而异；目前这已为 MXFP4 MoE checkpoint (gpt-oss) 配置，您可以选择使用 FP8 激活值：

```bash
vllm serve openai/gpt-oss-20b --quantization-config.moe.activation mxfp8
```

结合 `--moe-backend` 来固定特定的内核系列。

### 稠密层和 MoE 层的独立方案

您可以通过 `linear` 和 `moe` 字段对稠密线性层和 MoE 专家层应用不同的量化方案。每个字段可以接受完整的规范字典，或一个裸字符串指定在线简写名称（例如 `"fp8_per_block"`）或权重格式（例如 `"fp8_per_block_static"`）；未设置的字段将回退到简写的默认值。

```python
from vllm import LLM

# Linear：per-block FP8；MoE：per-tensor FP8（从简写继承）
llm = LLM(
    "ibm-granite/granite-3.0-1b-a400m-base",
    quantization="fp8_per_tensor",
    quantization_config={
        "linear": "fp8_per_block",
    },
)
```

或：

```python
from vllm import LLM

# Linear：per-tensor FP8（继承）；MoE：per-block FP8
llm = LLM(
    "ibm-granite/granite-3.0-1b-a400m-base",
    quantization="fp8_per_tensor",
    quantization_config={
        "moe": "fp8_per_block",
    },
)
```

### 从量化中排除层

使用 `ignore` 参数跳过特定层。它接受精确的层名称和正则表达式模式（以 `re:` 为前缀）：

```python
from vllm import LLM

llm = LLM(
    "ibm-granite/granite-3.0-1b-a400m-base",
    quantization="fp8_per_tensor",
    quantization_config={
        "ignore": [
            # 精确层名称
            "model.layers.1.self_attn.o_proj",
            # 正则表达式：跳过所有 QKV 投影
            "re:.*[qkv]_proj",
        ],
    },
)
```

!!! note
    对于融合层（例如融合了 `q_proj`、`k_proj`、`v_proj` 的 `qkv_proj`），忽略模式必须匹配**未融合**的分片名称（`q_proj`、`k_proj`、`v_proj`），而不是融合后的名称。
