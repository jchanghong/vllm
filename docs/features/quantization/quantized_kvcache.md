# 量化 KV Cache

## FP8 KV Cache 概述

高效的内存使用对于大型语言模型的工作至关重要。将 KV (Key-Value) cache 量化到 FP8 格式可以显著减少其内存占用。这一优化使您能够在内存中存储更多 token，从而提高吞吐量并支持更长的上下文窗口。

> **注意：** 当使用 Flash Attention 3 后端配合 FP8 KV cache 时，注意力运算也在量化 (FP8) 域中执行。在此配置下，除了键和值之外，查询也会被量化到 FP8。

### 支持的 FP8 KV-Cache 量化方案

vLLM 支持两种主要的 FP8 KV-cache 量化策略：

- **Per-tensor 量化：**  
  对每个 Q、K、V 张量分别应用单一的 scale。 (`q/k/v_scale = [1]`)
- **Per-attention-head 量化：**  
  每个 scale 对应一个注意力头：`q_scale = [num_heads]`，`k/v_scale = [num_kv_heads]`。

> **注意：**  
> Per-attention-head 量化目前**仅适用于 Flash Attention 后端**，并且需要 **llm-compressor** 提供的校准流程。

### Scale 校准方法

您可以通过三种不同的方式在 vLLM 中配置量化 scale 的计算：

1. **无校准（默认 scales）：**  
   所有量化 scale 设置为 `1.0`。  
   _配置方式：_  
   ```python
   kv_cache_dtype="fp8"
   calculate_kv_scales=False
   ```

2. **随机 token 校准（即时计算）：**  
   Scale 在预热期间从单个随机 token 批次中自动估算并固定。  
   _配置方式：_  
   ```python
   kv_cache_dtype="fp8"
   calculate_kv_scales=True
   ```

3. **[推荐] 使用数据集校准（通过 llm-compressor）：**  
   使用精心挑选的校准数据集来估算 scale，以获得最高精度。  
   这需要 [llm-compressor](https://github.com/vllm-project/llm-compressor) 库。  
   _请参见下面的示例！_

#### 其他 `kv_cache_dtype` 选项

- `kv_cache_dtype="auto"`：使用模型的默认数据类型
- `kv_cache_dtype="fp8_e4m3"`：支持 CUDA 11.8+ 和 ROCm (AMD GPU)
- `kv_cache_dtype="fp8_e5m2"`：支持 CUDA 11.8+

---

## 示例

### 1. 无校准 (`kv_cache_dtype="fp8"`, `calculate_kv_scales=False`)

所有量化 scale 均设置为 1.0。

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=False,
)
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```

---

### 2. 随机 token 校准 (`kv_cache_dtype="fp8"`, `calculate_kv_scales=True`)

Scale 在预热期间从单个 token 批次中自动估算。

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True,
)
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```

---

### 3. **[推荐] 使用数据集校准（通过 `llm-compressor`）**

为了获得最高质量的量化，我们建议使用 `llm-compressor` 对数据集进行校准。这可以实现每注意力头量化等高级策略。

#### 安装所需包

```bash
pip install llmcompressor
```

#### 示例：将 Llama 的 Attention 和 KV Cache 量化到 FP8

```python
"""
使用 llm-compressor 的一次性校准，将 Llama 的 Attention + KV cache 量化到 FP8（选择 'tensor' 或 'attn_head' 策略）。
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

# -----------------------------
# 配置
# -----------------------------
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
STRATEGY = "tensor"       # 或 "attn_head"
NUM_CALIB_SAMPLES = 512   # 良好的起始值
MAX_SEQ_LEN = 2048

# -----------------------------
# 辅助函数
# -----------------------------
def process_and_tokenize(example, tokenizer: AutoTokenizer):
    """将聊天消息转换为 token。"""
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        add_special_tokens=False,
    )

def build_recipe(strategy: str) -> QuantizationModifier:
    fp8_args = QuantizationArgs(num_bits=8, type="float", strategy=strategy)
    return QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=["LlamaAttention"],  # 量化查询：q_scale
                input_activations=fp8_args,
            )
        },
        kv_cache_scheme=fp8_args,           # 量化 KV cache：k/v_scale
    )

# -----------------------------
# 主函数
# -----------------------------
def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIB_SAMPLES}]")
    ds = ds.shuffle(seed=42)
    ds = ds.map(
        lambda ex: process_and_tokenize(ex, tokenizer),
        remove_columns=ds.column_names,
    )

    recipe = build_recipe(STRATEGY)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_CALIB_SAMPLES,
    )

    save_dir = f"{MODEL_ID.rstrip('/').split('/')[-1]}-kvattn-fp8-{STRATEGY}"
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()
```

有关更多详细和最新的示例，请参见 [`llm-compressor` 官方示例](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_kv_cache)。
