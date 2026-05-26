# Intel 量化支持

[AutoRound](https://github.com/intel/auto-round) 是 Intel 为大型语言模型 (LLM) 设计的高级量化算法。它可以生成高效的 **INT2、INT3、INT4、INT8、MXFP8、MXFP4、NVFP4** 和 **GGUF** 量化模型，在精度和推理性能之间取得平衡。AutoRound 也是 [Intel® Neural Compressor](https://github.com/intel/neural-compressor) 的一部分。如需更深入的介绍，请参阅 [AutoRound 逐步指南](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md)。

## 主要特性

✅ **卓越精度** 即使在 2-3 位精度下也能提供强劲性能 [示例模型](https://huggingface.co/collections/OPEA/2-3-bits)

✅ **快速混合 `Bits`/`Dtypes` 方案生成** 数分钟内自动配置

✅ **支持导出 AutoRound、AutoAWQ、AutoGPTQ 和 GGUF 格式**

✅ **支持 10+ 视觉语言模型 (VLM)**

✅ **逐层混合位量化** 实现精细控制

✅ **RTN (Round-To-Nearest) 模式** 快速量化，精度损失较小

✅ **多种量化方案**：最佳 (best)、基础 (base) 和轻量 (light)

✅ **高级工具**，如即时打包和支持 **10+ 后端**

## Intel 平台支持的方案

在 Intel 平台上，AutoRound 方案正按格式和硬件逐步启用。目前，vLLM 支持：

- **`W4A16`**：仅权重量化，4 位权重搭配 16 位激活
- **`W8A16`**：仅权重量化，8 位权重搭配 16 位激活

更多方案和格式将在未来版本中支持。

## 量化模型

### 安装

```bash
uv pip install auto-round
```

### 使用 CLI 量化

```bash
auto-round \
    --model Qwen/Qwen3-0.6B \
    --scheme W4A16 \
    --format auto_round \
    --output_dir ./tmp_autoround
```

### 使用 Python API 量化

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "Qwen/Qwen3-0.6B"
autoround = AutoRound(model_name, scheme="W4A16")

# 最佳精度，慢 4-5 倍，low_gpu_mem_usage 可节省约 20G 但慢约 30%
# autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size, sym=sym)

# 2-3 倍加速，W4G128 时精度略有下降
# autoround = AutoRound(model, tokenizer, nsamples=128, iters=50, lr=5e-3, bits=bits, group_size=group_size, sym=sym )

output_dir = "./tmp_autoround"
# format= 'auto_round'(默认), 'auto_gptq', 'auto_awq'
autoround.quantize_and_save(output_dir, format="auto_round")
```

## 在 vLLM 中部署 AutoRound 量化模型

```bash
vllm serve Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096
```

!!! note
     要在 Intel GPU/CPU 上部署 `wNa16` 模型，目前请添加 `--enforce-eager`。

## 使用 vLLM 评估量化模型

```bash
lm_eval --model vllm \
  --model_args pretrained="Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enforce_eager=True" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size 128
```
