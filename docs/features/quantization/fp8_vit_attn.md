# FP8 ViT Encoder Attention

对于需要处理大图像（例如 QHD、4K）且文本提示/生成相对较短的视觉理解工作负载，ViT encoder 注意力可能成为显著的瓶颈，尤其是在文本模型已被量化的情况下（例如 NVFP4）。vLLM 支持通过 FlashInfer cuDNN 后端对 ViT encoder 注意力进行可选的 FP8 量化。Q/K/V 在 cuDNN 注意力调用之前被即时量化为 FP8。

!!! note
    - 目前仅支持 Qwen3-VL 系列模型（`qwen3_vl`、`qwen3_vl_moe`、`qwen3_5`、`qwen3_5_moe` 以及其他使用 Qwen3 ViT 的模型）。
    - 动态缩放与 ViT 全 CUDA graphs 不兼容。
    - 性能提升主要在 QHD/4K 分辨率或多图像请求时可见。由于量化开销（3 次量化内核启动 + 去填充），较小的图像可能看不到加速效果。
    - FP8 tensor-core 加速在 GB300 上比 GB200 更显著。

## 要求

- FlashInfer cuDNN 后端，cuDNN >= 9.17.1。

## 使用方法

通过传递 `--mm-encoder-attn-dtype fp8` 配合 `--mm-encoder-attn-backend FLASHINFER` 来启用 FP8 ViT 注意力：

```bash
vllm serve $MODEL \
    --mm-encoder-attn-backend FLASHINFER \
    --mm-encoder-attn-dtype fp8
```

默认情况下（无 scale 文件），使用**动态缩放**：一个包含 16 个条目的循环缓冲区，用于存储观察到的 Q/K/V amax 值，驱动每次前向传播时的 scale 更新。这与 BF16 精度匹配，无需任何校准，但会带来少量每次前向传播的开销。

## 一次性校准、重复使用工作流（推荐）

对于生产环境，请在代表性数据集上一次性校准静态 scales 并重复使用，以避免动态开销：

```bash
# 步骤 1：校准并保存 scales（动态缩放运行 16 次，
# 然后将学习到的 scales 转储到 JSON 文件）。
vllm bench mm-processor \
    --model $MODEL --mm-encoder-attn-backend FLASHINFER \
    --mm-encoder-attn-dtype fp8 \
    --mm-encoder-fp8-scale-save-path /path/to/scales.json \
    --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat \
    --num-prompts 100

# 步骤 2：使用静态 scales 提供服务（无动态开销）。
vllm serve $MODEL \
    --mm-encoder-attn-backend FLASHINFER \
    --mm-encoder-attn-dtype fp8 \
    --mm-encoder-fp8-scale-path /path/to/scales.json
```

保存的 scales 会乘以 `--mm-encoder-fp8-scale-save-margin`（默认 `1.5`），以留出余量应对校准集中未出现的激活值异常。默认值已验证可在不同数据集上通用（例如，在 VisionArena-Chat 上校准的模型在 ChartQA 上仍能保持 BF16 精度）。

## Scale 文件格式

```json
{
    "visual.blocks.0.attn.attn": {"q": 224.0, "k": 198.0, "v": 210.0},
    "visual.blocks.1.attn.attn": {"q": 218.0, "k": 195.0, "v": 207.0}
}
```

也接受使用 `q_scale` / `k_scale` / `v_scale` 作为键的别名。

## 性能

**核心 cuDNN 注意力内核**（PyTorch profiler，`cudnn_generated_fort_native_sdpa_sm100_flash_fprop`，head_dim=128，seq_len=8192）：

| 硬件 | BF16 | FP8 | 加速比 |
| -------- | ---- | ---- | ------- |
| GB200 | 350 us | 312 us | **1.12x** |
| GB300 | 300 us | 211 us | **1.42x** |

**端到端 encoder 前向时间**（Qwen3-VL-30B-A3B-Instruct 在 GB200 上，每请求 3 张图像）：

| 分辨率 | BF16 中位数 | FP8 中位数 | 加速比 |
| ---------- | ----------- | ---------- | ------- |
| HD (720x1280) | 31.77 ms | 36.39 ms | 0.87x |
| FullHD (1080x1920) | 57.99 ms | 58.73 ms | ~相同 |
| QHD (1440x2560) | 131.83 ms | 122.30 ms | **1.08x** |
| 4K (2160x3840) | 543.44 ms | 460.31 ms | **1.18x** |

性能交叉点大约在 FullHD 分辨率且每请求 3 张图像时。在 QHD 及以上分辨率，FP8 胜出。

## 精度

ChartQA，Qwen3-VL-8B-Instruct，500 个样本。FP8 静态使用在 VisionArena-Chat 上校准的 scales（默认 1.5x 余量）：

| 指标 | BF16 | FP8 动态 | FP8 静态 |
| ------ | ---- | ----------- | ---------- |
| relaxed_accuracy | 0.780 | 0.776 | 0.780 |
| anywhere_accuracy | 0.806 | 0.816 | 0.814 |
| exact_match | 0.584 | 0.582 | 0.578 |

所有三种配置在统计噪声范围内保持一致，证实了在一个数据集上校准的静态 scales 可以泛化到另一个数据集。
