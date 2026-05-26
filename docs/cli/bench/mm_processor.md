# vllm bench mm-processor

## 概述

`vllm bench mm-processor` 对视觉语言模型的多模态输入处理器流水线进行性能分析。它测量从 HuggingFace 处理器到编码器前向传递的每个阶段延迟，帮助你识别预处理瓶颈，并理解不同图像分辨率或项目数量如何影响端到端请求时间。

该基准测试支持两种数据源：合成随机多模态输入（`random-mm`）和 HuggingFace 数据集（`hf`）。在测量前会运行预热请求以确保结果稳定。

## 快速开始

```bash
vllm bench mm-processor \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-name random-mm \
  --num-prompts 50 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
```

## 测量阶段

| 阶段 | 描述 |
| ----- | ----------- |
| `get_mm_hashes_secs` | 对多模态输入进行哈希处理的时间 |
| `get_cache_missing_items_secs` | 查找处理器缓存的时间 |
| `apply_hf_processor_secs` | HuggingFace 处理器中的处理时间 |
| `merge_mm_kwargs_secs` | 合并多模态 kwargs 的时间 |
| `apply_prompt_updates_secs` | 更新提示 token 的时间 |
| `preprocessor_total_secs` | 预处理总时间 |
| `encoder_forward_secs` | 编码器模型前向传递时间 |
| `num_encoder_calls` | 每个请求的编码器调用次数 |

该基准测试还会报告每个请求的端到端延迟（TTFT + 解码时间）。使用 `--metric-percentiles` 选择要报告的百分位数（默认：p99），使用 `--output-json` 保存结果。

更多示例（HF 数据集、预热、JSON 输出），请参见[基准测试 CLI — 多模态处理器基准测试](../../benchmarking/cli.md#多模态处理器基准测试)。

## JSON CLI 参数

--8<-- "docs/cli/json_tip.inc.md"

## 参数

--8<-- "docs/generated/argparse/bench_mm_processor.inc.md"
