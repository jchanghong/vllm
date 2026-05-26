# 性能基准测试描述

## 延迟测试

- 输入长度：32 个 token。
- 输出长度：128 个 token。
- 批大小：固定 (8)。
- GPU/HPU 模型：llama-3.1 8B、llama-3 70B、mixtral 8x7B。
- CPU 模型：llama-3.1 8B。
- 评估指标：端到端延迟（平均值、中位数、p99）。

{latency_tests_markdown_table}

## 吞吐量测试

- 输入长度：从 ShareGPT 数据集中随机采样 200 个提示（使用固定随机种子）。
- 输出长度：这 200 个提示对应的输出长度。
- 批大小：由 vllm 动态确定以达到最大吞吐量。
- GPU/HPU 模型：llama-3.1 8B、llama-3 70B、mixtral 8x7B。
- CPU 模型：llama-3.1 8B。
- 评估指标：吞吐量。

{throughput_tests_markdown_table}

## 服务测试

- 输入长度：从 ShareGPT 数据集中随机采样 200 个提示（使用固定随机种子）。
- 输出长度：这 200 个提示对应的输出长度。
- 批大小：由 vllm 和请求的到达模式动态确定。
- **平均 QPS（每秒查询数）**：1、4、16 和 inf。QPS = inf 表示所有请求同时到达。对于其他 QPS 值，每个查询的到达时间使用随机泊松过程确定（使用固定随机种子）。
- GPU/HPU 模型：llama-3.1 8B、llama-3 70B、mixtral 8x7B。
- 我们还为 GPU 上的 llama-3 70B 添加了一个推测解码测试，QPS 为 2。
- CPU 模型：llama-3.1 8B。
- 评估指标：吞吐量、TTFT（首 token 时间，含平均值、中位数和 p99）、ITL（token 间延迟，含平均值、中位数和 p99）。
- 对于 CPU，我们添加了随机数据集测试，使用 100 个提示测试固定输入/输出长度。

{serving_tests_markdown_table}

## 平台信息

{platform_markdown_table}

## 基准测试表格的 JSON 版本

本节包含上述 Markdown 表格的 JSON 格式数据。
您可以通过以下方式将基准测试表格加载到 pandas 数据框中：

```python
import json
import pandas as pd

benchmarking_results_json = """JSON 字符串"""
benchmarking_results = json.loads(benchmarking_results_json)
latency_results = pd.DataFrame.from_dict(benchmarking_results["latency"])
throughput_results = pd.DataFrame.from_dict(benchmarking_results["throughput"])
serving_results = pd.DataFrame.from_dict(benchmarking_results["serving"])
```

所有基准测试表格的 JSON 字符串：

```json
{benchmarking_results_in_json_string}
```

您还可以在 Buildkite 页面的 Artifact 选项卡中查看原始实验数据。
