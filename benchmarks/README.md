# 基准测试

此目录曾包含 vLLM 的基准测试脚本和工具，用于性能测试和评估。

## 内容

- **服务基准测试**：用于测试在线推理性能（延迟、吞吐量）的脚本
- **吞吐量基准测试**：用于测试离线批处理推理性能的脚本
- **专项基准测试**：用于测试特定功能的工具，如结构化输出、前缀缓存、长文档问答、请求优先级排序和多模态推理
- **数据集工具**：用于从各种基准测试数据集（ShareGPT、HuggingFace 数据集、合成数据等）加载和采样的框架

## 使用方法

有关详细的使用说明、示例和数据集信息，请参见[基准测试 CLI 文档](https://docs.vllm.ai/en/latest/benchmarking/cli/#benchmark-cli)。

完整的 CLI 参考请参见：

- <https://docs.vllm.ai/en/latest/cli/bench/latency.html>
- <https://docs.vllm.ai/en/latest/cli/bench/serve.html>
- <https://docs.vllm.ai/en/latest/cli/bench/throughput.html>
