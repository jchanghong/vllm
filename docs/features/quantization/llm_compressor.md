# LLM Compressor

[LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/en/latest/) 是一个用于优化模型以在 vLLM 上部署的库。
它提供了一套全面的量化算法，包括对 FP4、FP8、INT8 和 INT4 量化等技术的支持。

## 为什么使用 LLM Compressor？

现代 LLM 通常包含数十亿个以 16 位或 32 位浮点数存储的参数，需要大量的 GPU 内存，限制了部署选择。
量化通过将模型权重和激活的精度降低到更小的数据类型，在保持推理输出质量的同时降低了内存需求。

LLM Compressor 提供以下优势：

- **减少内存占用**：在更小的 GPU 上运行更大的模型。
- **降低推理成本**：每个 GPU 服务更多并发用户，直接降低生产部署中每次查询的成本。
- **更快的推理**：更小的数据类型意味着消耗更少的内存带宽，这通常会转化为更高的吞吐量，特别是对于内存密集型工作负载。

LLM Compressor 处理量化、校准和格式转换的复杂性，生成可直接在 vLLM 中使用的模型。

## 主要特性

- **多种量化算法**：支持 AWQ、GPTQ、AutoRound 和 Round-to-Nearest。
还包括对 QuIP 和 SpinQuant 风格变换以及 KV 缓存和注意力量化的支持。
- **多种量化方法**：支持 FP8、INT8、INT4、NVFP4、MXFP4 和混合精度量化
- **单次量化 (One-Shot Quantization)**：使用最少的校准数据快速量化模型
- **vLLM 集成**：使用 compressed-tensors 格式在 vLLM 中无缝部署量化模型
- **Hugging Face 兼容性**：与 Hugging Face Hub 中的模型兼容

## 资源

- [LLM Compressor 示例](https://github.com/vllm-project/llm-compressor/tree/main/examples)
- [GitHub 仓库](https://github.com/vllm-project/llm-compressor)
