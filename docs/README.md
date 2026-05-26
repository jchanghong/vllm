---
hide:
  - navigation
  - toc
---

# 欢迎使用 vLLM

<figure markdown="span">
  ![](./assets/logos/vllm-logo-text-light.png){ align="center" alt="vLLM Light" class="logo-light" width="60%" }
  ![](./assets/logos/vllm-logo-text-dark.png){ align="center" alt="vLLM Dark" class="logo-dark" width="60%" }
</figure>

<p style="text-align:center">
<strong>为每个人提供简单、快速、廉价的 LLM 服务
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/vllm-project/vllm" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/vllm-project/vllm/subscription" data-show-count="true" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/vllm-project/vllm/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>

vLLM 是一个快速且易于使用的 LLM 推理和服务库。

最初由加州大学伯克利分校的 [Sky Computing Lab](https://sky.cs.berkeley.edu) 开发，vLLM 已发展成为最活跃的开源 AI 项目之一，由来自 2000 多名贡献者、数十家学术机构和公司的多元化社区构建和维护。

根据用户类型不同，vLLM 的入门方式也有所不同。如果您希望：

- 在 vLLM 上运行开源模型，我们建议从[快速入门指南](./getting_started/quickstart.md)开始
- 使用 vLLM 构建应用程序，我们建议从[用户指南](./usage/README.md)开始
- 构建 vLLM 本身，我们建议从[开发者指南](./contributing/README.md)开始

关于 vLLM 的开发信息，请参阅：

- [路线图](https://roadmap.vllm.ai)
- [发布版本](https://github.com/vllm-project/vllm/releases)

vLLM 的快速体现在：

- 最先进的服务吞吐量
- 使用 [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html) 高效管理注意力键值内存
- 传入请求的连续批处理、分块预填充、前缀缓存
- 通过分段和完整 CUDA/HIP 图实现快速灵活的模型执行
- 量化：FP8、MXFP8/MXFP4、NVFP4、INT8、INT4、GPTQ/AWQ、GGUF、compressed-tensors、ModelOpt、TorchAO 以及[更多](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- 优化的注意力内核，包括 FlashAttention、FlashInfer、TRTLLM-GEN、FlashMLA 和 Triton
- 使用 CUTLASS、TRTLLM-GEN、CuTeDSL 针对各种精度的优化 GEMM/MoE 内核
- 推测解码，包括 n-gram、suffix、EAGLE、DFlash
- 使用 torch.compile 的自动内核生成和图级变换
- 分离式预填充、解码和编码

vLLM 的灵活性和易用性体现在：

- 与流行的 Hugging Face 模型无缝集成
- 高吞吐量服务，支持各种解码算法，包括*并行采样*、*束搜索*等
- 张量、流水线、数据、专家和上下文并行用于分布式推理
- 流式输出
- 使用 xgrammar 或 guidance 生成结构化输出
- 工具调用和推理解析器
- 兼容 OpenAI 的 API 服务器，以及 Anthropic Messages API 和 gRPC 支持
- 高效的稠密和 MoE 层多 LoRA 支持
- 支持 NVIDIA GPU、AMD GPU 以及 x86/ARM/PowerPC CPU。此外还有多种硬件插件，如 Google TPU、Intel Gaudi、IBM Spyre、华为昇腾、Rebellions NPU、Apple Silicon、MetaX GPU 等。

vLLM 无缝支持 Hugging Face 上的 200 多种模型架构，包括：

- 仅解码器 LLM（例如 Llama、Qwen、Gemma）
- 混合专家 LLM（例如 Mixtral、DeepSeek-V3、Qwen-MoE、GPT-OSS）
- 混合注意力和状态空间模型（例如 Mamba、Qwen3.5）
- 多模态模型（例如 LLaVA、Qwen-VL、Pixtral）
- 嵌入和检索模型（例如 E5-Mistral、GTE、ColBERT）
- 奖励和分类模型（例如 Qwen-Math）

支持的完整模型列表请参见[此处](./models/supported_models.md)。

更多信息，请查看以下内容：

- [vLLM 宣布博客文章](https://blog.vllm.ai/2023/06/20/vllm.html)（PagedAttention 介绍）
- [vLLM 论文](https://arxiv.org/abs/2309.06180)（SOSP 2023）
- [连续批处理如何在 LLM 推理中实现 23 倍吞吐量同时降低 p50 延迟](https://www.anyscale.com/blog/continuous-batching-llm-inference) - Cade Daniel 等
- [vLLM 见面会](community/meetups.md)
