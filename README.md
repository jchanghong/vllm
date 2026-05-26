<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
为每个人提供简单、快速、廉价的 LLM 服务
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>文档</b></a> | <a href="https://blog.vllm.ai/"><b>博客</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>论文</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>用户论坛</b></a> | <a href="https://slack.vllm.ai"><b>开发者 Slack</b></a> |
</p>

🔥 我们已建立一个 vLLM 网站，帮助您快速上手 vLLM。请访问 [vllm.ai](https://vllm.ai) 了解更多。
活动方面，请访问 [vllm.ai/events](https://vllm.ai/events) 加入我们。

---

## 关于

vLLM 是一个快速且易于使用的 LLM 推理和服务库。

最初由加州大学伯克利分校的 [Sky Computing Lab](https://sky.cs.berkeley.edu) 开发，vLLM 已发展成为最活跃的开源 AI 项目之一，由来自 2000 多名贡献者、数十家学术机构和公司的多元化社区构建和维护。

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
- 高吞吐量服务，支持各种解码算法，包括 *并行采样*、*束搜索* 等
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

支持的完整模型列表请参见[此处](https://docs.vllm.ai/en/latest/models/supported_models.html)。

## 快速开始

使用 [`uv`](https://docs.astral.sh/uv/)（推荐）或 `pip` 安装 vLLM：

```bash
uv pip install vllm
```

或[从源码构建](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source)用于开发。

访问我们的[文档](https://docs.vllm.ai/en/latest/)了解更多。

- [安装](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [快速入门](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [支持的模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html)

## 贡献

我们欢迎并重视任何贡献和合作。请查看[为 vLLM 做贡献](https://docs.vllm.ai/en/latest/contributing/index.html)了解如何参与。

## 引用

如果您在研究中使用了 vLLM，请引用我们的[论文](https://arxiv.org/abs/2309.06180)：

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## 联系我们

<!-- --8<-- [start:contact-us] -->
- 技术问题和功能请求，请使用 GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- 与其他用户讨论，请使用 [vLLM 论坛](https://discuss.vllm.ai)
- 协调贡献和开发，请使用 [Slack](https://slack.vllm.ai)
- 安全信息披露，请使用 GitHub 的 [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) 功能
- 合作与伙伴关系，请通过 [collaboration@vllm.ai](mailto:collaboration@vllm.ai) 联系我们
<!-- --8<-- [end:contact-us] -->

## 媒体资源

- 如果您希望使用 vLLM 的标识，请参考[我们的媒体资源仓库](https://github.com/vllm-project/media-kit)
