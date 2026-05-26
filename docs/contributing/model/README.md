# 概述

!!! important
    许多解码器语言模型现在可以使用 [Transformers 建模后端](../../models/supported_models.md#transformers) 自动加载，而无需在 vLLM 中实现。请先尝试 `vllm serve <model>` 是否有效！

vLLM 模型是专门的 [PyTorch](https://pytorch.org/) 模型，利用各种 [特性](../../features/README.md#compatibility-matrix) 来优化其性能。

将模型集成到 vLLM 的复杂性在很大程度上取决于模型的架构。
如果模型与 vLLM 中已有的模型架构相似，则过程会相当简单。
然而，对于包含新算子（例如，新的注意力机制）的模型，这个过程可能会更加复杂。

请阅读以下页面以获取逐步指南：

- [基础模型](basic.md)
- [注册模型](registration.md)
- [单元测试](tests.md)
- [多模态支持](multimodal.md)
- [语音转文本支持](transcription.md)

!!! tip
    如果在将模型集成到 vLLM 时遇到问题，欢迎提交 [GitHub issue](https://github.com/vllm-project/vllm/issues)
    或在我们的 [开发者 Slack](https://slack.vllm.ai) 上提问。
    我们将很乐意为您提供帮助！
