# vLLM-Project/Speculators

![用户流程浅色模式](../../assets/features/speculative_decoding/speculators-user-flow-light.svg#only-light)
![用户流程深色模式](../../assets/features/speculative_decoding/speculators-user-flow-dark.svg#only-dark)

[Speculators](https://docs.vllm.ai/projects/speculators/en/latest/) 是一个通过投机解码加速 LLM 推理的库，提供高效的草稿模型训练，与 vLLM 无缝集成，以降低延迟并提高吞吐量。

Speculators 提供以下关键功能：

- **使用 vLLM 进行离线训练数据生成**：支持使用 vLLM 生成隐藏状态。数据样本保存到磁盘，可用于草稿模型训练。
- **草稿模型训练支持**：端到端训练支持单层和多层草稿模型。同时支持非 MoE 和 MoE 模型的训练。
- **标准化、可扩展的格式**：提供 Hugging Face 兼容的格式用于定义推测模型，并附带工具，可从外部研究仓库转换为标准 Speculators 格式，便于采用。
- **无缝 vLLM 集成**：专为直接部署到 vLLM 而构建，以最小的开销实现低延迟、生产级推理。

## 为什么使用 Speculators？

大语言模型一次生成一个 token，这造成了一个根本性瓶颈：每个 token 都需要经过模型的完整前向传播，导致 GPU 计算在等待内存绑定操作时未被充分利用。
投机解码通过使用更小、更快的"草稿"模型（通常只是一个 transformer 层）预测多个后续 token，然后与主模型并行验证 token 来解决这一问题。

投机解码提供以下优势：

- **降低延迟**：对于聊天机器人和代码助手等交互式应用，生成速度提高 2-3 倍，响应时间直接影响用户体验。
- **更好的 GPU 利用率**：将大模型中的延迟和内存绑定解码转换为计算绑定的并行 token 验证，提高硬件利用率。
- **无质量损失**：投机解码不会近似目标模型。被接受的 token 正是目标模型在相同采样配置下本应生成的 token；被拒绝的草稿 token 会被丢弃并由目标模型重新生成。
- **成本效益**：通过减少每个请求占用硬件的时间，实现每个 GPU 服务更多请求。

Speculators 特别适用于对延迟敏感的应用程序，即用户正在实时等待响应的场景，例如对话式 AI、交互式编码助手和流式文本生成。

## 资源

- [Speculators 示例](https://github.com/vllm-project/speculators/tree/main/examples)
- [GitHub 仓库](https://github.com/vllm-project/speculators)
