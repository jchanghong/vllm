# 常见问题解答

> 问：如何使用 OpenAI API 在单个端口上服务多个模型？

答：假设您指的是使用兼容 OpenAI 的服务器同时服务多个模型，目前不支持此功能。您可以同时运行多个服务器实例（每个实例服务不同的模型），并使用另一层将传入请求路由到相应的服务器。

---

> 问：离线推理嵌入应该使用哪个模型？

答：您可以尝试 [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) 和 [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)；更多模型请参见[此处](../models/supported_models.md)。

通过提取隐藏状态，vLLM 可以自动将文本生成模型（如 [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)、[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)）转换为嵌入模型，但它们的性能预计不如专门针对嵌入任务训练的模型。

---

> 问：在 vLLM 中，同一个提示的输出会因运行而异吗？

答：是的，可能会。vLLM 不保证输出令牌的对数概率（logprobs）稳定。对数概率的变化可能由于 Torch 操作中的数值不稳定性，或批处理 Torch 操作在批处理变化时的非确定性行为。更多详情，请参见[数值精度部分](https://pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations)。

在 vLLM 中，相同的请求可能因其他并发请求、批处理大小变化或推测解码中的批处理扩展等因素而以不同方式批处理。这些批处理变化，加上 Torch 操作的数值不稳定性，可能导致每一步的 logit/logprob 值略有不同。这种差异会累积，可能导致采样到不同的令牌。一旦采样到不同的令牌，进一步的分歧就更可能发生。

## 缓解策略

- 为获得更好的稳定性和更低的方差，请使用 `float32`。请注意，这将需要更多内存。
- 如果使用 `bfloat16`，切换到 `float16` 也有帮助。
- 使用请求种子有助于在 temperature > 0 时实现更稳定的生成，但由于精度差异导致的偏差仍可能发生。
