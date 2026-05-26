# 可重现性

vLLM 默认不保证结果的可重现性，这是出于性能考虑。要实现可重现的结果：

- 在离线模式下，您可以设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 使调度具有确定性，或者启用[批处理不变性](../features/batch_invariance.md)使输出对调度不敏感。
- 在在线模式下，您只能启用[批处理不变性](../features/batch_invariance.md)。

示例：[examples/features/batch_invariance/reproducibility_offline.py](../../examples/features/batch_invariance/reproducibility_offline.py)

!!! warning

    设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 将改变用户代码（即构造 [LLM][vllm.LLM] 类的代码）的随机状态。

!!! note

    即使使用上述设置，vLLM 仅在相同硬件和相同 vLLM 版本上运行时才提供可重现性。

## 设置全局种子

vLLM 中的 `seed` 参数用于控制各种随机数生成器的随机状态。

如果提供了特定的种子值，`random`、`np.random` 和 `torch.manual_seed` 的随机状态将相应设置。

### 默认行为

在 V1 中，`seed` 参数默认为 `0`，这会设置每个工作进程的随机状态，因此即使 `temperature > 0`，每次 vLLM 运行的结果也将保持一致。

无法为 V1 取消指定种子，因为不同的工作进程需要对相同输出进行采样，以便用于推测解码等工作流程。更多信息请参见：<https://github.com/vllm-project/vllm/pull/17929>

!!! note

    用户代码（即构造 [LLM][vllm.LLM] 类的代码）中的随机状态仅在工作进程与用户代码在同一个进程中运行时才会被 vLLM 更新，即：`VLLM_ENABLE_V1_MULTIPROCESSING=0`。

    默认情况下 `VLLM_ENABLE_V1_MULTIPROCESSING=1`，因此您可以使用 vLLM 而无需担心意外使依赖于随机状态的后续操作变得确定。
