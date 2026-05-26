# Transformers 强化学习

[Transformers 强化学习](https://huggingface.co/docs/trl)（TRL）是一个全栈库，提供了一套用于训练 transformer 语言模型的工具，支持监督微调（SFT）、组相对策略优化（GRPO）、直接偏好优化（DPO）、奖励建模等方法。该库与 🤗 transformers 集成。

GRPO 或在线 DPO 等在线方法需要模型生成补全内容。vLLM 可用于生成这些补全内容！

有关更多信息，请参阅 TRL 文档中的 [vLLM 集成指南](https://huggingface.co/docs/trl/main/en/vllm_integration)。

TRL 目前支持以下使用 vLLM 的在线训练器：

- [GRPO](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [在线 DPO](https://huggingface.co/docs/trl/main/en/online_dpo_trainer)
- [RLOO](https://huggingface.co/docs/trl/main/en/rloo_trainer)
- [Nash-MD](https://huggingface.co/docs/trl/main/en/nash_md_trainer)
- [XPO](https://huggingface.co/docs/trl/main/en/xpo_trainer)

要在 TRL 中启用 vLLM，请将训练器配置中的 `use_vllm` 标志设置为 `True`。

## 训练期间使用 vLLM 的模式

TRL 支持**两种模式**在训练期间集成 vLLM：**服务器模式**和**共置模式**。您可以通过 `vllm_mode` 参数控制 vLLM 在训练期间的运行方式。

### 服务器模式

在**服务器模式**下，vLLM 作为独立进程在专用 GPU 上运行，并通过 HTTP 请求与训练器通信。当您拥有用于推理的独立 GPU 时，此配置是理想选择，因为它将生成工作负载与训练隔离开，确保了稳定的性能和更轻松的扩展。

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="server",  # 默认值，可省略
)
```

### 共置模式

在**共置模式**下，vLLM 在训练器进程内部运行，并与训练模型共享 GPU 内存。这避免了启动单独的服务器，可以提高 GPU 利用率，但可能导致训练 GPU 上的内存争用。

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="colocate",
)
```

某些训练器还支持 **vLLM 休眠模式**，该模式在训练期间将参数和缓存卸载到 GPU RAM，有助于减少内存使用。了解更多信息，请参阅[内存优化文档](https://huggingface.co/docs/trl/main/en/reducing_memory_usage#vllm-sleep-mode)。

!!! info
    有关详细的配置选项和标志，请参阅您正在使用的特定训练器的文档。
