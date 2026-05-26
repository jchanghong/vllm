# 基础模型

本指南将引导您完成实现一个基础 vLLM 模型的步骤。

## 1. 引入您的模型代码

首先，从源代码仓库克隆 PyTorch 模型代码。
例如，vLLM 的 [OPT 模型](../../../vllm/model_executor/models/opt.py) 改编自
HuggingFace 的 [modeling_opt.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py) 文件。

!!! warning
    请务必审查并遵守原始代码的版权和许可条款！

## 2. 使您的代码与 vLLM 兼容

为了确保与 vLLM 的兼容性，您的模型必须满足以下要求：

### 初始化代码

模型中所有 vLLM 模块必须在构造函数中包含一个 `prefix` 参数。这个 `prefix` 通常是模块在模型状态字典中的完整名称，对于以下方面至关重要：

- 运行时支持：vLLM 的注意力算子通过其完整名称在模型状态中注册。每个注意力算子必须有一个唯一的 `prefix` 作为其层名称，以避免冲突。
- 非均匀量化支持：量化检查点可以选择性地量化某些层，同时保持其他层为全精度。通过在初始化时提供 `prefix`，vLLM 可以将当前层的 `prefix` 与量化配置匹配，以确定该层是否应以量化模式初始化。

初始化代码应如下所示：

??? code

    ```python
    from torch import nn
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention import Attention

    class MyAttention(nn.Module):
        def __init__(self, vllm_config: VllmConfig, prefix: str):
            super().__init__()
            self.attn = Attention(prefix=f"{prefix}.attn")

    class MyDecoderLayer(nn.Module):
        def __init__(self, vllm_config: VllmConfig, prefix: str):
            super().__init__()
            self.self_attn = MyAttention(prefix=f"{prefix}.self_attn")

    class MyModel(nn.Module):
        def __init__(self, vllm_config: VllmConfig, prefix: str):
            super().__init__()
            self.layers = nn.ModuleList(
                [MyDecoderLayer(vllm_config, prefix=f"{prefix}.layers.{i}") for i in range(vllm_config.model_config.hf_config.num_hidden_layers)]
            )

    class MyModelForCausalLM(nn.Module):
        def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
            super().__init__()
            self.model = MyModel(vllm_config, prefix=f"{prefix}.model")
    ```

### 计算代码

- 在 `MyModel` 模块内部添加一个 `embed_input_ids` 方法，该方法根据给定的 `input_ids` 返回文本嵌入。这相当于直接调用文本嵌入层，但在 `MyModel` 用于复合多模态模型时提供了统一的接口。

```python
class MyModel(nn.Module):
        ...

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        ... 
```

- 重写模型的 [forward][torch.nn.Module.forward] 方法，移除任何不必要的代码，例如训练专用代码。修改输入参数，将 `input_ids` 和 `positions` 视为具有单个批次大小维度的扁平化张量，而无需最大序列长度维度。

```python
def forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

!!! note
    目前，vLLM 支持基础的多头注意力机制及其带旋转位置嵌入的变体。
    如果您的模型采用了不同的注意力机制，您将需要在 vLLM 中实现一个新的注意力层。

作为参考，请查看我们的 [Llama 实现](../../../vllm/model_executor/models/llama.py)。vLLM 已经支持了大量模型。建议找到一个与您的模型相似的模型，并根据您的模型架构进行适配。查看 [vllm/model_executor/models](../../../vllm/model_executor/models) 获取更多示例。

## 3. （可选）实现张量并行和量化支持

如果您的模型太大而无法放入单个 GPU，您可以使用张量并行来管理它。
为此，请将模型的线性层和嵌入层替换为它们的张量并行版本。
对于嵌入层，您可以简单地将 [torch.nn.Embedding][] 替换为 `VocabParallelEmbedding`。对于输出 LM head，您可以使用 `ParallelLMHead`。
对于线性层，我们提供以下选项来并行化它们：

- `ReplicatedLinear`：在多个 GPU 上复制输入和权重。不节省内存。
- `RowParallelLinear`：输入张量沿隐藏维度分割。权重矩阵沿行（输入维度）分割。矩阵乘法后执行 *all-reduce* 操作以归约结果。通常用于第二个 FFN 层和注意力层的输出线性变换。
- `ColumnParallelLinear`：输入张量被复制。权重矩阵沿列（输出维度）分割。结果沿列维度分割。通常用于第一个 FFN 层和原始 Transformer 中注意力层分离的 QKV 变换。
- `MergedColumnParallelLinear`：合并多个 `ColumnParallelLinear` 算子的列并行线性层。通常用于具有加权激活函数（例如 SiLU）的第一个 FFN 层。该类处理多个权重矩阵的分片权重加载逻辑。
- `QKVParallelLinear`：用于多头和分组查询注意力机制的查询、键和值投影的并行线性层。当键/值头数小于世界大小时，该类会正确地复制键/值头。该类处理权重矩阵的加载和复制。

请注意，上述所有线性层都将 `linear_method` 作为输入。vLLM 将根据不同的量化方案设置此参数以支持权重量化。

## 4. 实现权重加载逻辑

您现在需要在 `*ForCausalLM` 类中实现 `load_weights` 方法。
此方法应从 HuggingFace 的检查点文件加载权重，并将它们分配到模型中对应的层。具体来说，对于 `MergedColumnParallelLinear` 和 `QKVParallelLinear` 层，如果原始模型有分离的权重矩阵，您需要分别加载不同的部分。

## 5. 注册您的模型

请参阅[此页面](registration.md)了解如何注册您的新模型以供 vLLM 使用。

## 常见问题

### 如何支持交错滑动窗口的模型？

为了支持具有交错滑动窗口的模型，我们需要注意以下细节：

- 确保模型的 `config.json` 包含 `layer_types`。
- 在建模代码中，为每一层解析正确的滑动窗口值，并将其传递给注意力层的 `per_layer_sliding_window` 参数。作为参考，请查看[这一行](https://github.com/vllm-project/vllm/blob/996357e4808ca5eab97d4c97c7d25b3073f46aab/vllm/model_executor/models/llama.py#L171)。

完成这两步后，交错滑动窗口应该可以在模型中正常工作。

### 如何支持使用 Mamba 的模型？

我们考虑 3 种不同的场景：

1. 使用 Mamba 层（Mamba-1 或 Mamba-2）但不使用注意力层的模型。
2. 将 Mamba 层（Mamba-1 或 Mamba-2）与注意力层结合使用的模型。
3. 将类 Mamba 机制（例如线性注意力、ShortConv）与注意力层结合使用的模型。

对于情况 (1)，我们建议参考 [`MambaForCausalLM`](../../../vllm/model_executor/models/mamba.py)（适用于 Mamba-1）或 [`Mamba2ForCausalLM`](../../../vllm/model_executor/models/mamba2.py)（适用于 Mamba-2）的实现。
模型应继承协议 `IsAttentionFree`，并实现类方法 `get_mamba_state_dtype_from_config` 和 `get_mamba_state_shape_from_config`，以根据配置计算状态形状和数据类型。
对于 mamba 层本身，请使用 [`MambaMixer`](../../../vllm/model_executor/layers/mamba/mamba_mixer.py)（适用于 Mamba-1）或 [`MambaMixer2`](../../../vllm/model_executor/layers/mamba/mamba_mixer2.py)（适用于 Mamba-2）类。
模型还应添加到 [vllm/model_executor/models/config.py](../../../vllm/model_executor/models/config.py) 中的 `MODELS_CONFIG_MAP` 字典中，以确保运行时默认值得到优化。

对于情况 (2)，我们建议使用 [`JambaForCausalLM`](../../../vllm/model_executor/models/jamba.py)（适用于 Mamba-1 和注意力结合的模型示例）或 [`BambaForCausalLM`](../../../vllm/model_executor/models/bamba.py)（适用于 Mamba-2 和注意力结合的模型示例）的实现作为参考。
这些模型应遵循与情况 (1) 相同的指示，但应继承协议 `IsHybrid`（而不是 `IsAttentionFree`），并且*不需要*将它们添加到 `MODELS_CONFIG_MAP` 中（它们的运行时默认值将从协议中推断）。

对于情况 (3)，我们建议参考 [`MiniMaxText01ForCausalLM`](../../../vllm/model_executor/models/minimax_text_01.py) 或 [`Lfm2ForCausalLM`](../../../vllm/model_executor/models/lfm2.py) 的实现，它们分别使用了自定义的"类 Mamba"层 `MiniMaxText01LinearAttention` 和 `ShortConv`。
在实现这些模型时，请遵循与情况 (2) 相同的指南。
我们使用"类 Mamba"来指代那些具有原地更新状态的层，而不是追加状态（就像注意力机制的 KV 缓存一样）。
为了实现新的自定义类 Mamba 层，应继承 `MambaBase` 并实现方法 `get_state_dtype`、`get_state_shape` 以在运行时计算数据类型和状态形状，以及 `mamba_type` 和 `get_attn_backend`。
还需要实现"注意力元数据"类，该类处理所有层共有的元数据。
请参阅 [`LinearAttentionMetadata`](../../../vllm/v1/attention/backends/linear_attn.py) 或 [`ShortConvAttentionMetadata`](../../../vllm/v1/attention/backends/short_conv_attn.py) 了解相关示例。
还需要注意，在添加新的 mamba 后端时，我们应该更新 [`registry.py`](../../../vllm/v1/attention/backends/registry.py) 中的 `MambaAttentionBackendEnum`。
最后，如果要支持 torch compile 和 CUDA graphs，需要将对类 Mamba 层的调用包装在自定义算子内部并进行注册。
请参阅 [vllm/model_executor/models/minimax_text_01.py](../../../vllm/model_executor/models/minimax_text_01.py) 或 [vllm/model_executor/layers/mamba/short_conv.py](../../../vllm/model_executor/layers/mamba/short_conv.py) 中对 `direct_register_custom_op` 的调用作为示例。
然后应将新的自定义算子添加到 [vllm/config/compilation.py](../../../vllm/config/compilation.py) 的 `_attention_ops` 列表中，以确保分段 CUDA graphs 按预期工作。
