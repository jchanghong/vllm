# CustomOp

`CustomOp` 是一个抽象类，用于将各种操作的前向方法分派到适当的后端。它还提供了一种机制，供 vLLM 和 OOT（树外）插件注册其自定义操作。

本文档将介绍 CustomOp 在 vLLM 中的工作原理以及如何实现一个新的 `CustomOp`。

## CustomOp 在 vLLM 中的工作原理

`CustomOp` 在其类中管理两个字典，包含所有自定义操作（即操作类，按注册名称索引），分别用于 vLLM 和 OOT 插件。

我们可以使用 `@CustomOp.register("op_name")` 将一个操作类注册到 `CustomOp` 系统。之后，`op_name` 及其类将被添加到 `op_registry` 字典中。此外，我们还可以通过 `@CustomOp.register_oot("op_name")` 注册一个 OOT 操作。我们将在后面详细介绍这种机制。

当调用 `CustomOp` 时（即调用其 `forward()` 方法），如果它被启用（即带有 `--compilation_config.custom_ops '["+op_name"]'`），它将根据 `current_platform` 自动将前向方法分派到适当的后端。否则（即被禁用），它将只调用 `forward_native()` 方法来使用该前向方法的 PyTorch 原生实现。

- **CPU 平台：** 分派到 `forward_cpu()`。
- **CUDA 平台：** 分派到 `forward_cuda()`。
- **ROCm 平台：** 分派到 `forward_hip()`。如果未实现 `forward_hip()`，将使用 `forward_cuda()` 作为回退。
- **XPU 平台：** 分派到 `forward_xpu()`。
- **TPU 平台：** 分派到 `forward_tpu()`。
- **OOT 平台：** 分派到 `forward_oot()`。这仅会在 OOT 平台上调用。
- **默认：** 分派到 `forward_native()` 作为所有平台的最终回退。

!!! note
    注意，由于类继承的原因，分派逻辑可能不是绝对的。派生类可能覆盖此行为。

此外，vLLM 根据 `compilation_config.custom_ops` 决定启用或禁用 `CustomOp`。具体来说，如果 `CustomOp` 未在 `compilation_config.custom_ops` 中注册（即使用默认配置），则如果 `compilation_config.custom_ops` 包含 `all` 则启用，如果包含 `none` 则禁用。

!!! note
    注意，`all` 和 `none` 不能在 `compilation_config.custom_ops` 中共存。

默认情况下，如果 `compilation_config.backend == "inductor"` 且 `compilation_config.mode != CompilationMode.NONE`，则 `none` 会被附加到 `compilation_config.custom_ops` 中，否则附加 `all`。换句话说，这意味着 `CustomOp` 在某些平台（即那些在 `torch.compile` 运行时使用 `inductor` 作为默认后端的平台）上会在 torch 编译模式下被禁用。在这种情况下，Inductor 会为这些禁用的自定义操作生成（融合的）Triton 内核。

!!! note
    对于多模态模型，vLLM 强制启用了某些自定义操作，以便在 ViT 部分使用设备特定的深度优化内核以获得更好的性能，例如 `MMEncoderAttention` 和 `ApplyRotaryEmb`。我们也可以在 `CustomOp` 的 `__init__()` 方法中传递 `enforce_enable=True` 参数，以在对象级别强制启用自身。

    注意，此 `enforce_enable` 机制将在我们为多模态部分添加单独的 `compilation_config` 后移除。

## 如何自定义 CustomOp 的配置

vLLM 还为用户提供了细粒度的控制，以启用或禁用特定的自定义操作，方法是在启动服务器时手动传递 `--compilation_config.custom_ops '["..."]'`。

例如：

- 使用 `--compilation_config.custom_ops '["all"]'` 启用所有自定义操作。
- 使用 `--compilation_config.custom_ops '["none"]'` 禁用所有自定义操作。
- 使用 `--compilation_config.custom_ops '["all,-op1"]'` 启用除 op1 之外的所有自定义操作（即前缀为 `-` 表示"禁用"）。
- 使用 `--compilation_config.custom_ops '["none,+op1,+op2"]'` 仅启用 op1 和 op2（即前缀为 `+` 表示"启用"）。

## vLLM 中支持的自定义操作类型

**1. 注意力：**

```python
--8<-- "vllm/model_executor/layers/mla.py:multi_head_latent_attention"

```

**2. 激活：**

```python
--8<-- "vllm/model_executor/layers/activation.py:silu_and_mul"

--8<-- "vllm/model_executor/layers/activation.py:mul_and_silu"

--8<-- "vllm/model_executor/layers/activation.py:gelu_new"

--8<-- "vllm/model_executor/layers/activation.py:gelu_fast"

--8<-- "vllm/model_executor/layers/activation.py:quick_gelu"

--8<-- "vllm/model_executor/layers/activation.py:gelu_and_mul"

--8<-- "vllm/model_executor/layers/activation.py:gelu_and_mul_sparse"

--8<-- "vllm/model_executor/layers/activation.py:relu2"

--8<-- "vllm/model_executor/layers/activation.py:xielu"

--8<-- "vllm/model_executor/layers/activation.py:swigluoai_and_mul"

--8<-- "vllm/model_executor/layers/activation.py:fatrelu_and_mul"
```

**3. MM-Conv：**

```python
--8<-- "vllm/model_executor/layers/conv.py:conv2d"

--8<-- "vllm/model_executor/layers/conv.py:conv3d"
```

**4. 嵌入：**

```python
--8<-- "vllm/model_executor/layers/vocab_parallel_embedding.py:vocab_parallel_embedding"

--8<-- "vllm/model_executor/layers/vocab_parallel_embedding.py:parallel_lm_head"
```

**5. 线性：**

```python
--8<-- "vllm/model_executor/layers/linear.py:row_parallel_linear"

--8<-- "vllm/model_executor/layers/linear.py:column_parallel_linear"

--8<-- "vllm/model_executor/layers/linear.py:replicated_linear"
```

**6. Logits 处理器：**

```python
--8<-- "vllm/model_executor/layers/logits_processor.py:logits_processor"
```

**7. Mamba：**

```python
--8<-- "vllm/model_executor/layers/mamba/mamba_mixer.py:mamba_mixer"

--8<-- "vllm/model_executor/layers/mamba/mamba_mixer2.py:mamba_mixer2"

--8<-- "vllm/model_executor/layers/mamba/mamba_mixer2.py:mixer2_gated_rms_norm"

--8<-- "vllm/model_executor/models/plamo2.py:plamo2_mamba_mixer"

--8<-- "vllm/model_executor/layers/mamba/short_conv.py:short_conv"
```

**8. MoE：**

```python
--8<-- "vllm/model_executor/layers/fused_moe/layer.py:fused_moe"

--8<-- "vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py:modular_fused_moe"

--8<-- "vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:unquantized_fused_moe"

--8<-- "vllm/model_executor/models/transformers/moe.py:transformers_fused_moe"

--8<-- "vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:grouped_topk"
```

**9. 归一化：**

```python
--8<-- "vllm/model_executor/layers/layernorm.py:rms_norm"

--8<-- "vllm/model_executor/layers/layernorm.py:rms_norm_gated"

--8<-- "vllm/model_executor/layers/layernorm.py:gemma_rms_norm"
```

**10. 量化：**

```python
--8<-- "vllm/model_executor/layers/quantization/input_quant_fp8.py:quant_fp8"
```

**11. RoPE：**

```python
--8<-- "vllm/model_executor/layers/rotary_embedding/base.py:rotary_embedding"

--8<-- "vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py:dual_chunk_rotary_embedding"

--8<-- "vllm/model_executor/layers/rotary_embedding/common.py:apply_rotary_emb"
```

**12. 编码器：**

```python
--8<-- "vllm/model_executor/models/deepencoder2.py:qwen2_decoder"

--8<-- "vllm/model_executor/layers/attention/mm_encoder_attention.py:mm_encoder_attn"

--8<-- "vllm/model_executor/models/deepencoder.py:rel_pos_attention"
```

## 实现新 CustomOp 的指南

### 在 vLLM 中实现新的 CustomOp

本部分是如何在 vLLM 中实现新的 `CustomOp` 的教程。

步骤：

1. 实现一个新的操作类，该类继承自 `CustomOp` 基类。
2. 在此操作类上添加 `@CustomOp.register("op_name")` 装饰器，将其注册到 `CustomOp` 系统。
3. 根据需要实现不同的 `forward_xxx()` 方法。

以 `MMEncoderAttention` 为例：

??? code

    ```python
    @CustomOp.register("mm_encoder_attn")
    class MMEncoderAttention(CustomOp):

        def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float | None = None,
            num_kv_heads: int | None = None,
            prefix: str = "",
            multimodal_config: MultiModalConfig | None = None,
        ) -> None:
            super().__init__()
            # 初始化...

        def forward_native(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # 仅用于 Flash Attention
        ) -> torch.Tensor:
            # 调用 TORCH_SDPA 实现...

        def forward_cuda(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # 仅用于 Flash Attention
        ) -> torch.Tensor:
            # 调用 FA 或 TORCH_SDPA 实现...

        def forward_cpu(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # 仅用于 Flash Attention
        ) -> torch.Tensor:
            # 调用 TORCH_SDPA 实现...

        def forward_xpu(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # 仅用于 Flash Attention
        ) -> torch.Tensor:
            # 调用 FA 实现...

        def forward_tpu(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # 仅用于 Flash Attention
        ) -> torch.Tensor:
            # 调用 PALLAS 实现...
    ```

### 在 OOT 设备插件中注册新的 CustomOp

目前，得益于 [vLLM 的硬件插件机制](./plugin_system.md)，出现了各种 OOT 设备插件，使 vLLM 能够在不同硬件上无缝运行。您可以在[介绍 vLLM 硬件插件，Ascend NPU 最佳实践](https://blog.vllm.ai/2025/05/12/hardware-plugin.html)中找到关于此机制的更多细节。

- **官方设备插件：** [vllm-ascend](https://github.com/vllm-project/vllm-ascend)（用于华为昇腾 NPU）、[vllm-spyre](https://github.com/vllm-project/vllm-spyre)（用于 Spyre）、[vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)（用于 Intel Gaudi）、[vllm-neuron](https://github.com/vllm-project/vllm-neuron)（用于 AWS Neuron）、[vllm-meta](https://github.com/vllm-project/vllm-metal)（用于 Apple Silicon）等。
- **非官方设备插件：** [vllm-metax](https://github.com/MetaX-MACA/vLLM-metax)（用于 MetaX GPU）、[vllm-kunlun](https://github.com/baidu/vLLM-Kunlun)（用于百度昆仑 XPU）、[vllm-musa](https://github.com/MooreThreads/vllm-musa)（用于摩尔线程 GPU）等。

在这种情况下，`CustomOp` 使这些硬件制造商能够在运行时无缝地用其深度优化的内核替换 vLLM 的操作，只需注册一个 OOT `CustomOp` 并实现 `forward_oot()` 方法即可。

现在，本部分将展示如何为设备插件注册一个 OOT `CustomOp`。

以 `MMEncoderAttention` 为例：

1. 实现一个 `CustomMMEncoderAttention` 类，该类继承自 `MMEncoderAttention` 并实现其 `forward_oot()` 方法。
2. 将您的 `CustomMMEncoderAttention` 注册到 vLLM 以替换 `MMEncoderAttention`。

??? code

    ```python
    from vllm.model_executor.layers.attention import MMEncoderAttention
    from vllm.model_executor.custom_op import CustomOp


    @CustomOp.register_oot("MMEncoderAttention")
    class CustomMMEncoderAttention(MMEncoderAttention):

        def __init__(...):
            super().__init__(...)

        def forward_oot(...):
            # 调用优化的设备特定内核。
            ...
    ```

在这种情况下，一个新条目 `{"MMEncoderAttention": CustomMMEncoderAttention}` 将被添加到 `op_registry_oot` 中。当初始化 `MMEncoderAttention` 操作对象时，如果类名（即 `MMEncoderAttention`）包含在 `op_registry_oot` 的键中，vLLM 将用我们注册的类（即 `CustomMMEncoderAttention`）替换它并实例化。

之后，当调用此 `MMEncoderAttention` 操作时，如果它被启用，您的 `forward_oot()` 将被调用。因此，您将在您的硬件上获得预期的性能，而无需直接修改 vLLM。

此外，您还可以在一个地方注册所有 `CustomOp` 以进行更好的管理。

??? code

    ```python
    from vllm.model_executor.custom_op import CustomOp


    REGISTERED_CUSTOM_OPS = {
        "CustomOP1": YourCustomOp1,
        "CustomOP2": YourCustomOp2,
        "CustomOP3": YourCustomOp3,
    }

    for op_name, op_cls in REGISTERED_CUSTOM_OPS.items():
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=op_name)
    ```
