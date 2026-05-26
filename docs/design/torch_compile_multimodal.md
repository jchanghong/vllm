# 使用多模态编码器的 torch.compile

`torch.compile` 现在可以应用于 vLLM 中的多模态编码器和各种 nn 模块，包括视觉语言模型如 LLaMA 4、Qwen-VL，
以及类似的基于编码器的架构。

本文档涵盖了 `torch.compile` 集成在 vLLM 中如何为多模态编码器工作的基础知识，以及如何将装饰器
应用于新模型以提升性能。

!!! note
    关于 vLLM 中 `torch.compile` 集成的通用信息，请参见 [torch.compile 设计文档](./torch_compile.md)。

## 概述

我们最近使 `@support_torch_compile` 装饰器能够处理模型类型中的多个 nn 模块组件；这使得
可以为多模态编码器启用编译，为堆栈的其他组件带来性能提升。

当应用于 [`Qwen2_5_vl`](https://github.com/vllm-project/vllm/pull/23207) 的视觉块时，我们观察到约 4.5% 的端到端性能提升，
同时编译时间有所增加。

此功能默认关闭，但可以通过在编译配置中设置 `compile_mm_encoder: true` 来启用，前提是模型具有
`@support_torch_compile` 装饰器。

## 多模态组件的编译方式

### 启用 API

要编译多模态组件（例如编码器），我们遵循与 LLM 文本主干相同的机制，只需添加一些额外的脚手架：

1. `@support_torch_compile` 装饰器应包含 `enable_if=should_torch_compile_mm_encoder`。这将把编译的控制权交给我们的
`compile_mm_encoder` 配置。

2. `@support_torch_compile` 装饰器应对编码器组件包含 `is_encoder=True`。这是编译范围集成所需要的
（参见编译范围集成）。装饰器自动使用类名作为缓存目录前缀，避免
独立编译的子模块（例如视觉编码器组件与文本主干）之间的冲突。

### CompilationConfig

除了 `compile_mm_encoder: true` 之外，多模态编码器将继承与文本 LLM 相同的编译配置。我们将来可能会扩展
此配置以支持更多设置。

## 将 torch.compile 应用于新的多模态模型/组件

要将 `support_torch_compile` 应用于新的通用 nn.Module，我们建议遵循 [`debug_vllm_compile`](./debug_vllm_compile.md) 中的相同步骤；这包括：

1. 首先在小型模块（如基本 MLP 层）上应用 `support_torch_compile`，然后逐步提升到更通用的模块，直到达到良好的性能
权衡点。

2. 利用 [`tlparse`](https://github.com/meta-pytorch/tlparse) 识别并消除重新编译和 graph 断裂的来源。

3. 使用 `dynamic_arg_dims` 和适当的 `dynamic_shapes_config` 来处理动态性。

### 常见陷阱

## VllmBackend 功能支持

### 编译范围

torch.compile 集成将尝试依赖 max_batch_size 来推断动态形状的编译范围；然而，对于编码器中使用的模块，由于编码器可能看到的输入形状范围不确定，这个形状可能难以推断。因此，我们依赖于 `@support_torch_compile` 装饰器中的 `is_encoder=True` 来提醒 torch.compile 此范围无法推断，我们默认使用范围 (1, MAX_INT)。

!!! note
    我们将来可能会缩小此范围以获得更好的性能。

### Cudagraphs

我们尚未探索多模态编码器与 CUDAGraph 集成的编译；行为目前未指定。

## 故障排除

### 视觉编码器中的 Graph 断裂

某些视觉编码器操作可能导致 graph 断裂。要识别它们：

```bash
TORCH_LOGS="+dynamo" vllm serve <MODEL>
```

多模态模型中 graph 断裂的常见原因：

- **动态图像大小**：使用 `dynamic_shapes_config` 处理可变分辨率
- **不可追踪的操作**：某些操作（如 to_list）可能不受 Dynamo 支持
- **条件处理**：基于图像属性的数据相关分支

### 编译错误

如果多模态模型的编译失败：

1. **禁用并测试**：首先验证模型在不编译的情况下是否工作：
   ```bash
   VLLM_TORCH_COMPILE_LEVEL=0 vllm serve <model> --compilation-config='{"compile_mm_encoder":"false"}'
   ```

2. **检查日志**：启用调试日志以查看编译详情：
   ```bash
   VLLM_LOGGING_LEVEL=DEBUG vllm serve <model> --compilation-config='{"compile_mm_encoder":"true"}'
   ```

3. **报告问题**：如果您发现错误，[在 GitHub 上提交 issue](https://github.com/vllm-project/vllm/issues/new/choose)

## 参见

- [torch.compile 集成](./torch_compile.md) - 核心设计文档
- [调试 torch.compile](./debug_vllm_compile.md) - 详细调试指南
- [多模态输入](../features/multimodal_inputs.md) - 如何传递多模态数据
- [分离式编码器](../features/disagg_encoder.md) - 扩展视觉编码器
- [支持的多模态模型](../models/supported_models.md#list-of-multimodal-language-models) - 模型兼容性
