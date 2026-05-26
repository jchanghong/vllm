# vLLM V1

!!! announcement

    我们已完全弃用 V0。请阅读 [RFC #18571](https://github.com/vllm-project/vllm/issues/18571) 了解更多详情。

    如果您有在 V0 引擎上运行但在 V1 上无法运行的用例，请在 [GitHub](https://github.com/vllm-project/vllm) 或 [vLLM Slack](https://inviter.co/vllm-slack) 上分享。

vLLM V0 成功支持了广泛的模型和硬件，但随着新功能的独立开发，系统变得越来越复杂。这种复杂性使得集成新功能变得更加困难，并引入了技术债务，暴露出对更精简和统一设计的需求。

基于 V0 的成功，vLLM V1 保留了 V0 中稳定且经过验证的组件（如模型、GPU 内核和实用工具）。同时，它对核心系统进行了显著重构，涵盖了调度器、KV 缓存管理器、工作进程、采样器和 API 服务器，以提供一个凝聚、可维护的框架，更好地适应持续增长和创新。

具体来说，V1 的目标是：

- 提供**简单、模块化且易于修改的代码库**。
- 确保**高性能**，接近零 CPU 开销。
- 将**关键优化**结合到统一的架构中。
- 默认启用功能/优化，实现**零配置**。

我们看到了升级到 V1 核心引擎带来的显著性能提升，特别是在长上下文场景中。请参见性能基准测试（待添加）。

更多详情，请查看 vLLM V1 博客文章[vLLM V1：vLLM 核心架构的重大升级](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)（发表于 2025 年 1 月 27 日）。

这份实时用户指南概述了 vLLM V1 引入的一些已知**重要变更和限制**。团队一直在积极致力于将 V1 设为默认引擎，因此本指南将随着 V1 上更多功能的支持而不断更新。

## 与 V0 的差异

本节列出了 V0 和 V1 之间的一些行为差异。

### 分块预填充

分块预填充在可能的情况下默认启用，而在 V0 中，它是根据模型特征有条件地启用的。

### CUDA 图

V1 中的 CUDA 图捕获比 V0 占用更多内存。

### 对数概率的语义变更

#### 对数概率计算

默认情况下，V1 中的 logprobs 在从模型的原始输出计算后立即返回（即在对数后处理（如温度缩放或惩罚调整）之前）。因此，返回的 logprobs 不反映采样时使用的最终调整概率。

您可以通过设置 `--logprobs-mode` 标志来调整此行为。支持四种模式：`raw_logprobs`（默认）、`processed_logprobs`、`raw_logits`、`processed_logits`。Raw 表示应用任何 logit 处理器（如禁用词）之前的值。Processed 表示应用所有处理器（包括温度和 top_k/top_p）之后的值。

#### 带前缀缓存的提示对数概率

虽然 V1 支持在启用前缀缓存的情况下传递提示 logprobs，但它不再缓存 logprobs。对于需要提示 logprobs 的请求，引擎将忽略前缀缓存并重新计算完整提示的预填充以生成 logprobs。

## 功能支持

对于每个项目，其在 vLLM V1 中的支持状态属于以下之一：

- **🟢 可用**：完全可用，优化程度与 V0 相当或更优。
- **🟡 进行中**：计划纳入 vLLM V1，有开放的 PR/RFC。
- **🔴 已移除**：已从 vLLM V1 中删除。只有在有强烈需求时才会考虑重新引入。

!!! note
    vLLM V1 的统一调度器通过使用简单的字典（例如 `{request_id: num_tokens}`）动态分配每个请求的固定令牌预算，以相同的方式处理提示和输出令牌，从而在没有严格分离预填充和解码阶段的情况下支持分块预填充、前缀缓存和推测解码等功能。

V1 调度器支持多种调度策略，包括先到先服务（FCFS）和基于优先级的调度（根据分配的优先级处理请求，FCFS 作为平局决胜），可通过 `--scheduling-policy` 参数配置。

### 硬件

| 硬件           | 状态            |
| -------------- | --------------- |
| **NVIDIA**     | <nobr>🟢</nobr> |
| **AMD**        | <nobr>🟢</nobr> |
| **INTEL GPU**  | <nobr>🟢</nobr> |
| **TPU**        | <nobr>🟢</nobr> |
| **CPU**        | <nobr>🟢</nobr> |

!!! note

    更多硬件平台可能通过插件支持，例如：

    - [vllm-ascend](https://github.com/vllm-project/vllm-ascend)
    - [vllm-spyre](https://github.com/vllm-project/vllm-spyre)
    - [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)
    - [vllm-openvino](https://github.com/vllm-project/vllm-openvino)

    请查看其对应的仓库以获取更多详情。

### 模型

| 模型类型                     | 状态                                      |
| --------------------------- | ----------------------------------------- |
| **仅解码器模型**            | <nobr>🟢</nobr>                           |
| **编码器-解码器模型**       | <nobr>🟢（Whisper），🔴（其他）</nobr>     |
| **池化模型**                | <nobr>🟢</nobr>                           |
| **Mamba 模型**              | <nobr>🟢</nobr>                           |
| **多模态模型**              | <nobr>🟢</nobr>                           |

请参见下方，了解尚未支持或在 V1 中有更多功能规划的模型状态。

#### 池化模型

现已完全支持，新的最后池化模型可使用前缀缓存和分块预填充。

我们正在努力为更多类别的池化模型启用前缀缓存和分块预填充。

#### Mamba 模型

使用选择性状态空间机制而非标准 Transformer 注意力的模型得到支持。使用 Mamba-2 和 Mamba-1 层的模型（例如 `Mamba2ForCausalLM`、`MambaForCausalLM`、`FalconMambaForCausalLM`）得到支持。

将 Mamba-2 和 Mamba-1 层与标准注意力层结合的混合模型也得到支持（例如 `BambaForCausalLM`、`Zamba2ForCausalLM`、`NemotronHForCausalLM`、`FalconH1ForCausalLM` 和 `GraniteMoeHybridForCausalLM`、`JambaForCausalLM`、`Plamo2ForCausalLM`）。

具有不同于 Mamba 机制的混合模型也得到支持（例如 `MiniMaxText01ForCausalLM`、`MiniMaxM1ForCausalLM`、`Lfm2ForCausalLM`）。

请注意，上述所有模型尚未支持前缀缓存。

#### 编码器-解码器模型

Whisper 原生支持。其他编码器-解码器模型通过插件系统支持：

- **BART**：`BartForConditionalGeneration` 通过官方 [bart-plugin](https://github.com/vllm-project/bart-plugin) 支持。
- **Florence-2**：`Florence2ForConditionalGeneration` 通过官方 [bart-plugin](https://github.com/vllm-project/bart-plugin) 支持。

对于其他编码器-解码器模型（例如 `MllamaForConditionalGeneration`），我们建议通过[插件系统](../design/plugin_system.md)遵循类似模式来实现支持。

### 功能

| 功能                                        | 状态                                                                                |
| ------------------------------------------- | ----------------------------------------------------------------------------------- |
| **前缀缓存**                                | <nobr>🟢 可用</nobr>                                                               |
| **分块预填充**                              | <nobr>🟢 可用</nobr>                                                               |
| **LoRA**                                    | <nobr>🟢 可用</nobr>                                                               |
| **对数概率计算**                            | <nobr>🟢 可用</nobr>                                                               |
| **FP8 KV 缓存**                             | <nobr>🟢 可用</nobr>                                                               |
| **推测解码**                                | <nobr>🟢 可用</nobr>                                                               |
| **带前缀缓存的提示对数概率**                | <nobr>🟢 可用</nobr>                                                               |
| **结构化输出替代后端**                      | <nobr>🟢 可用</nobr>                                                               |
| **并发部分预填充**                          | <nobr>🟡 [进行中](https://github.com/vllm-project/vllm/issues/14003)</nobr>         |
| **best_of**                                 | <nobr>🔴 [已移除](https://github.com/vllm-project/vllm/issues/13361)</nobr>         |
| **每个请求的 Logits 处理器**                | <nobr>🔴 [已移除](https://github.com/vllm-project/vllm/pull/13360)</nobr>           |
| **GPU <> CPU KV 缓存交换**                  | <nobr>🔴 已移除</nobr>                                                             |
| **请求级结构化输出后端**                    | <nobr>🔴 已移除</nobr>                                                             |

!!! note

    vLLM V1 的统一调度器通过使用简单的字典（例如 `{request_id: num_tokens}`）动态分配每个请求的固定令牌预算，以相同的方式处理提示和输出令牌，从而在没有严格分离预填充和解码阶段的情况下支持分块预填充、前缀缓存和推测解码等功能。

#### 已移除的功能

作为 vLLM V1 重大架构重构的一部分，一些遗留功能已被移除。

##### 采样功能

- **best_of**：由于使用有限，此功能已被移除。详情请见 [RFC #13361](https://github.com/vllm-project/vllm/issues/13361)。
- **每个请求的 Logits 处理器**：在 V0 中，用户可以传递自定义处理函数以在每个请求基础上调整 logits。在 vLLM V1 中，此功能已被移除。取而代之的是，我们现在支持在启动时设置的**全局 logits 处理器**，请参见 [RFC #17799](https://github.com/vllm-project/vllm/issues/17799)。

##### KV 缓存功能

- **GPU <> CPU KV 缓存交换**：使用新的简化核心架构，vLLM V1 不再需要 KV 缓存交换来处理请求抢占。

##### 结构化输出功能

- **请求级结构化输出后端**：已移除；现在支持带有回退的替代后端（outlines、guidance）。
