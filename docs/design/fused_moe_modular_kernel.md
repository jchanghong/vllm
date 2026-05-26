# 融合 MoE 模块化内核

## 介绍

FusedMoEModularKernel 在[此处](../../vllm/model_executor/layers/fused_moe/modular_kernel.py)实现。

根据输入激活的格式，FusedMoE 实现大致分为 2 种类型：

* 连续 / 标准 / 非批处理，以及
* 批处理

!!! note
    术语连续（Contiguous）、标准（Standard）和非批处理（Non-Batched）在本文档中可互换使用。

输入激活格式完全取决于所使用的 All2All 调度。

* 在连续变体中，All2All 调度将激活作为形状为 (M, K) 的连续张量返回，以及形状为 (M, num_topk) 的 TopK ID 和 TopK 权重。请参见 `DeepEPHTPrepareAndFinalize` 的示例。
* 在批处理变体中，All2All 调度将激活作为形状为 (num_experts, max_tokens, K) 的张量返回。在这里，订阅同一专家的激活/token 被批处理在一起。请注意，并非张量的所有条目都是有效的。激活张量通常附带一个大小为 `num_experts` 的 `expert_num_tokens` 张量，其中 `expert_num_tokens[i]` 表示订阅第 i 个专家的有效 token 数。请参见 `DeepEPLLPrepareAndFinalize` 的示例。

FusedMoE 操作通常由多个操作组成，在连续和批处理变体中都是如此，如下图所示：

![FusedMoE 非批处理](../assets/design/fused_moe_modular_kernel/fused_moe_non_batched.png)

![FusedMoE 批处理](../assets/design/fused_moe_modular_kernel/fused_moe_batched.png)

!!! note
    在操作方面，批处理和非批处理情况之间的主要区别在于 Permute / Unpermute 操作。所有其他操作保持不变。

## 动机

从图中可以看出，有很多操作，并且每个操作可能有各种各样的实现。将这些操作组合成有效的 FusedMoE 实现的方式迅速变得难以处理。模块化内核框架通过将操作分组为逻辑组件来解决这个问题。这种广泛分类使组合变得可管理，并防止代码重复。这也将 All2All 调度和组合实现与 FusedMoE 实现解耦，允许它们独立开发和测试。此外，模块化内核框架为不同组件引入了抽象类，从而为未来的实现提供了良好定义的框架。

本文档的其余部分将专注于连续/非批处理的情况。外推到批处理情况应该是直接的。

## ModularKernel 组件

FusedMoEModularKernel 将 FusedMoE 操作分为 3 个部分：

1. TopKWeightAndReduce
2. FusedMoEPrepareAndFinalizeModular
3. FusedMoEExpertsModular

### TopKWeightAndReduce

TopK 权重应用和规约组件发生在 Unpermute 操作之后、All2All Combine 之前。注意，`FusedMoEExpertsModular` 负责 Unpermute，而 `FusedMoEPrepareAndFinalizeModular` 负责 All2All Combine。在 `FusedMoEExpertsModular` 中进行 TopK 权重应用和规约是有价值的。但有些实现选择在 `FusedMoEPrepareAndFinalizeModular` 中完成。为了实现这种灵活性，我们有一个 TopKWeightAndReduce 抽象类。

请在此[处](../../vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py)找到 TopKWeightAndReduce 的实现。

`FusedMoEPrepareAndFinalizeModular::finalize()` 方法接受一个 `TopKWeightAndReduce` 参数，该方法内部调用该参数。
`FusedMoEModularKernel` 充当 `FusedMoEExpertsModular` 和 `FusedMoEPrepareAndFinalize` 实现之间的桥梁，以确定 TopK 权重应用和规约发生的位置。

* 如果 `FusedMoEExpertsModular` 实现自己完成权重应用和规约，`FusedMoEExpertsModular::finalize_weight_and_reduce_impl` 方法返回 `TopKWeightAndReduceNoOp`。
* 如果 `FusedMoEExpertsModular` 实现需要 `FusedMoEPrepareAndFinalizeModular::finalize()` 来完成权重应用和规约，`FusedMoEExpertsModular::finalize_weight_and_reduce_impl` 方法返回 `TopKWeightAndReduceContiguous` / `TopKWeightAndReduceNaiveBatched` / `TopKWeightAndReduceDelegate`。

### FusedMoEPrepareAndFinalizeModular

`FusedMoEPrepareAndFinalizeModular` 抽象类暴露了 `prepare`、`prepare_no_receive` 和 `finalize` 函数。
`prepare` 函数负责输入激活量化和 All2All 调度。如果实现了，`prepare_no_receive` 类似于 `prepare`，只是它不等待接收来自其他工作器的结果。而是返回一个"接收器"回调，必须调用该回调以等待工作器的最终结果。并非所有 `FusedMoEPrepareAndFinalizeModular` 类都需要支持此方法，但如果可用，它可以用于将工作与初始的全到全通信交错，例如将共享专家与融合专家交错。`finalize` 函数负责调用 All2All Combine。此外，`finalize` 函数可能会或可能不会执行 TopK 权重应用和规约（请参阅 TopKWeightAndReduce 部分）。

![FusedMoEPrepareAndFinalizeModular 块](../assets/design/fused_moe_modular_kernel/prepare_and_finalize_blocks.png)

### FusedMoEExpertsModular

`FusedMoEExpertsModular` 类是 MoE 操作核心发生的地方。`FusedMoEExpertsModular` 抽象类暴露了几个重要函数：

* apply()
* workspace_shapes()
* finalize_weight_and_reduce_impl()

#### apply()

`apply` 方法是实现执行以下操作的地方：

* Permute
* 与权重 W1 的 Matmul
* Act + Mul
* 量化
* 与权重 W2 的 Matmul
* Unpermute
* 可能进行 TopK 权重应用 + 规约

#### workspace_shapes()

核心 FusedMoE 实现执行一系列操作。为这些操作中的每一个单独创建输出内存是低效的。为此，实现需要声明 2 个工作空间形状、工作空间数据类型和 FusedMoE 输出形状，作为 `workspace_shapes()` 方法的输出。此信息用于在 `FusedMoEModularKernel::forward()` 中分配工作空间张量和输出张量，并传递给 `FusedMoEExpertsModular::apply()` 方法。然后，工作空间可以在 FusedMoE 实现中用作中间缓冲区。

#### finalize_weight_and_reduce_impl()

有时在 `FusedMoEExpertsModular::apply()` 内部执行 TopK 权重应用和规约是高效的。请在此[处](https://github.com/vllm-project/vllm/pull/20228)查找示例。我们有一个 `TopKWeightAndReduce` 抽象类来促进此类实现。请参阅 TopKWeightAndReduce 部分。
`FusedMoEExpertsModular::finalize_weight_and_reduce_impl()` 返回实现希望 `FusedMoEPrepareAndFinalizeModular::finalize()` 使用的 `TopKWeightAndReduce` 对象。

![FusedMoEExpertsModular 块](../assets/design/fused_moe_modular_kernel/fused_experts_blocks.png)

### FusedMoEModularKernel

`FusedMoEModularKernel` 由 `FusedMoEPrepareAndFinalizeModular` 和 `FusedMoEExpertsModular` 对象组成。
`FusedMoEModularKernel` 伪代码/草图：

```py
class FusedMoEModularKernel:
    def __init__(self,
                 prepare_finalize: FusedMoEPrepareAndFinalizeModular,
                 fused_experts: FusedMoEExpertsModular):

        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts

    def forward(self, DP_A):

        Aq, A_scale, _, _, _ = self.prepare_finalize.prepare(DP_A, ...)

        workspace13_shape, workspace2_shape, _, _ = self.fused_experts.workspace_shapes(...)

        # 分配工作空间
        workspace_13 = torch.empty(workspace13_shape, ...)
        workspace_2 = torch.empty(workspace2_shape, ...)

        # 执行 fused_experts
        fe_out = self.fused_experts.apply(Aq, A_scale, workspace13, workspace2, ...)

        # war_impl 是一个 TopKWeightAndReduceNoOp 类型的对象，如果 fused_experts 实现
        # 执行了 TopK 权重应用和规约。
        war_impl = self.fused_experts.finalize_weight_and_reduce_impl()

        output = self.prepare_finalize.finalize(fe_out, war_impl,...)

        return output
```

## 操作指南

### 如何添加 FusedMoEPrepareAndFinalizeModular 类型

通常，FusedMoEPrepareAndFinalizeModular 类型由 All2All 调度和组合实现/内核支持。例如：

* DeepEPHTPrepareAndFinalize 类型由 DeepEP 高吞吐量 All2All 内核支持，并且
* DeepEPLLPrepareAndFinalize 类型由 DeepEP 低延迟 All2All 内核支持。

#### 步骤 1：添加 All2All 管理器

All2All 管理器的目的是设置 All2All 内核实现。`FusedMoEPrepareAndFinalizeModular` 实现通常从 All2All 管理器中获取内核实现的"句柄"，以调用调度和组合函数。请查看 All2All 管理器的实现[此处](../../vllm/distributed/device_communicators/all2all.py)。

#### 步骤 2：添加 FusedMoEPrepareAndFinalizeModular 类型

本节描述了 `FusedMoEPrepareAndFinalizeModular` 抽象类暴露的各种函数的意义。

`FusedMoEPrepareAndFinalizeModular::prepare()`：prepare 方法实现了量化和 All2All 调度。通常会调用相关 All2All Manager 中的 Dispatch 函数。

`FusedMoEPrepareAndFinalizeModular::has_prepare_no_receive()`：指示此子类是否实现了 `prepare_no_receive`。默认为 False。

`FusedMoEPrepareAndFinalizeModular::prepare_no_receive()`：prepare_no_receive 方法实现了量化和 All2All 调度。它不等待调度操作的结果，而是返回一个可调用的 thunk，该 thunk 可以被调用来等待最终结果。通常会调用相关 All2All Manager 中的 Dispatch 函数。

`FusedMoEPrepareAndFinalizeModular::finalize()`：可能执行 TopK 权重应用和规约以及 All2All Combine。通常会调用相关 All2AllManager 中的 Combine 函数。

`FusedMoEPrepareAndFinalizeModular::activation_format()`：如果 prepare 方法（即 All2All 调度）的输出是批处理的，则返回 `FusedMoEActivationFormat.BatchedExperts`。否则返回 `FusedMoEActivationFormat.Standard`。

`FusedMoEPrepareAndFinalizeModular::topk_indices_dtype()`：TopK ID 的数据类型。一些 All2All 内核对 TopK ID 的数据类型有严格要求。此要求传递给 `FusedMoe::select_experts` 函数，以便可以遵守。如果没有严格的要求，返回 None。

`FusedMoEPrepareAndFinalizeModular::max_num_tokens_per_rank()`：一次性提交给 All2All 调度的最大 token 数。

`FusedMoEPrepareAndFinalizeModular::num_dispatchers()`：调度单元的总数。此值决定了调度输出的大小。调度输出的形状为 (num_local_experts, max_num_tokens, K)。这里 max_num_tokens = num_dispatchers() * max_num_tokens_per_rank()。

我们建议选择一个与您的 All2All 实现密切匹配的现有 `FusedMoEPrepareAndFinalizeModular` 实现，并将其用作参考。

### 如何添加 FusedMoEExpertsModular 类型

FusedMoEExpertsModular 执行 FusedMoE 操作的核心部分。抽象类暴露的各种函数及其意义如下：

`FusedMoEExpertsModular::activation_formats()`：返回支持的输入和输出激活格式，即连续/批处理格式。

`FusedMoEExpertsModular::supports_expert_map()`：如果实现支持专家映射，返回 True。

`FusedMoEExpertsModular::workspace_shapes()` /
`FusedMoEExpertsModular::finalize_weight_and_reduce_impl` /
`FusedMoEExpertsModular::apply`：请参阅上面的 `FusedMoEExpertsModular` 部分。

### FusedMoEModularKernel 初始化

`FusedMoEMethodBase` 类有 3 个方法共同负责创建 `FusedMoEModularKernel` 对象。它们是：

* maybe_make_prepare_finalize,
* select_gemm_impl, 以及
* init_prepare_finalize

#### maybe_make_prepare_finalize

`maybe_make_prepare_finalize` 方法负责根据当前的 all2all 后端（例如，当 EP + DP 启用时）在适当时构造 `FusedMoEPrepareAndFinalizeModular` 的实例。基类方法目前为 EP+DP 情况构造所有 `FusedMoEPrepareAndFinalizeModular` 对象。派生类可以重写此方法以为不同场景构造 prepare/finalize 对象，例如 `ModelOptNvFp4FusedMoE` 可以为 EP+TP 情况构造 `FlashInferCutlassMoEPrepareAndFinalize`。
请参考以下实现：

* `ModelOptNvFp4FusedMoE`

#### select_gemm_impl

`select_gemm_impl` 方法在基类中未定义。派生类有责任实现一个方法来构造有效的/适当的 `FusedMoEExpertsModular` 对象。
请参考以下实现：

* `UnquantizedFusedMoEMethod`
* `CompressedTensorsW8A8Fp8MoEMethod`
* `CompressedTensorsW8A8Fp8MoECutlassMethod`
* `Fp8MoEMethod`
* `ModelOptNvFp4FusedMoE`
派生类。

#### init_prepare_finalize

根据输入和环境设置，`init_prepare_finalize` 方法创建适当的 `FusedMoEPrepareAndFinalizeModular` 对象。然后，该方法查询 `select_gemm_impl` 以获取适当的 `FusedMoEExpertsModular` 对象，并构建 `FusedMoEModularKernel` 对象。

请查看 [init_prepare_finalize](https://github.com/vllm-project/vllm/blob/1cbf951ba272c230823b947631065b826409fa62/vllm/model_executor/layers/fused_moe/layer.py#L188)。
**重要**：`FusedMoEMethodBase` 派生类在其 `apply` 方法中使用 `FusedMoEMethodBase::fused_experts` 对象。当设置允许构建有效的 `FusedMoEModularKernel` 对象时，我们用其覆盖 `FusedMoEMethodBase::fused_experts`。这基本上使派生类不关心使用哪个 FusedMoE 实现。

### 如何进行单元测试

我们在 [test_modular_kernel_combinations.py](../../tests/kernels/moe/test_modular_kernel_combinations.py) 中有 `FusedMoEModularKernel` 单元测试。

单元测试遍历 `FusedMoEPrepareAndFinalizeModular` 和 `FusedMoEPremuteExpertsUnpermute` 类型的所有组合，如果它们兼容，则运行一些正确性测试。
如果您要添加一些 `FusedMoEPrepareAndFinalizeModular` / `FusedMoEExpertsModular` 实现：

1. 将实现类型分别添加到 [mk_objects.py](../../tests/kernels/moe/modular_kernel_tools/mk_objects.py) 中的 `MK_ALL_PREPARE_FINALIZE_TYPES` 和 `MK_FUSED_EXPERT_TYPES` 中。
2. 更新 [/tests/kernels/moe/modular_kernel_tools/common.py](../../tests/kernels/moe/modular_kernel_tools/common.py) 中的 `Config::is_batched_prepare_finalize()`、`Config::is_batched_fused_experts()`、`Config::is_standard_fused_experts()`、`Config::is_fe_16bit_supported()`、`Config::is_fe_fp8_supported()`、`Config::is_fe_block_fp8_supported()` 方法。

这样做会将新实现添加到测试套件中。

### 如何检查 `FusedMoEPrepareAndFinalizeModular` & `FusedMoEExpertsModular` 兼容性

单元测试文件 [test_modular_kernel_combinations.py](../../tests/kernels/moe/test_modular_kernel_combinations.py) 也可以作为独立脚本执行。
示例：`python3 -m tests.kernels.moe.test_modular_kernel_combinations --pf-type DeepEPLLPrepareAndFinalize --experts-type BatchedTritonExperts`
作为副作用，此脚本可用于测试 `FusedMoEPrepareAndFinalizeModular` & `FusedMoEExpertsModular` 的兼容性。当使用不兼容的类型调用时，脚本将报错。

### 如何进行性能分析

请查看 [profile_modular_kernel.py](../../tests/kernels/moe/modular_kernel_tools/profile_modular_kernel.py)
该脚本可用于为任何兼容的 `FusedMoEPrepareAndFinalizeModular` 和 `FusedMoEExpertsModular` 类型生成单个 `FusedMoEModularKernel::forward()` 调用的 Torch 追踪。
示例：`python3 -m tests.kernels.moe.modular_kernel_tools.profile_modular_kernel --pf-type DeepEPLLPrepareAndFinalize --experts-type BatchedTritonExperts`

## FusedMoEPrepareAndFinalizeModular 实现

请参见[融合 MoE 内核特性](./moe_kernel_features.md#fused-moe-modular-all2all-backends)，了解所有可用的模块化 prepare 和 finalize 子类的列表。

## FusedMoEExpertsModular

请参见[融合 MoE 内核特性](./moe_kernel_features.md#fused-moe-experts-kernels)，了解所有可用的模块化专家子类的列表。
