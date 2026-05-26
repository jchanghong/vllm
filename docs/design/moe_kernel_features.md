# Fused MoE 内核特性

本文档旨在概述各种 MoE 内核（包括模块化和非模块化），以便更轻松地为任何特定情况选择合适的内核集合。其中包括关于模块化内核使用的 all2all 后端的信息。

## Fused MoE 模块化 All2All 后端

有许多 all2all 通信后端用于为 `FusedMoE` 层实现专家并行（EP）。不同的 `FusedMoEPrepareAndFinalizeModular` 子类为每个 all2all 后端提供了接口。

下表描述了每个后端的相关特性，即激活格式、支持的量化方案和异步支持。

输出激活格式（标准或批处理）对应于 `FusedMoEPrepareAndFinalizeModular` 子类的准备步骤的输出，而最终确定步骤需要相同的格式。所有后端的 `prepare` 方法期望输入为标准格式的激活，所有 `finalize` 方法返回标准格式的激活。有关格式的更多详情，请参见 [Fused MoE 模块化内核](./fused_moe_modular_kernel.md) 文档。

量化类型和格式枚举了每个 `FusedMoEPrepareAndFinalizeModular` 类支持的量化方案。量化可以基于 all2all 后端支持的格式在调度之前或之后发生，例如 `deepep_high_throughput` 仅支持块量化 fp8 格式。任何其他格式将导致以更高精度进行调度，然后再进行量化。每个后端 prepare 步骤的输出是量化后的类型。最终确定步骤通常需要与原始激活相同的输入类型，例如如果原始输入是 bfloat16 且量化方案是带有 per-tensor scales 的 fp8，则 `prepare` 将返回 fp8/per-tensor scale 激活，而 `finalize` 将接受 bfloat16 激活。有关 MoE 过程中每一步的激活类型和格式的更多详情，请参见 [Fused MoE 模块化内核](./fused_moe_modular_kernel.md) 中的图示。如果未指定量化类型，内核将在 float16 和/或 bfloat16 上运行。

异步后端支持使用 DBO（双批次重叠）和共享专家重叠（在合并步骤期间计算共享专家）。

某些模型要求在 topk==1 时将 topk 权重应用于输入激活而非输出激活，例如 Llama。对于模块化内核，此功能由 `FusedMoEPrepareAndFinalizeModular` 子类支持。对于非模块化内核，由专家函数处理此标志。

除非另有说明，后端通过 `--all2all-backend` 命令行参数（或 `ParallelConfig` 中的 `all2all_backend` 参数）控制。除 `flashinfer` 外的所有后端仅适用于 EP+DP 或 EP+TP。`Flashinfer` 可以在不使用 EP 的情况下与 EP 或 DP 配合使用。

<style>
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}
</style>

| 后端 | 输出激活格式 | 量化类型 | 量化格式 | 异步 | 权重应用于输入 | 子类 |
| ------- | ------------------ | ------------ | ------------- | ----- | --------------------- | --------- |
| naive | standard | all<sup>1</sup> | G,A,T | N | <sup>6</sup> | [layer.py][vllm.model_executor.layers.fused_moe.layer.FusedMoE] |
| deepep_high_throughput | standard | fp8 | G(128),A,T<sup>2</sup> | Y | Y | [`DeepEPHTPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht.DeepEPHTPrepareAndFinalize] |
| deepep_low_latency | batched | fp8 | G(128),A,T<sup>3</sup> | Y | Y | [`DeepEPLLPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll.DeepEPLLPrepareAndFinalize] |
| flashinfer_nvlink_two_sided | standard | nvfp4,fp8 | G,A,T | N | N | [`FlashInferNVLinkTwoSidedPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_two_sided.FlashInferNVLinkTwoSidedPrepareAndFinalize] |
| flashinfer_nvlink_one_sided | standard | nvfp4,bf16,mxfp8 | G,A,T | N | N | [`FlashInferNVLinkOneSidedPrepareAndFinalize`][vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_one_sided.FlashInferNVLinkOneSidedPrepareAndFinalize] |

!!! info "表格说明"
    1. 所有类型：mxfp4、nvfp4、int4、int8、fp8
    2. A、T 量化在调度后发生。
    3. 所有量化在调度后发生。
    4. 由不同的环境变量控制（`VLLM_FLASHINFER_MOE_BACKEND` "throughput" 或 "latency"）
    5. 这是一个无操作调度器，可用于与任何模块化专家配对，生成无需调度或合并即可运行的模块化内核。这些不能通过环境变量选择。通常用于测试或将专家子类适配到 `fused_experts` API。
    6. 这取决于专家的实现。

    ---

    - G - 分组
    - G(N) - 分组，块大小为 N
    - A - 每激活 token
    - T - 每张量

以下 `FusedMoEMethodBase` 类支持模块化内核。

- [`ModelOptFp8MoEMethod`][vllm.model_executor.layers.quantization.modelopt.ModelOptFp8MoEMethod]
- [`Fp8MoEMethod`][vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod]
- [`CompressedTensorsW4A4Nvfp4MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4.CompressedTensorsW4A4Nvfp4MoEMethod]
- [`CompressedTensorsW8A8Fp8MoEMethod`][vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8.CompressedTensorsW8A8Fp8MoEMethod]
- [`GptOssMxfp4MoEMethod`][vllm.model_executor.layers.quantization.mxfp4.GptOssMxfp4MoEMethod]
- [`UnquantizedFusedMoEMethod`][vllm.model_executor.layers.fused_moe.layer.UnquantizedFusedMoEMethod]

## Fused Experts 内核

有许多针对不同量化类型和架构的 MoE 专家内核实现。大多数遵循基础 Triton [`fused_experts`][vllm.model_executor.layers.fused_moe.fused_moe.fused_experts] 函数的通用 API。许多具有模块化内核适配器，因此可以与兼容的 all2all 后端一起使用。此表列出了每个专家内核及其特定属性。

每个内核必须提供支持的输入激活格式之一。某些风格的内核通过不同的入口点支持标准和批处理格式，例如 `TritonExperts` 和 `BatchedTritonExperts`。批处理格式内核目前仅需要与某些 all2all 后端匹配，例如 `DeepEPLLPrepareAndFinalize`。

与后端内核类似，每个专家内核仅支持某些量化格式。对于非模块化专家，激活将是原始类型并由内核内部量化。模块化专家期望激活已经是量化格式。两种类型的专家都将以原始激活类型产生输出。

每个专家内核支持一种或多种激活函数，例如 silu 或 gelu，它们应用于中间结果。

与后端一样，某些专家支持在输入激活上应用 topk 权重。此表中的列条目仅适用于非模块化专家。

大多数专家风格包含一个等价的模块化接口，它将是 `FusedMoEExpertsModular` 的子类。

要与特定的 `FusedMoEPrepareAndFinalizeModular` 子类一起使用，MoE 内核必须具有兼容的激活格式、量化类型和量化格式。

| 内核 | 输入激活格式 | 量化类型 | 量化格式 | 激活函数 | 权重应用于输入 | 模块化 | 来源 |
| ------ | ----------------- | ------------ | ------------- | ------------------- | --------------------- | ------- | ------ |
| triton | standard | all<sup>1</sup> | G,A,T | silu, gelu,</br>swigluoai,</br>silu_no_mul,</br>gelu_no_mul | Y | Y | [`fused_experts`][vllm.model_executor.layers.fused_moe.fused_moe.fused_experts],</br>[`TritonExperts`][vllm.model_executor.layers.fused_moe.experts.triton_moe.TritonExperts] |
| triton (batched) | batched | all<sup>1</sup> | G,A,T | silu, gelu | <sup>6</sup> | Y | [`BatchedTritonExperts`][vllm.model_executor.layers.fused_moe.experts.fused_batched_moe.BatchedTritonExperts] |
| deep gemm | standard,</br>batched | fp8 | G(128),A,T | silu, gelu | <sup>6</sup> | Y | </br>[`DeepGemmExperts`][vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe.DeepGemmExperts],</br>[`BatchedDeepGemmExperts`][vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe.BatchedDeepGemmExperts] |
| cutlass_fp4 | standard,</br>batched | nvfp4 | A,T | silu | Y | Y | [`CutlassExpertsFp4`][vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassExpertsFp4] |
| cutlass_fp8 | standard,</br>batched | fp8 | A,T | silu, gelu | Y | Y | [`CutlassExpertsFp8`][vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassExpertsFp8],</br>[`CutlasBatchedExpertsFp8`][vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassBatchedExpertsFp8] |
| flashinfer | standard | nvfp4,</br>fp8 | T | <sup>5</sup> | N | Y | [`FlashInferExperts`][vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe.FlashInferExperts] |
| gpt oss triton | standard | N/A | N/A | <sup>5</sup> | Y | Y | [`triton_kernel_fused_experts`][vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe.triton_kernel_fused_experts],</br>[`OAITritonExperts`][vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe.OAITritonExperts] |
| marlin | standard,</br>batched | <sup>3</sup> / N/A | <sup>3</sup> / N/A | silu,</br>swigluoai | Y | Y | [`fused_marlin_moe`][vllm.model_executor.layers.fused_moe.experts.marlin_moe.fused_marlin_moe],</br>[`MarlinExperts`][vllm.model_executor.layers.fused_moe.experts.marlin_moe.MarlinExperts],</br>[`BatchedMarlinExperts`][vllm.model_executor.layers.fused_moe.experts.marlin_moe.BatchedMarlinExperts] |
| trtllm | standard | mxfp4,</br>nvfp4 | G(16),G(32) | <sup>5</sup> | N | Y | [`TrtLlmMxfp4ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe.TrtLlmMxfp4ExpertsMonolithic],</br>[`TrtLlmMxfp4ExpertsModular`][vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe.TrtLlmMxfp4ExpertsModular],</br>[`TrtLlmNvFp4ExpertsMonolithic`][vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe.TrtLlmNvFp4ExpertsMonolithic],</br>[`TrtLlmNvfp4ExpertsModular`][vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe.TrtLlmNvFp4ExpertsModular] |
| rocm aiter moe | standard | mxfp4,</br>fp8 | G(32),G(128),A,T | silu, gelu,</br>swigluoai | Y | N | `rocm_aiter_fused_experts`,</br>`AiterExperts` |
| cpu_fused_moe | standard | N/A | N/A | silu | N | N | [`CPUFusedMOE`][vllm.model_executor.layers.fused_moe.cpu_fused_moe.CPUFusedMOE] |
| naive batched<sup>4</sup> | batched | int8,</br>fp8 | G,A,T | silu, gelu | <sup>6</sup> | Y | [`NaiveBatchedExperts`][vllm.model_executor.layers.fused_moe.experts.fused_batched_moe.NaiveBatchedExperts] |

!!! info "表格说明"
    1. 所有类型：mxfp4、nvfp4、int4、int8、fp8
    2. 围绕 triton 和 deep gemm 专家的调度器包装器。将根据类型+形状+量化参数进行选择。
    3. uint4、uint8、fp8、fp4
    4. 这是支持批处理格式的专家的简单实现。主要用于测试。
    5. `activation` 参数被忽略，默认使用 SwiGlu。
    6. 仅在使用模块化内核时处理或支持。

## 模块化内核"系列"

下表显示了旨在协同工作的模块化内核"系列"。某些组合可能可以工作但尚未经过测试，例如 flashinfer 与其他 fp8 专家的组合。

| 后端 | `FusedMoEPrepareAndFinalizeModular` 子类 | `FusedMoEExpertsModular` 子类 |
| ------- | ---------------------------------------------- | ----------------------------------- |
| deepep_high_throughput | `DeepEPHTPrepareAndFinalize` | `DeepGemmExperts`、</br>`TritonExperts`、</br>`TritonOrDeepGemmExperts`、</br>`CutlassExpertsFp8`、 </br>`MarlinExperts` |
| deepep_low_latency | `DeepEPLLPrepareAndFinalize` | `BatchedDeepGemmExperts`、</br>`BatchedTritonExperts`、</br>`CutlassBatchedExpertsFp8`、</br>`BatchedMarlinExperts` |
| flashinfer | `FlashInferCutlassMoEPrepareAndFinalize` | `FlashInferExperts` |
