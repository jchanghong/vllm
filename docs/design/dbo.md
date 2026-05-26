# 双批次重叠

## 动机

vLLM 中 DBO 系统的核心动机是将 MoE 层中的稀疏 all-to-all 通信与周围的计算重叠。该系统目前仅针对 DP+EP 部署。

## 引言

双批次重叠系统的工作原理是在模型运行器中将批次拆分为两半，创建两个工作线程，然后在每个工作线程上运行模型。当启用 DBO 时，`FusedMoEModularKernel` 内的让位点（yield points）允许两个 CPU 工作线程（也称为 UBatch 线程）相互乒乓切换，使得一个线程在运行计算时，另一个线程在等待通信。在整个代码中，ubatch 可能用作 microbatch 的简写；这是 µ-batch 简写的 ASCII 友好版本。

DBO 系统包括对 `GpuModelRunner` 和 `ModularKernel` 的修改，并定义了两个实用类：`UBatchWrapper` 和 `UBatchContext`。`UBatchWrapper` 管理线程生命周期和模型的 CUDA graph 执行。`UBatchContext` 包装了 `ForwardContext`，用于协调两个 UBatch 线程之间的同步。

以下是 vLLM 当前实现的重叠调度。

```python
# 调度符号图例：
#    S = 共享专家
#    A0 = MLA qkv proj,
#    A1 = 核心注意力 + out proj + MoE 门控
#    D = 调度
#    C = 合并

# Comp: |-A0₀-A1₀-||-MLP₁-||-S₁-MLP₀-||-S₀-A0₁-A1₁-|
# Comm: |----D₁---||--D₀--||----C₁---||-----C₀-----|
# Order: D₁ send, A0₀, A1₀, D₁ recv, D₀ send, MLP₁, D₀ recv,
#        C₁ send, S₁, MLP₀, C₁ recv, C₀ send, S₀, A0₁, A1₁, C₀ recv.
# MLP_SHARED_OVERLAP = "mlp_shared_overlap"
```

## 使用 DBO 运行

要启用 DBO 系统，请在 vllm serve 命令中传入 `--enable-dbo` 参数。这必须与 `--data-parallel-size N`（其中 N 大于 1）和 `--enable-expert-parallel` 一起运行。此外，还有两个配置旋钮。

* `--dbo-decode-token-threshold` 启用 DBO 所需的仅解码批次中的最小 token 数量
* `--dbo-prefill-token-threshold` 启用 DBO 所需的包含至少一个预填充的批次中的最小 token 数量

目前，DBO 仅支持 DeepEP，因此必须安装 DeepEP，并且如果您的工作负载主要是解码请求，则 `--all2all-backend` 参数必须设置为 `deepep_low_latency`；如果工作负载主要是预填充请求，则设置为 `deepep_high_throughput`。

以下命令将启动一个具有专家并行和 DBO 启用的两个 DP rank 服务器。
例如：`vllm serve deepseek-ai/DeepSeek-V2-Lite --trust-remote-code --data-parallel-size 2 --enable-expert-parallel --enable-dbo --all2all-backend deepep_low_latency`

请注意，`CUDA_VISIBLE_DEVICES` 中必须至少有两个 GPU 可见。

## DBO 组件

* GPUModelRunner
* UBatchWrapper
* UBatchContext

### GPU Model Runner

批次由 `GPUModelRunner` 类拆分为微批次。这通过两步完成。首先，跨所有 DP rank 进行协调以确定是否应用微批处理。微批处理必须在所有 DP rank 之间保持一致。如果任何 DP rank 无法进行微批处理，则对所有 rank 禁用。如果所有 DP rank 都将进行微批处理，则总 token 数将填充到所有 rank 中的最大 token 数。如果应用填充后任何 rank 最终得到空的第二个微批次，则微批处理将被中止，所有 rank 都不会进行微批处理。一旦所有 rank 启动了微批处理，就执行第二步。`GPUModelRunner` 将 `CommonAttentionMetadata` 切成两半，使得每个微批次有一个注意力元数据。

### UBatchWrapper

gpu_ubatch_wrapper

`UBatchWrapper` 类是一个模型包装器，负责 DBO 的所有线程、UBatchContext 和 CUDA graph 管理。它设计为对 GPU Model Runner 相对透明。

实现将模型运行两次，每个微批次一次。每次模型调用发生在一个 UBatch 线程内。这些线程并行启动，并使用 `UBatchContext` 进行同步。每个线程提供一份切片后的注意力元数据，用于运行其处理的那一半批次。

DBO 的 CUDA graph 完全由 `UBatchWrapper` 管理。因此，DBO 仅支持使用 Full CUDA graphs 运行。然而，一旦捕获了 DBO CUDA graph，它可以在没有任何多线程或 CPU 同步的情况下回放。

#### 接口

`__init__` 方法接受模型、VllmConfig、CUDAGraphMode 和设备。

`forward` 方法独占接受模型参数。它根据 `forward_context` 中是否存在 `ubatch_slices` 对象来决定是否使用 DBO 运行。否则，模型在没有 DBO 的情况下运行。

### UBatchContext

ubatch_context

`UBatchContext` 类是一个 `ForwardContext` 包装类，由 `UBatchWrapper` 类用于同步两个 UBatch 线程。它应该只通过使用 `make_ubatch_contexts` 来实例化。

当其中一个 UBatch 线程达到 `dbo_yield` 调用时，它暂停并启动另一个线程，另一个线程将运行直到达到相同的 `dbo_yield` 调用。这种"乒乓"动态继续，线程在每次 `dbo_yield` 调用处交换，直到模型执行完成。

当前实现将所有 `dbo_yield` 和 `dbo_maybe_run_recv_hook` 调用放在 `FusedMoEModularKernel.forward` 方法中。

#### 接口

`make_ubatch_context` 函数初始化两个 `UBatchContext`，每个 UBatch 线程一个。它接受两个 CUDA 流、预先存在的 `ForwardContexts` 和一个 CPU 线程屏障。应仅使用此函数来实例化 `UBatchContexts`。它将处理所有事件初始化。

`dbo_register_recv_hook` 方法注册一个回调，该回调可由另一个 UBatch 线程的 `UBatchContext` 中的 `FusedMoEPrepareAndFinalizeModular` 类返回。当另一个线程调用 `dbo_maybe_run_recv_hook` 时，将运行该回调。这通常用于等待 all-to-all 内核。

`dbo_maybe_run_recv_hook` 方法运行由 `dbo_register_recv_hook` 函数设置的回调（如果该回调存在）。

`dbo_yield` 方法使当前线程进入休眠并唤醒另一个 UBatch 线程。
