# Model Runner V2 设计文档

## 引言

自从 vLLM V1 首次实现以来，我们发现了一些根本性的设计错误，并积累了大量的技术债务。许多功能是在原始设计中未曾考虑的后来添加的。我们还获得了关于采样技术（例如 Gumbel-max 采样）、工具（例如 Triton）和 CUDA 特性（例如 UVA）的宝贵见解。基于这些知识，我们从第一性原理出发实现了 Model Runner V2（MRV2），使其更简洁、更高效、更模块化。

事后看来，V1 的许多设计选择都不是最优的。虽然 MRV2 尚未功能完备、未经过严格测试，并且仍然存在一些开放的设计决策，但我们相信它相比 V1 有了实质性的改进。

本文档描述了 MRV2 的设计。

## 1. 持久化批次

V1 中一个重要的摩擦来源是其持久化批次的实现。

### 背景

V1 引入了持久化批次，以最小化输入准备过程中的 CPU 开销。当请求被调度执行一步时，模型运行器必须构建连续的输入张量（例如 block table 和每个请求的温度值）以输入模型。在 Python 中从头构建这些张量每一步通常非常慢，尤其是对于像 block table 这样的大张量。

持久化批次优化利用了连续步骤中的请求批次大部分相同这一事实。每步只有少数请求（如果有的话）加入或完成。通过维护持久化状态张量并应用增量差异而不是从头重构输入，可以显著降低 CPU 开销。

### V1 方法的问题

虽然高效，但 V1 的持久化批次设计由于将持久化状态与输入张量耦合而引入了不必要的复杂性。V1 直接将持久化状态张量用作模型和采样器的输入，这强加了严格的布局和排序要求。当请求加入或完成时，这通常需要复杂的跨张量重排序，而不是简单的行插入/删除。

V1 还必须维护 `CachedRequestState`，这是请求状态的冗余备份副本，因为持久化张量中的行可能在请求仍然活跃时被覆盖。

结果是复杂的簿记工作，在异步调度下变得更加困难。

![V1 中的持久化批次](../assets/design/model_runner_v2/persistent_batch_v1.png)

### MRV2 的解决方案

MRV2 将持久化状态张量与每步输入张量解耦。给定该步的请求排序（通常由注意力后端决定），MRV2 从持久化状态中收集输入张量。

1. 预分配一个固定大小的张量，具有 `max_num_reqs` 行（大多数平台上默认为 1024）。
2. 为每个请求分配一个在其活跃生命周期内（直到完成或被抢占）的永久行。
3. 将被抢占视为完成。恢复时，将请求数据作为新状态重新添加。

这消除了对 `CachedRequestState` 的需求，并简化了簿记。大型状态张量主要存储在 GPU 内存中，因此收集操作在 GPU 上并行运行，开销较低。

![MRV2 中的持久化批次](../assets/design/model_runner_v2/persistent_batch_mrv2.png)

## 2. 异步优先

vLLM 现在严重依赖异步调度。调度器和工作器为第 `N+1` 步准备输入，同时 GPU 执行第 `N` 步，重叠 CPU 和 GPU 工作以最大化利用率。

V1 最初并非为异步调度而设计，支持需要改造行为和 hack。MRV2 则假设核心模型执行循环是一个没有 CPU 同步点的 CUDA 流。CPU 入口点将工作排队到该流上。

![异步执行时间线](../assets/design/model_runner_v2/async_sched.png)

## 3. 移除异步屏障

异步执行的一个关键要求是 CPU 操作保持非阻塞。必须避免显式同步（例如 `torch.accelerator.synchronize`）和隐式同步（例如非固定内存的 `.to("cuda")`）。

然而，当 CPU 和 GPU 同时访问同一内存时，异步执行可能会引入竞态条件。

示例（不安全）：

```python
class ModelRunner:
    def __init__(self, ...):
        # 固定内存缓冲区
        self.states = torch.zeros(
            max_num_reqs, dtype=torch.int32, device="cpu", pin_memory=True
        )

    def execute_step(self, ...):
        self.states[req_idx] = new_req.data
        states = self.states.to("cuda", non_blocking=True)
```

CPU 可能修改 `self.states`，而 GPU 仍然通过异步拷贝从中读取。

V1 通过在关键部分周围设置异步屏障来解决此问题。这避免了竞态，但也有缺点：

1. 容易遗漏受保护的缓冲区（容易出错）。
2. 组织方式不灵活（所有 CPU 工作必须保持在屏障内）。
3. 由于同步，可能减少重叠。

![共享 CPU 缓冲区的竞态条件](../assets/design/model_runner_v2/async_race_condition.png)

### MRV2 的解决方案：消除竞态

MRV2 将持久化 CPU 状态与复制的张量分离：

```python
class ModelRunner:
    def __init__(self, ...):
        # 非固定内存
        self.states = torch.zeros(
            max_num_reqs, dtype=torch.int32, device="cpu", pin_memory=False
        )

    def execute_step(self, ...):
        self.states[req_idx] = new_req.data
        tmp_states = self.states.pin_memory()
        states = tmp_states.to("cuda", non_blocking=True)
```

现在 CPU 写入 `self.states`，而 GPU 从 `tmp_states` 读取，无需显式同步即可消除竞态。

![使用临时固定内存副本消除竞态](../assets/design/model_runner_v2/async_no_race_condition.png)

## 4. StagedWriteTensor

对于像 block table 这样的大型张量，MRV2 避免每步进行完整的 CPU 到 GPU 拷贝，而是使用 `StagedWriteTensor`：

1. 将基础张量保留在 GPU 上。
2. 在 CPU 上暂存差异。
3. 将差异打包到连续缓冲区中。
4. 将打包的差异拷贝到 GPU。
5. 启动一个内核应用差异。

使用示例：

```python
# 在 GPU 上初始化状态
state = StagedWriteTensor(size=(1024, 1000), dtype=torch.int32, device="cuda")

# 将 [3, 1, 2] 写入第 2 行，从索引 3 开始
state.stage_write(row=2, start=3, value=[3, 1, 2])

# 将 [-1, -2, -5] 写入第 0 行，从索引 1 开始
state.stage_write(row=0, start=1, value=[-1, -2, -5])

# 应用暂存的更改
state.apply_write()
```

这支持不规则更新，无需 CPU-GPU 同步，且内核启动次数最少。对于 block table 和混合 CPU/GPU 写入的状态（如 `num_computed_tokens`）尤其有用。

## 5. GPU 原生输入元数据准备和输出处理

MRV2 使用 Triton 内核准备输入，如 `input_ids`、`positions`、`query_start_loc` 和 `seq_lens`。

优势：

1. 更好的异步行为：GPU 可以推导出 CPU 可能还不知道的值（例如在推测解码中）。
2. 更低的 CPU 开销：输入准备在 GPU 上非常廉价，避免了 Python 瓶颈。

### 通用虚拟寻址（UVA）

MRV2 在某些路径中使用 UVA，让 GPU 内核直接访问大型 CPU 驻留张量（例如 `prefill_token_ids`），而无需将这些张量复制到 GPU 内存中。

## 6. Triton 原生采样器

MRV2 主要在 Triton 中重新实现了采样，以获得更好的数值/内存控制和优化。

### Gumbel 采样内核

MRV2 引入了 Triton Gumbel 采样内核，避免显式的 softmax 物化，并使用从种子输入得到的无状态内核内 RNG。

### 高效 Top-K Logprobs

V1 在 top-k 之前物化全词汇量的 logprobs。MRV2 先从 logits 中识别 top-k token，然后仅计算所选 token 的 logprobs。这降低了峰值 GPU 内存使用。

### 内存高效的 Prompt Logprobs

MRV2 支持更细粒度的分块，包括在单个 prompt 内进行分块，以避免长 prompt 上的内存峰值。

### 更好的推测解码兼容性

MRV2 不使用扩展每个请求的采样状态以匹配每个 logits 的形状，而是在内核内部使用间接寻址（`idx_mapping`）将每个 logits 向量映射到正确的请求状态。这简化了对复杂采样参数和 logits 处理器的支持。

## 7. 模块化

MRV2 强调模块化。与 V1 庞大且纠缠在一起的 `gpu_model_runner.py` 相比，MRV2 将功能逻辑分散到专用文件中（例如 `mrope_utils.py`、`penalties.py` 等）。

它还将模型输入整合到 `InputBatch` 类中，并减少了直接的模型运行器属性耦合。

## 8. 不滥用 `dummy_run`

在 V1 中，`dummy_run` 承担了太多职责：

- 初始内存分析和 `torch.compile`
- CUDA graph 捕获
- 预热
- EP+DP 的空 DP 前向传播

MRV2 简化了这一点：

1. `execute_model` 支持不影响状态的虚拟运行。
2. `dummy_run` 委托给 `execute_model` 进行分析、预热和空 DP 前向传播。
3. CUDA graph 捕获使用独立的专用路径。

这降低了复杂性，并消除了由 `execute_model` 和 `dummy_run` 行为差异引起的 bug。

## 9. 显式 CUDA Graph 管理

V1 的 CUDA graph 处理是隐式的，难以推理。MRV2 使用 `CUDAGraphManager`，通过标准的 PyTorch API 显式捕获和启动完整的 CUDA graphs。

这使得 graph 生命周期和执行模式决策更加易于理解且更容易扩展。例如：MRV2 可以将多个草稿模型前向传播捕获到一个 CUDA graph 中。

## 开发理念

MRV2 的更改应满足更高的代码质量标准。随着与 V1 功能差距的填补，应在 MRV2 设计背景下从第一性原理重新考虑功能，而不是快速移植 V1 的行为。

一个关键要求是保持模块化和清晰的抽象边界，即使这需要更多的前期设计迭代。
