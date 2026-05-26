# 优化与调优

本指南涵盖了 vLLM V1 的优化策略和性能调优。

!!! tip
    内存不足？请参阅[本指南](./conserving_memory.md)了解如何节省内存。

## 优化级别

vLLM 提供 4 个优化级别（`-O0`、`-O1`、`-O2`、`-O3`），允许用户在启动时间和性能之间进行权衡：

- `-O0`：无优化。启动时间最快，但性能最低。
- `-O1`：快速优化。简单编译和快速融合，以及 PIECEWISE cudagraphs。
- `-O2`：默认优化。额外的编译范围，额外的融合，FULL_AND_PIECEWISE cudagraphs。
- `-O3`：激进优化。目前等同于 `-O2`，但未来可能包括额外的耗时或实验性优化。

更多信息请参见[优化级别文档](../design/optimization_levels.md)。

## 抢占

由于 transformer 架构的自回归特性，有时 KV 缓存空间不足以处理所有批处理的请求。在这种情况下，vLLM 可以抢占请求以释放 KV 缓存空间给其他请求。当有足够的 KV 缓存空间可用时，被抢占的请求会重新计算。当这种情况发生时，您可能会看到以下警告：

```text
WARNING 05-09 00:49:33 scheduler.py:1057 Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory. total_cumulative_preemption_cnt=1
```

虽然这种机制确保了系统的鲁棒性，但抢占和重新计算会对端到端延迟产生不利影响。如果您经常遇到抢占问题，请考虑以下措施：

- 增加 `gpu_memory_utilization`。vLLM 使用此百分比的内存预先分配 GPU 缓存。通过提高利用率，您可以提供更多的 KV 缓存空间。
- 减少 `max_num_seqs` 或 `max_num_batched_tokens`。这会减少批处理中的并发请求数量，从而减少对 KV 缓存空间的需求。
- 增加 `tensor_parallel_size`。这会将模型权重分片到多个 GPU 上，使每个 GPU 有更多内存可用于 KV 缓存。但是，增加此值可能会导致过多的同步开销。
- 增加 `pipeline_parallel_size`。这会将模型层分布到多个 GPU 上，减少每个 GPU 上模型权重所需的内存，从而间接为 KV 缓存留出更多内存。但是，增加此值可能会导致延迟损失。

您可以通过 vLLM 暴露的 Prometheus 指标监控抢占请求的数量。此外，您可以通过设置 `disable_log_stats=False` 来记录累计的抢占请求数量。

在 vLLM V1 中，默认的抢占模式是 `RECOMPUTE` 而不是 `SWAP`，因为在 V1 架构中重新计算的开销更小。

## 分块预填充

分块预填充允许 vLLM 将大的预填充处理为更小的块，并将它们与解码请求一起批处理。这一特性通过更好地平衡计算密集型（预填充）和内存密集型（解码）操作来帮助提高吞吐量和延迟。

在 V1 中，**分块预填充在可能的情况下默认启用**。启用分块预填充后，调度策略优先处理解码请求。它在调度任何预填充操作之前批处理所有待处理的解码请求。当 `max_num_batched_tokens` 预算中有可用令牌时，它会调度待处理的预填充。如果待处理的预填充请求无法完全放入 `max_num_batched_tokens`，它会自动进行分块。

此策略有两个好处：

- 它改善了 ITL 和生成解码，因为解码请求被优先处理。
- 它通过将计算密集型（预填充）和内存密集型（解码）请求定位到同一批次中，有助于实现更好的 GPU 利用率。

### 使用分块预填充进行性能调优

您可以通过调整 `max_num_batched_tokens` 来调优性能：

- 较小的值（例如 2048）可以实现更好的令牌间延迟（ITL），因为减慢解码速度的预填充更少。
- 较高的值可以实现更好的首次令牌时间（TTFT），因为您可以在一个批次中处理更多的预填充令牌。
- 为了获得最佳吞吐量，我们建议设置 `max_num_batched_tokens > 8192`，特别是在大型 GPU 上使用较小模型时。
- 如果 `max_num_batched_tokens` 与 `max_model_len` 相同，则几乎等同于 V0 的默认调度策略（除了它仍然优先处理解码）。

!!! warning
    当分块预填充被禁用时，`max_num_batched_tokens` 必须大于 `max_model_len`。
    在这种情况下，如果 `max_num_batched_tokens < max_model_len`，vLLM 可能会在服务器启动时崩溃。

```python
from vllm import LLM

# 设置 max_num_batched_tokens 以调优性能
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_num_batched_tokens=16384)
```

请参见相关论文以获得更多详细信息（<https://arxiv.org/pdf/2401.08671> 或 <https://arxiv.org/pdf/2308.16369>）。

## 并行策略

vLLM 支持多种可以组合使用的并行策略，以优化不同硬件配置下的性能。

### 张量并行（TP）

张量并行将模型参数在单个模型层内的多个 GPU 之间进行分片。这是单节点内大型模型推理最常用的策略。

**何时使用：**

- 当模型太大而无法放入单个 GPU 时
- 当您需要减少每个 GPU 上的内存压力以提供更多 KV 缓存空间来实现更高吞吐量时

```python
from vllm import LLM

# 将模型拆分到 4 个 GPU
llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=4)
```

对于无法放入单个 GPU 的模型（如 70B 参数模型），张量并行是必不可少的。

### 流水线并行（PP）

流水线并行将模型层分布到多个 GPU 上。每个 GPU 按顺序处理模型的不同部分。

**何时使用：**

- 当您已经用尽了高效的张量并行但需要进一步分布模型，或者在节点间分布时
- 对于非常深且窄的模型，层分布比张量分片更高效时

流水线并行可以与张量并行结合使用，用于非常大的模型：

```python
from vllm import LLM

# 结合流水线并行和张量并行
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
)
```

### 专家并行（EP）

专家并行是混合专家（MoE）模型的一种专门的并行形式，其中不同的专家网络分布在多个 GPU 上。

**何时使用：**

- 专门用于 MoE 模型（如 DeepSeekV3、Qwen3MoE、Llama-4）
- 当您想要平衡跨 GPU 的专家计算负载时

专家并行通过设置 `enable_expert_parallel=True` 启用，这将使 MoE 层使用专家并行而不是张量并行。它将使用与张量并行设置的相同并行度。

### 数据并行（DP）

数据并行将整个模型复制到多个 GPU 集上，并并行处理不同的请求批次。

**何时使用：**

- 当您有足够的 GPU 来复制整个模型时
- 当您需要扩展吞吐量而不是模型大小时
- 在多用户环境中，请求批次之间的隔离是有益的

数据并行可以与其他并行策略结合使用，通过 `data_parallel_size=N` 设置。请注意，MoE 层将根据张量并行大小和数据并行大小的乘积进行分片。

### 多插槽 GPU 节点的 NUMA 绑定

在多插槽 GPU 服务器上，GPU 工作进程的性能可能会下降，如果其 CPU 执行和内存分配偏离了离 GPU 最近的 NUMA 节点。vLLM 可以在 Python 子进程启动前使用 `numactl` 固定每个工作进程，以便解释器、导入和早期分配器状态从开始就以所需的 NUMA 策略创建。

使用 `--numa-bind` 启用该功能。默认情况下，vLLM 自动检测 GPU 到 NUMA 的映射，并为每个工作进程使用 `--cpunodebind=<node> --membind=<node>`。当您需要自定义 CPU 策略时，添加 `--numa-bind-cpus`，vLLM 将切换到 `--physcpubind=<cpu-list> --membind=<node>`。

这些 `--numa-bind*` 选项仅适用于 GPU 执行进程。它们不配置 CPU 后端的单独线程亲和性控制。自动 GPU 到 NUMA 检测当前已为基于 CUDA/NVML 以及基于 ROCM 的平台实现；其他 GPU 后端如果使用这些选项，必须提供显式绑定列表。

`--numa-bind-nodes` 为每个可见 GPU 接受一个非负的 NUMA 节点索引，顺序与 GPU 索引相同。
`--numa-bind-cpus` 为每个可见 GPU 接受一个 `numactl` CPU 列表，顺序与 GPU 索引相同。每个 CPU 列表必须使用 `numactl --physcpubind` 语法，如 `0-3`、`0,2,4-7` 或 `16-31,48-63`。

```bash
# 自动检测可见 GPU 的 NUMA 节点
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4 \
  --numa-bind

# 显式 NUMA 节点映射
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4 \
  --numa-bind \
  --numa-bind-nodes 0 0 1 1

# 显式 CPU 绑定，适用于 PCT 或其他高频核心布局
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4 \
  --numa-bind \
  --numa-bind-nodes 0 0 1 1 \
  --numa-bind-cpus 0-3 4-7 48-51 52-55
```

注意：

- CLI 使用会自动将多进程强制使用 `spawn` 方法。如果您通过 Python API 启用 NUMA 绑定，也请设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn`。
- 自动检测依赖于主机的 NVML 和 NUMA 支持。如果无法可靠地确定映射，请显式传递 `--numa-bind-nodes`。
- 显式的 `--numa-bind-nodes` 和 `--numa-bind-cpus` 值必须是有效的 `numactl` 输入。vLLM 进行少量验证，但有效的绑定语义仍由 `numactl` 决定。
- 当前实现对 GPU 执行进程（如 `EngineCore` 和多进程工作进程）进行绑定。它不适用于前端 API 服务器进程或 DP 协调器。
- 在容器化环境中，NUMA 策略系统调用可能需要额外权限，例如在通过 `docker run` 运行时需要 `--cap-add SYS_NICE`。

### CPU 后端线程亲和性

CPU 后端使用与 `--numa-bind` 不同的机制。CPU 执行通过 CPU 特定的环境变量（如 `VLLM_CPU_OMP_THREADS_BIND`、`VLLM_CPU_NUM_OF_RESERVED_CPU` 和 `CPU_VISIBLE_MEMORY_NODES`）进行配置，而不是面向 GPU 的 `--numa-bind*` CLI 选项。

默认情况下，`VLLM_CPU_OMP_THREADS_BIND=auto` 会根据每个 CPU 工作进程可用的 CPU 和 NUMA 拓扑派生出 OpenMP 放置策略。要覆盖自动策略，使用为 CPU 后端记录的 CPU 列表格式显式设置 `VLLM_CPU_OMP_THREADS_BIND`，或使用 `nobind` 禁用此行为。

有关当前 CPU 后端设置和调优指南，请参见：

- [相关运行时环境变量](../getting_started/installation/cpu.md#related-runtime-environment-variables)
- [如何决定 `VLLM_CPU_OMP_THREADS_BIND`](../getting_started/installation/cpu.md#how-to-decide-vllm_cpu_omp_threads_bind)

仅 GPU 的 `--numa-bind`、`--numa-bind-nodes` 和 `--numa-bind-cpus` 选项不配置 CPU 工作进程的亲和性。

### 多模态编码器的批处理级 DP

默认情况下，TP 用于像语言解码器一样分片多模态编码器的权重，以减少每个 GPU 上的内存和计算负载。

然而，由于多模态编码器的大小与语言解码器相比非常小，TP 带来的收益相对较小。另一方面，TP 会在每一层之后执行 all-reduce，从而产生显著的通信开销。

鉴于此，使用 TP 分片批处理输入数据（本质上是执行批处理级 DP）可能更有利。这已在 `tensor_parallel_size=8` 的情况下被证明可以将吞吐量和 TTFT 提高约 10%。对于使用硬件未优化的 Conv3D 操作的视觉编码器，批处理级 DP 与常规 TP 相比可以提供额外 40% 的改进。

尽管如此，由于多模态编码器的权重在每个 TP 等级上都被复制，内存消耗会略有增加，如果模型已经勉强能放得下，可能会导致 OOM。

您可以通过设置 `mm_encoder_tp_mode="data"` 来启用批处理级 DP，例如：

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-VL-72B-Instruct",
    tensor_parallel_size=4,
    # 当 mm_encoder_tp_mode="data" 时，
    # 视觉编码器使用 TP=4（而不是 DP=1）来分片输入数据，
    # 因此 TP 大小成为有效的 DP 大小。
    # 请注意，这与用于语言解码器的 DP 大小无关，后者用于专家并行设置。
    mm_encoder_tp_mode="data",
    # 无论 mm_encoder_tp_mode 的设置如何，
    # 语言解码器都使用 TP=4 来分片权重
)
```

!!! important
    批处理级 DP 不要与 API 请求级 DP 混淆（后者由 `data_parallel_size` 控制）。

批处理级 DP 需要按模型逐个实现，并通过在模型类中设置 `supports_encoder_tp_data = True` 来启用。无论如何，您需要在引擎参数中设置 `mm_encoder_tp_mode="data"` 才能使用此功能。

已知支持的模型（附相应基准测试链接）：

- dots_ocr (<https://github.com/vllm-project/vllm/pull/25466>)
- GLM-4.1V 或更高版本 (<https://github.com/vllm-project/vllm/pull/23168>)
- InternVL (<https://github.com/vllm-project/vllm/pull/23909>)
- Kimi-VL (<https://github.com/vllm-project/vllm/pull/23817>)
- Llama4 (<https://github.com/vllm-project/vllm/pull/18368>)
- MiniCPM-V-2.5 或更高版本 (<https://github.com/vllm-project/vllm/pull/23327>, <https://github.com/vllm-project/vllm/pull/23948>)
- Qwen2-VL 或更高版本 (<https://github.com/vllm-project/vllm/pull/22742>, <https://github.com/vllm-project/vllm/pull/24955>, <https://github.com/vllm-project/vllm/pull/25445>)
- Step3 (<https://github.com/vllm-project/vllm/pull/22697>)

## 输入处理

### fastokens 后端

默认情况下，vLLM 使用标准的 Hugging Face `tokenizers` 库来驱动快速分词器。对于 BPE 分词器（Qwen、Llama、DeepSeek、GPT-OSS 等），您可以切换到 [fastokens](https://github.com/crusoecloud/fastokens) Rust 后端，这是一个即插即用的替代品，在编码/解码和流式解码标记化方面速度显著更快。通过设置 `VLLM_USE_FASTOKENS=1` 启用：

```console
VLLM_USE_FASTOKENS=1 vllm serve Qwen/Qwen3-8B
```

离线 API 中的等效设置：

```python
import os
os.environ["VLLM_USE_FASTOKENS"] = "1"

from vllm import LLM
llm = LLM(model="Qwen/Qwen3-8B")
```

`fastokens` Python 包（>= 0.2.0）必须安装；如果没有安装，vLLM 在分词器加载时会引发明确的 `ImportError`。此覆盖适用于任何最终加载 HF 快速分词器的 `--tokenizer-mode`（`hf`、`deepseek_v32`、`deepseek_v4`、`qwen_vl` 等）。不使用 HF 快速分词器的模式（`mistral`、`grok2`、`kimi_audio`）会忽略此标志。

分词器密集型工作负载——长共享前缀、突发的短提示、批处理解码标记化——收益最大。如果您的瓶颈是 GPU 预填充/解码，分词器更改不太可能在端到端中可见。

### 并行处理

您可以通过 [API 服务器扩展](../serving/data_parallel_deployment.md#internal-load-balancing)并行运行输入处理。当输入处理（在 API 服务器内部运行）相对于模型执行（在引擎核心内部运行）成为瓶颈，且您有额外的 CPU 容量时，这非常有用。

```console
# 运行 4 个 API 进程和 1 个引擎核心进程
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --api-server-count 4

# 运行 4 个 API 进程和 2 个引擎核心进程
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --api-server-count 4 -dp 2
```

!!! note
    API 服务器扩展仅适用于在线推理。

!!! warning
    默认情况下，每个 API 服务器使用 8 个 CPU 线程从请求数据加载媒体项目（如图像）。

    如果您应用 API 服务器扩展，请考虑调整 `VLLM_MEDIA_LOADING_THREAD_COUNT` 以避免 CPU 资源耗尽。

!!! note
    API 服务器扩展会禁用[多模态 IPC 缓存](#ipc-caching)，因为它需要 API 和引擎核心进程之间的一一对应关系。

    这不会影响[多模态处理器缓存](#processor-caching)。

## 多模态缓存

多模态缓存避免了对相同多模态数据的重复传输或处理，这在多轮对话中常见。

### 处理器缓存

多模态处理器缓存自动启用，以避免在 `BaseMultiModalProcessor` 中重复处理相同的多模态输入。

### IPC 缓存

当 API（`P0`）和引擎核心（`P1`）进程之间存在一一对应关系时，多模态 IPC 缓存自动启用，以避免在它们之间重复传输相同的多模态输入。

#### 键复制缓存

默认情况下，IPC 缓存使用**键复制缓存**，其中缓存键存在于 API（`P0`）和引擎核心（`P1`）进程中，但实际的缓存数据仅驻留在 `P1` 中。

#### 共享内存缓存

当涉及多个工作进程时（例如，当 TP > 1 时），**共享内存缓存**更高效。可以通过设置 `mm_processor_cache_type="shm"` 启用。在此模式下，缓存键存储在 `P0` 上，而缓存数据本身位于所有进程可访问的共享内存中。

### 配置

您可以通过设置 `mm_processor_cache_gb` 的值（默认为 4 GiB）来调整缓存的大小。

如果您从缓存中获益不多，可以通过 `mm_processor_cache_gb=0` 完全禁用 IPC 和处理器缓存。

示例：

```python
# 使用更大的缓存
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    mm_processor_cache_gb=8,
)

# 使用基于共享内存的 IPC 缓存
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    tensor_parallel_size=2,
    mm_processor_cache_type="shm",
    mm_processor_cache_gb=8,
)

# 禁用缓存
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    mm_processor_cache_gb=0,
)
```

### 缓存放置

根据配置，`P0` 和 `P1` 上多模态缓存的内容如下：

| mm_processor_cache_type | 缓存类型 | `P0` 缓存 | `P1` 引擎缓存 | `P1` 工作进程缓存 | 最大内存 |
| ----------------- | ----------- | ---------- | ---------- | ----------- | ----------- |
| lru | 处理器缓存 | K + V | N/A | N/A | `mm_processor_cache_gb * data_parallel_size` |
| lru | 键复制缓存 | K | K + V | N/A | `mm_processor_cache_gb * api_server_count` |
| shm | 共享内存缓存 | K | N/A | V | `mm_processor_cache_gb * api_server_count` |
| N/A | 已禁用 | N/A | N/A | N/A | `0` |

K：存储多模态项目的哈希值
V：存储多模态项目的处理后的张量数据

## GPU 部署的 CPU 资源

vLLM V1 使用多进程架构（参见 [V1 进程架构](../design/arch_overview.md#v1-process-architecture)），其中每个进程都需要 CPU 资源。CPU 核心配置不足是性能下降的常见原因，尤其是在虚拟化环境中。

### 最低 CPU 要求

对于具有 `N` 个 GPU 的部署，至少需要：

- **1 个 API 服务器进程**——处理 HTTP 请求、分词化和输入处理
- **1 个引擎核心进程**——运行调度器并协调 GPU 工作进程
- **N 个 GPU 工作进程**——每个 GPU 一个，执行模型前向传播

这意味着至少有 **`2 + N`** 个进程在竞争 CPU 时间。

!!! warning
    使用少于进程数的物理 CPU 核心会导致争用，并显著降低吞吐量和延迟。引擎核心进程运行一个繁忙循环，对 CPU 资源不足特别敏感。

最低要求是 `2 + N` 个物理核心（1 个用于 API 服务器，1 个用于引擎核心，每个 GPU 工作进程 1 个）。实际上，分配更多核心可以提高性能，因为操作系统、PyTorch 后台线程和其他系统进程也需要 CPU 时间。

!!! important
    请注意，我们这里指的是**物理 CPU 核心**。如果您的系统启用了超线程，那么 1 个 vCPU = 1 个超线程 = 1/2 个物理 CPU 核心，因此您至少需要 `2 x (2 + N)` 个 vCPU。

### 数据并行和多 API 服务器部署

当使用数据并行或多个 API 服务器时，CPU 需求会增加：

```console
最低物理核心数 = A + DP + N + (如果 DP > 1 则为 1，否则为 0)
```

其中 `A` 是 API 服务器数量（默认为 `DP`），`DP` 是数据并行大小，`N` 是 GPU 总数。例如，在 8 个 GPU 上使用 `DP=4, TP=2` 时：

```console
4 个 API 服务器 + 4 个引擎核心 + 8 个 GPU 工作进程 + 1 个 DP 协调器 = 17 个进程
```

### 性能影响

CPU 配置不足特别影响以下方面：

- **输入处理吞吐量**——分词化、聊天模板渲染和多模态数据加载都在 CPU 上运行
- **调度延迟**——引擎核心调度器在 CPU 上运行，直接影响新令牌分发到 GPU 工作进程的速度
- **输出处理**——解码标记化、网络通信，特别是流式令牌响应，都会消耗 CPU 周期

如果您观察到 GPU 利用率低于预期，CPU 争用可能是瓶颈。增加可用 CPU 核心数量甚至提高时钟频率可以显著改善端到端性能。

## 注意力后端选择

vLLM 支持多种针对不同硬件和用例优化的注意力后端。后端会根据您的 GPU 架构、模型类型和配置自动选择，但您也可以手动指定以获得最佳性能。

有关可用后端、其功能支持以及如何配置它们的详细信息，请参见[注意力后端功能支持](../design/attention_backends.md)文档。
