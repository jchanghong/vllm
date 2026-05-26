---
toc_depth: 3
---

# CPU

vLLM 是一个 Python 库，支持以下 CPU 变体。选择您的 CPU 类型以查看特定于供应商的说明：

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:installation"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:installation"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:installation"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:installation"

## 技术讨论

主要讨论在 [vLLM Slack](https://slack.vllm.ai/) 的 `#sig-cpu` 频道进行。

当在 GitHub 上提交关于 CPU 后端的 Issue 时，请在标题中添加 `[CPU Backend]`，它将被标记为 `cpu` 以便更好地关注。

## 要求

- Python：3.10 -- 3.13

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:requirements"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:requirements"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:requirements"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:requirements"

## 使用 Python 设置

### 创建新的 Python 环境

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

### 预构建的 Wheel 包

指定索引 URL 时，请确保使用 `cpu` 变体子目录。
例如，nightly 构建索引是：`https://wheels.vllm.ai/nightly/cpu/`。

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:pre-built-wheels"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:pre-built-wheels"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:pre-built-wheels"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:pre-built-wheels"

### 从源码构建 Wheel 包

#### 仅使用 Python 构建（无需编译） {#python-only-build}

此方法需要您平台的[预构建 wheel 包](#pre-built-wheels)。

请参考 [GPU 上的 Python-only 构建](./gpu.md#python-only-build) 的说明，并将构建命令替换为：

```bash
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cpu VLLM_TARGET_DEVICE=cpu uv pip install --editable .
```

#### 完整构建（需编译） {#full-build}

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:build-wheel-from-source"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:build-wheel-from-source"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:build-wheel-from-source"

=== "IBM Z (s390x)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:build-wheel-from-source"

## 使用 Docker 设置

### 预构建镜像

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:pre-built-images"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:pre-built-images"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:pre-built-images"

=== "IBM Z (S390X)"

    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:pre-built-images"

### 从源码构建镜像

=== "Intel/AMD x86"

    --8<-- "docs/getting_started/installation/cpu.x86.inc.md:build-image-from-source"

=== "ARM AArch64"

    --8<-- "docs/getting_started/installation/cpu.arm.inc.md:build-image-from-source"

=== "Apple silicon"

    --8<-- "docs/getting_started/installation/cpu.apple.inc.md:build-image-from-source"

=== "IBM Z (S390X)"
    --8<-- "docs/getting_started/installation/cpu.s390x.inc.md:build-image-from-source"

## 相关的运行时环境变量

- `VLLM_CPU_KVCACHE_SPACE`：指定 KV 缓存大小（例如，`VLLM_CPU_KVCACHE_SPACE=40` 表示 40 GiB 的 KV 缓存空间），更大的设置将允许 vLLM 并行处理更多请求。此参数应根据用户的硬件配置和内存管理模式进行设置。默认值为 `0`。
- `VLLM_CPU_OMP_THREADS_BIND`：指定专用于 OpenMP 线程的 CPU 核心，可设置为 CPU id 列表、`auto`（默认）或 `nobind`（禁用绑定到单个 CPU 核心并继承用户定义的 OpenMP 变量）。例如，`VLLM_CPU_OMP_THREADS_BIND=0-31` 表示将有 32 个 OpenMP 线程绑定在 0-31 CPU 核心上。`VLLM_CPU_OMP_THREADS_BIND=0-31|32-63` 表示将有 2 个张量并行进程，rank0 的 32 个 OpenMP 线程绑定在 0-31 CPU 核心上，rank1 的 OpenMP 线程绑定在 32-63 CPU 核心上。设置为 `auto` 时，每个 rank 的 OpenMP 线程分别绑定到每个 NUMA 节点的 CPU 核心上。如果设置为 `nobind`，OpenMP 线程数由标准的 `OMP_NUM_THREADS` 环境变量决定。
- `VLLM_CPU_NUM_OF_RESERVED_CPU`：指定每个 rank 不专用于 OpenMP 线程的 CPU 核心数。此变量仅在 `VLLM_CPU_OMP_THREADS_BIND` 设置为 `auto` 时生效。默认值为 `None`。如果未设置值且使用 `auto` 线程绑定，则 `world_size == 1` 时不保留 CPU，`world_size > 1` 时每个 rank 保留 1 个 CPU。
- `CPU_VISIBLE_MEMORY_NODES`：指定 vLLM CPU worker 可见的 NUMA 内存节点，类似于 ```CUDA_VISIBLE_DEVICES```。此变量仅在 `VLLM_CPU_OMP_THREADS_BIND` 设置为 `auto` 时生效。该变量为自动线程绑定功能提供了更多控制，例如屏蔽节点和更改节点绑定顺序。
- `VLLM_CPU_SGL_KERNEL`（仅 x86，实验性）：是否为线性层和 MoE 层使用小批量优化内核，特别适用于低延迟需求（如在线服务）。这些内核需要 AMX 指令集、BFloat16 权重类型且权重维度可被 32 整除。默认值为 `0`（关闭）。

## 常见问题

### 应该使用哪个 `dtype`？

- 目前，vLLM CPU 使用模型默认设置作为 `dtype`。但是，由于 torch CPU 中对 float16 的支持不稳定，如果出现任何性能或精度问题，建议显式设置 `dtype=bfloat16`。

### 如何在 CPU 上启动 vLLM 服务？

- 使用在线服务时，建议为服务框架预留 1-2 个 CPU 核心以避免 CPU 过度订阅。例如，在具有 32 个物理 CPU 核心的平台上，为框架保留 CPU 31，并使用 CPU 0-30 进行推理线程：

```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-30
vllm serve facebook/opt-125m --dtype=bfloat16
```

或使用默认的自动线程绑定：

```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_NUM_OF_RESERVED_CPU=1
vllm serve facebook/opt-125m --dtype=bfloat16
```

注意，建议在 `world_size == 1` 时手动为 vLLM 前端进程保留 1 个 CPU。

### CPU 上支持哪些模型？

有关在 CPU 平台上验证的模型的完整和最新列表，请参阅官方文档：[CPU 上支持的模型](../../models/hardware_supported_models/cpu.md)

### 如何找到支持的 CPU 模型的基准测试配置示例？

对于 [CPU 上支持的模型](../../models/hardware_supported_models/cpu.md) 下列出的任何模型，vLLM 基准测试套件的 CPU 测试用例中提供了优化的运行时配置，这些配置在 CPU 测试用例中定义为 serving-tests-cpu.json。纯文本模型、多模态模型和嵌入模型的完整测试用例分别在 CPU 纯文本测试用例 serving-tests-cpu-text.json、CPU 多模态测试用例 serving-tests-cpu-multimodal.json 和 CPU 嵌入测试用例 serving-tests-cpu-embed.json 中。
有关如何确定这些优化配置的详细信息，请参阅：[性能基准测试详情](../../../.buildkite/performance-benchmarks/README.md#performance-benchmark-details)。
要使用这些优化设置对支持的模型进行基准测试，请按照[手动运行 vLLM 基准测试套件](../../benchmarking/dashboard.md#manually-trigger-the-benchmark)中的步骤，在 CPU 环境中运行基准测试套件。

以下是一个使用优化配置对所有 CPU 支持的模型进行基准测试的命令示例。

```bash
ON_CPU=1 bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

基准测试结果将保存在 `./benchmark/results/` 目录中。
在该目录中，生成的 `.commands` 文件包含所有基准测试的命令示例。

我们建议将 tensor-parallel-size 配置为与系统上的 NUMA 节点数匹配。请注意，当前版本不支持 tensor-parallel-size=6。
要确定可用的 NUMA 节点数，请使用以下命令：

```bash
lscpu | grep "NUMA node(s):" | awk '{print $3}'
```

关于性能参考，用户还可以查阅 [vLLM 性能仪表板](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm&deviceName=cpu)
，该仪表板发布了使用相同基准测试套件生成的默认模型 CPU 结果。

#### 空跑模式

对于只需要获取优化运行时配置而不运行基准测试的用户，提供了空跑模式。
通过传递环境变量 `DRY_RUN=1` 给 run-performance-benchmarks.sh，
所有命令将生成在 `./benchmark/results/` 下。

```bash
ON_CPU=1 DRY_RUN=1 bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

通过提供不同的 JSON 文件，用户可以为不同模型（如嵌入模型）获取运行时配置。

```bash
ON_CPU=1 SERVING_JSON=serving-tests-cpu-embed.json DRY_RUN=1 bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

通过提供 MODEL_FILTER 和 DTYPE_FILTER，将仅生成相关模型 ID 和数据类型的命令。

```bash
ON_CPU=1 SERVING_JSON=serving-tests-cpu-text.json DRY_RUN=1 MODEL_FILTER=meta-llama/Llama-3.1-8B-Instruct DTYPE_FILTER=bfloat16  bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

### 如何决定 `VLLM_CPU_OMP_THREADS_BIND`？

- 大多数情况下推荐使用默认的 `auto` 线程绑定。理想情况下，每个 OpenMP 线程将分别绑定到一个专用的物理核心，每个 rank 的线程将分别绑定到同一个 NUMA 节点，当 `world_size > 1` 时，每个 rank 将保留 1 个 CPU 用于其他 vLLM 组件。如果您遇到任何性能问题或意外的绑定行为，请尝试如下方式绑定线程。

- 在启用超线程、具有 16 个逻辑 CPU 核心 / 8 个物理 CPU 核心的平台上：

??? console "命令"

    ```console
    $ lscpu -e # 检查逻辑 CPU 核心与物理 CPU 核心之间的映射

    # "CPU" 列表示逻辑 CPU 核心 ID，"CORE" 列表示物理核心 ID。在此平台上，两个逻辑核心共享一个物理核心。
    CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
    0    0      0    0 0:0:0:0          yes 2401.0000 800.0000  800.000
    1    0      0    1 1:1:1:0          yes 2401.0000 800.0000  800.000
    2    0      0    2 2:2:2:0          yes 2401.0000 800.0000  800.000
    3    0      0    3 3:3:3:0          yes 2401.0000 800.0000  800.000
    4    0      0    4 4:4:4:0          yes 2401.0000 800.0000  800.000
    5    0      0    5 5:5:5:0          yes 2401.0000 800.0000  800.000
    6    0      0    6 6:6:6:0          yes 2401.0000 800.0000  800.000
    7    0      0    7 7:7:7:0          yes 2401.0000 800.0000  800.000
    8    0      0    0 0:0:0:0          yes 2401.0000 800.0000  800.000
    9    0      0    1 1:1:1:0          yes 2401.0000 800.0000  800.000
    10   0      0    2 2:2:2:0          yes 2401.0000 800.0000  800.000
    11   0      0    3 3:3:3:0          yes 2401.0000 800.0000  800.000
    12   0      0    4 4:4:4:0          yes 2401.0000 800.0000  800.000
    13   0      0    5 5:5:5:0          yes 2401.0000 800.0000  800.000
    14   0      0    6 6:6:6:0          yes 2401.0000 800.0000  800.000
    15   0      0    7 7:7:7:0          yes 2401.0000 800.0000  800.000

    # 在此平台上，建议仅将 openMP 线程绑定到逻辑 CPU 核心 0-7 或 8-15
    $ export VLLM_CPU_OMP_THREADS_BIND=0-7
    $ python examples/basic/offline_inference/basic.py
    ```

- 在具有 NUMA 的多插槽机器上部署 vLLM CPU 后端并启用张量并行或流水线并行时，每个 NUMA 节点被视为一个 TP/PP rank。因此请注意将单个 rank 的 CPU 核心设置在同一个 NUMA 节点上，以避免跨 NUMA 节点内存访问。

### 如何决定 `VLLM_CPU_KVCACHE_SPACE`？

此值默认为 4GB。更大的空间可以支持更多的并发请求、更长的上下文长度。但是，用户应注意每个 NUMA 节点的内存容量。每个 TP rank 的内存使用量是 `weight shard size` 和 `VLLM_CPU_KVCACHE_SPACE` 之和，如果超过单个 NUMA 节点的容量，TP worker 将因内存不足而以 `exitcode 9` 被终止。

### 如何对 vLLM CPU 进行性能调优？

首先，请确保线程绑定和 KV 缓存空间已正确设置并生效。您可以通过运行 vLLM 基准测试并使用 `htop` 观察 CPU 核心使用情况来检查线程绑定。

使用 32 的倍数作为 `--block-size`，默认为 128。

推理批次大小是影响性能的重要参数。较大的批次通常提供更高的吞吐量，较小的批次提供更低的延迟。从默认值开始调整最大批次大小以平衡吞吐量和延迟，是在特定平台上提高 vLLM CPU 性能的有效方法。vLLM 中有两个相关的重要参数：

- `--max-num-batched-tokens`，定义单个批次中的 token 数量限制，对首个 token 的性能影响更大。默认值设置为：
    - 离线推理：`4096 * world_size`
    - 在线服务：`2048 * world_size`
- `--max-num-seqs`，定义单个批次中的序列数量限制，对输出 token 的性能影响更大。
    - 离线推理：`256 * world_size`
    - 在线服务：`128 * world_size`

vLLM CPU 支持数据并行 (DP)、张量并行 (TP) 和流水线并行 (PP) 以利用多个 CPU 插槽和内存节点。有关调整 DP、TP 和 PP 的更多详细信息，请参考[优化与调优](../../configuration/optimization.md)。对于 vLLM CPU，如果有足够的 CPU 插槽和内存节点，建议同时使用 DP、TP 和 PP。

### vLLM CPU 支持哪些量化配置？

- vLLM CPU 支持的量化方式：
    - AWQ（仅 x86）
    - GPTQ（仅 x86）
    - compressed-tensor INT8 W8A8（x86、s390x）

### 为什么在 Docker 中运行时看到 `get_mempolicy: Operation not permitted`？

在某些容器环境（如 Docker）中，vLLM 使用的 NUMA 相关系统调用（例如 `get_mempolicy`、`migrate_pages`）在运行时的默认 seccomp/能力设置中被阻止/拒绝。这可能导致出现 `get_mempolicy: Operation not permitted` 等警告。功能不受影响，但 NUMA 内存绑定/迁移优化可能不会生效，性能可能不是最优。

要在 Docker 中以最小权限启用这些优化，您可以参考以下提示：

```bash
docker run ... --cap-add SYS_NICE --security-opt seccomp=unconfined  ...

# 1) `--cap-add SYS_NICE` 用于解决 `get_mempolicy` 的 EPERM 问题。

# 2) `--security-opt seccomp=unconfined` 用于启用 `numa_migrate_pages()` 的 `migrate_pages`。
# 实际上，`seccomp=unconfined` 绕过了容器的 seccomp，
# 如果这不可接受，您可以自定义自己的 seccomp 配置文件，
# 基于 docker/runtime 默认的 default.json，并将 `migrate_pages` 添加到 `SCMP_ACT_ALLOW` 列表中。

# 参考：https://docs.docker.com/engine/security/seccomp/
```

或者，使用 `--privileged=true` 也可以，但权限范围更广，一般不建议使用。

在 K8S 中，可以将以下配置添加到工作负载 yaml 中，以实现与上述相同的效果：

```yaml
securityContext:
  seccompProfile:
    type: Unconfined
  capabilities:
    add:
    - SYS_NICE
```
