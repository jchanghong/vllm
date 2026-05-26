# 故障排除

本文档概述了一些您可以考虑的故障排除策略。如果您认为自己发现了一个错误，请先[搜索现有问题](https://github.com/vllm-project/vllm/issues?q=is%3Aissue)以查看是否已被报告。如果没有，请[提交新问题](https://github.com/vllm-project/vllm/issues/new/choose)，并提供尽可能多的相关信息。

!!! note
    调试完问题后，请记得关闭任何已定义的调试环境变量，或者简单地启动一个新的 shell，以避免受到残留调试设置的影响。否则，系统可能会因调试功能保持激活状态而运行缓慢。

## 下载模型时挂起

如果模型尚未下载到磁盘，vLLM 将从互联网下载，这可能需要时间，具体取决于您的互联网连接。建议先使用 [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) 下载模型，然后将模型本地路径传递给 vLLM。这样，您可以隔离问题。

## 从磁盘加载模型时挂起

如果模型很大，从磁盘加载可能需要很长时间。请注意模型的存储位置。某些集群在节点之间具有共享文件系统，例如分布式文件系统或网络文件系统，这可能会很慢。最好将模型存储在本地磁盘中。此外，请查看 CPU 内存使用情况，当模型太大时，可能会占用大量 CPU 内存，导致操作系统因频繁在磁盘和内存之间进行交换而变慢。

!!! note
    为了隔离模型下载和加载问题，您可以使用 `--load-format dummy` 参数跳过加载模型权重。这样，您可以检查模型下载和加载是否是瓶颈。

## 内存不足

如果模型太大而无法放入单个 GPU，您将收到内存不足（OOM）错误。请考虑采用[这些选项](../configuration/conserving_memory.md)来减少内存消耗。

## 生成质量发生变化

在 v0.8.0 中，默认采样参数的来源在 <https://github.com/vllm-project/vllm/pull/12622> 中发生了变化。在 v0.8.0 之前，默认采样参数来自 vLLM 的一组中性默认值。从 v0.8.0 开始，默认采样参数来自模型创建者提供的 `generation_config.json`。

在大多数情况下，这应该会带来更高质量的响应，因为模型创建者更了解哪种采样参数最适合他们的模型。然而，在某些情况下，模型创建者提供的默认值可能会导致性能下降。

您可以通过尝试旧版默认值来检查是否发生这种情况，在线模式使用 `--generation-config vllm`，离线模式使用 `generation_config="vllm"`。如果尝试后，您的生成质量有所提高，我们建议继续使用 vLLM 默认值，并敦促模型创建者在 <https://huggingface.co> 上更新他们的默认 `generation_config.json`，以便产生更高质量的生成结果。

## 启用更多日志

如果其他策略无法解决问题，很可能是 vLLM 实例在某个地方卡住了。您可以使用以下环境变量来帮助调试问题：

- `export VLLM_LOGGING_LEVEL=DEBUG` 开启更多日志。
- `export VLLM_LOG_STATS_INTERVAL=1.` 更频繁地获取日志统计信息，以跟踪运行队列、等待队列和缓存命中状态。
- `export CUDA_LAUNCH_BLOCKING=1` 识别导致问题的 CUDA 内核。
- `export NCCL_DEBUG=TRACE` 为 NCCL 开启更多日志。
- `export VLLM_TRACE_FUNCTION=1` 记录所有函数调用到日志文件中以供检查，以确定哪个函数崩溃或挂起。（警告：此标志会将令牌生成速度降低**超过 100 倍**。除非绝对必要，否则不要使用。）

## 断点

如果在子进程中执行，在 vLLM 代码库中设置普通的 `pdb` 断点可能不起作用。您会遇到类似这样的情况：

``` text
  File "/usr/local/uv/cpython-3.12.11-linux-x86_64-gnu/lib/python3.12/bdb.py", line 100, in trace_dispatch
    return self.dispatch_line(frame)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/uv/cpython-3.12.11-linux-x86_64-gnu/lib/python3.12/bdb.py", line 125, in dispatch_line
    if self.quitting: raise BdbQuit
                      ^^^^^^^^^^^^^
bdb.BdbQuit
```

一种解决方法是使用 [forked-pdb](https://github.com/Lightning-AI/forked-pdb)。通过 `pip install fpdb` 安装，并使用类似以下方式设置断点：

``` python
__import__('fpdb').ForkedPdb().set_trace()
```

另一个选项是使用 `VLLM_ENABLE_V1_MULTIPROCESSING` 环境变量完全禁用多进程。这会将调度器保留在同一进程中，因此您可以使用标准的 `pdb` 断点：

``` python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
```

## 网络配置不正确

如果您的网络配置复杂，vLLM 实例可能无法获取正确的 IP 地址。您可以找到类似 `DEBUG 06-10 21:32:17 parallel_state.py:88] world_size=8 rank=0 local_rank=0 distributed_init_method=tcp://xxx.xxx.xxx.xxx:54641 backend=nccl` 的日志，IP 地址应该是正确的。如果不是，请使用环境变量 `export VLLM_HOST_IP=<your_ip_address>` 覆盖 IP 地址。

您可能还需要设置 `export NCCL_SOCKET_IFNAME=<your_network_interface>` 和 `export GLOO_SOCKET_IFNAME=<your_network_interface>` 来指定 IP 地址的网络接口。

## 在 `self.graph.replay()` 附近出错

如果 vLLM 崩溃并且错误跟踪在 `vllm/worker/model_runner.py` 中的 `self.graph.replay()` 附近捕获到它，那么这是 CUDAGraph 内部的 CUDA 错误。要识别导致错误的特定 CUDA 操作，您可以在命令行中添加 `--enforce-eager`，或向 [LLM][vllm.LLM] 类添加 `enforce_eager=True` 来禁用 CUDAGraph 优化并隔离导致错误的精确 CUDA 操作。

## 不正确的硬件/驱动程序

如果无法建立 GPU/CPU 通信，您可以使用以下 Python 脚本并按照下面的说明来确认 GPU/CPU 通信是否正常工作。

??? code

    ```python
    # 测试 PyTorch NCCL
    import torch
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % torch.accelerator.device_count()
    torch.accelerator.set_device_index(local_rank)
    data = torch.FloatTensor([1,] * 128).to("cuda")
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    torch.accelerator.synchronize()
    value = data.mean().item()
    world_size = dist.get_world_size()
    assert value == world_size, f"期望 {world_size}，得到 {value}"

    print("PyTorch NCCL 成功！")

    # 测试 PyTorch GLOO
    gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    cpu_data = torch.FloatTensor([1,] * 128)
    dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
    value = cpu_data.mean().item()
    assert value == world_size, f"期望 {world_size}，得到 {value}"

    print("PyTorch GLOO 成功！")

    if world_size <= 1:
        exit()

    # 测试 vLLM NCCL，使用 cuda graph
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    pynccl = PyNcclCommunicator(group=gloo_group, device=local_rank)
    # 对于 0.6.5+ 版本，pynccl 默认启用，
    # 但对于 0.6.4 及以下版本，我们需要手动启用它。
    # 保留此代码以实现向后兼容，因为人们
    # 更倾向于阅读最新文档。
    pynccl.disabled = False

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        data.fill_(1)
        out = pynccl.all_reduce(data, stream=s)
        value = out.mean().item()
        assert value == world_size, f"期望 {world_size}，得到 {value}"

    print("vLLM NCCL 成功！")

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph=g, stream=s):
        out = pynccl.all_reduce(data, stream=torch.cuda.current_stream())

    data.fill_(1)
    g.replay()
    torch.cuda.current_stream().synchronize()
    value = out.mean().item()
    assert value == world_size, f"期望 {world_size}，得到 {value}"

    print("vLLM NCCL with cuda graph 成功！")

    dist.destroy_process_group(gloo_group)
    dist.destroy_process_group()
    ```

如果您在单个节点上进行测试，请调整 `--nproc-per-node` 为您要使用的 GPU 数量：

```bash
NCCL_DEBUG=TRACE torchrun --nproc-per-node=<GPU数量> test.py
```

如果您在多节点上进行测试，请根据您的设置调整 `--nproc-per-node` 和 `--nnodes`，并将 `MASTER_ADDR` 设置为主节点的正确 IP 地址和端口（例如 `10.0.0.1:29400`），所有节点均可访问。然后运行：

```bash
NCCL_DEBUG=TRACE torchrun --nnodes 2 \
    --nproc-per-node=2 \
    --rdzv_backend=static \
    --rdzv_endpoint=$MASTER_ADDR \
    --node-rank $NODE_RANK test.py
```

将 `MASTER_ADDR` 设置为主节点的 IP 地址和端口（例如 `10.0.0.1:29400`），所有节点均可访问。在主节点上将 `NODE_RANK` 设置为 `0`，在工作节点上设置为 `1`、`2`…根据您的设置调整 `--nproc-per-node` 和 `--nnodes`。

!!! note
    我们使用 `--rdzv_backend=static` 而不是 `c10d`，因为 `c10d` 会合后端在多节点设置中可能因 DNS 解析错误而失败（参见 [pytorch/pytorch#85300](https://github.com/pytorch/pytorch/issues/85300)）。`static` 后端通过要求显式节点排名来避免此问题。

如果脚本成功运行，您应该看到消息 `sanity check is successful!`。

如果测试脚本挂起或崩溃，通常意味着硬件/驱动程序在某种意义上有问题。您应该尝试联系您的系统管理员或硬件供应商以获取进一步帮助。作为一种常见的变通方法，您可以尝试调整一些 NCCL 环境变量，例如 `export NCCL_P2P_DISABLE=1`，看看是否有帮助。请查阅[他们的文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)以获取更多信息。请仅将这些环境变量用作临时变通方法，因为它们可能会影响系统的性能。最佳解决方案仍然是修复硬件/驱动程序，使测试脚本能够成功运行。

## Python 多进程

### `RuntimeError` 异常

如果您在日志中看到类似这样的警告：

```console
WARNING 12-11 14:50:37 multiproc_worker_utils.py:281] CUDA was previously
    initialized. We must use the `spawn` multiprocessing start method. Setting
    VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See
    https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
    for more information.
```

或来自 Python 的类似这样的错误：

??? console "日志"

    ```console
    RuntimeError:
            An attempt has been made to start a new process before the
            current process has finished its bootstrapping phase.

            This probably means that you are not using fork to start your
            child processes and you have forgotten to use the proper idiom
            in the main module:

                if __name__ == '__main__':
                    freeze_support()
                    ...

            The "freeze_support()" line can be omitted if the program
            is not going to be frozen to produce an executable.

            To fix this issue, refer to the "Safe importing of main module"
            section in https://docs.python.org/3/library/multiprocessing.html
    ```

那么您必须更新 Python 代码，将对 `vllm` 的使用放在 `if __name__ == '__main__':` 块之后。例如，不要这样写：

```python
import vllm

llm = vllm.LLM(...)
```

改为这样：

```python
if __name__ == '__main__':
    import vllm

    llm = vllm.LLM(...)
```

## `torch.compile` 错误

vLLM 严重依赖 `torch.compile` 来优化模型以获得更好的性能，这引入了对 `torch.compile` 功能和 `triton` 库的依赖。默认情况下，我们使用 `torch.compile` 来[优化模型中的某些函数](https://github.com/vllm-project/vllm/pull/10406)。在运行 vLLM 之前，您可以通过运行以下脚本检查 `torch.compile` 是否按预期工作：

??? code

    ```python
    import torch

    @torch.compile
    def f(x):
        # 一个测试 torch.compile 的简单函数
        x = x + 1
        x = x * 2
        x = x.sin()
        return x

    x = torch.randn(4, 4).cuda()
    print(f(x))
    ```

如果它从 `torch/_inductor` 目录引发错误，通常意味着您有一个自定义的 `triton` 库，与您使用的 PyTorch 版本不兼容。参见 <https://github.com/vllm-project/vllm/issues/12219> 的示例。

## 模型检查失败

如果您看到类似这样的错误：

```text
  File "vllm/model_executor/models/registry.py", line xxx, in _raise_for_unsupported
    raise ValueError(
ValueError: Model architectures ['<arch>'] failed to be inspected. Please check the logs for more details.
```

这意味着 vLLM 无法导入模型文件。通常这与缺少依赖项或 vLLM 构建中的二进制文件过时有关。请仔细阅读日志以确定错误的根本原因。

## 模型不受支持

如果您看到类似这样的错误：

```text
Traceback (most recent call last):
...
  File "vllm/model_executor/models/registry.py", line xxx, in inspect_model_cls
    for arch in architectures:
TypeError: 'NoneType' object is not iterable
```

或：

```text
  File "vllm/model_executor/models/registry.py", line xxx, in _raise_for_unsupported
    raise ValueError(
ValueError: Model architectures ['<arch>'] are not supported for now. Supported architectures: [...]
```

但您确定该模型在[支持的模型列表](../models/supported_models.md)中，那么可能是 vLLM 的模型解析存在问题。在这种情况下，请按照[这些步骤](../configuration/model_resolution.md)显式指定模型的 vLLM 实现。

## 推断设备类型失败

如果您看到类似 `RuntimeError: Failed to infer device type` 的错误，这意味着 vLLM 推断运行时环境的设备类型失败。您可以查看[代码](../../vllm/platforms/__init__.py)以了解 vLLM 如何推断设备类型以及为什么它不能按预期工作。在[此 PR](https://github.com/vllm-project/vllm/pull/14195)之后，您还可以设置环境变量 `VLLM_LOGGING_LEVEL=DEBUG` 以查看更多详细日志来帮助调试问题。

## NCCL 错误：`ncclCommInitRank` 期间出现未处理的系统错误

如果您的服务工作负载使用 GPUDirect RDMA 进行跨多个节点的分布式服务，并在 `ncclCommInitRank` 期间遇到错误，即使设置了 `NCCL_DEBUG=INFO` 也没有明确的错误消息，可能看起来像这样：

```text
Error executing method 'init_device'. This might cause deadlock in distributed execution.
Traceback (most recent call last):
...
   File "/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/pynccl.py", line 99, in __init__
     self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
   File "/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/pynccl_wrapper.py", line 277, in ncclCommInitRank
     self.NCCL_CHECK(self._funcs["ncclCommInitRank"](ctypes.byref(comm),
   File "/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/pynccl_wrapper.py", line 256, in NCCL_CHECK
     raise RuntimeError(f"NCCL error: {error_str}")
 RuntimeError: NCCL error: unhandled system error (run with NCCL_DEBUG=INFO for details)
...
```

这表示 vLLM 初始化 NCCL 通信器失败，可能是由于缺少 `IPC_LOCK` Linux 能力或 `/dev/shm` 未挂载。请参考[启用 GPUDirect RDMA](../serving/parallelism_scaling.md#enabling-gpudirect-rdma) 以获取正确配置 GPUDirect RDMA 环境的指导。

## CUDA 错误：提供的 PTX 是使用不受支持的工具链编译的

如果您看到类似 `RuntimeError: CUDA error: the provided PTX was compiled with an unsupported toolchain` 的错误，这意味着 vLLM wheel 中的 CUDA PTX 是使用您的系统不支持的工具链编译的。如果您收到错误 `RuntimeError: The NVIDIA driver on your system is too old`，本节也适用。

发布的 vLLM wheel 是使用特定版本的 CUDA 工具包编译的，编译后的代码可能无法在较低版本的 CUDA 驱动程序上运行。阅读 [CUDA 兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/)了解更多详情。**这仅受选定的专业和数据中心 NVIDIA GPU 支持。**

如果您使用 vLLM 官方 Docker 镜像，可以通过在 `docker run` 命令中添加 `-e VLLM_ENABLE_CUDA_COMPATIBILITY=1` 来解决此问题。这将启用预安装的 CUDA 前向兼容库。

如果您在 Docker 之外运行 vLLM，解决方法是使用启用了 [CUDA 仓库](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)的包管理器安装 `cuda-compat` 包。例如，在 Ubuntu 上，您可以运行 `sudo apt-get install cuda-compat-12-9`，然后设置 `export VLLM_ENABLE_CUDA_COMPATIBILITY=1` 和 `export VLLM_CUDA_COMPATIBILITY_PATH="/usr/local/cuda-12.9/compat"`。

在 Conda 上，您可以安装 `conda-forge::cuda-compat` 包（例如 `conda install -c conda-forge cuda-compat=12.9`），然后在激活环境后，设置 `export VLLM_ENABLE_CUDA_COMPATIBILITY=1` 和 `export VLLM_CUDA_COMPATIBILITY_PATH="${CONDA_PREFIX}/cuda-compat"`。

您可以通过运行一个通过 vLLM 初始化 CUDA 的最小 Python 脚本来验证配置是否有效：

```bash
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH="/usr/local/cuda-12.9/compat"

python3 - << 'EOF'
import vllm
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.accelerator.device_count()}")
EOF
```

请注意，我们在此处以 CUDA 12.9 为例，您可能希望安装更高版本的 cuda-compat 包，以防 vLLM 的默认 CUDA 版本更高。

## ptxas fatal：未为选项 'gpu-name' 定义值 'sm_110a'

如果您在 CUDA 13 中使用 triton 内核，可能会遇到类似 `ptxas fatal: Value 'sm_110a' is not defined for option 'gpu-name'` 的错误：

```text
(EngineCore_0 pid=9492) triton.runtime.errors.PTXASError: PTXAS error: Internal Triton PTX codegen error
(EngineCore_0 pid=9492) `ptxas` stderr:
(EngineCore_0 pid=9492) ptxas fatal   : Value 'sm_110a' is not defined for option 'gpu-name'
(EngineCore_0 pid=9492) 
(EngineCore_0 pid=9492) Repro command: /home/jetson/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas -lineinfo -v --gpu-name=sm_110a /tmp/tmp95oy_b9d.ptx -o /tmp/tmp95oy_b9d.ptx.o
(EngineCore_0 pid=9492) 
    outputs = self.engine_core.get_output()
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jetson/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 668, in get_output
    raise self._format_exception(outputs) from None
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
```

这意味着 triton 包中的 ptxas 与您的设备不兼容。您需要设置 `TRITON_PTXAS_PATH` 环境变量，手动使用 CUDA 工具包的 ptxas：

```shell
export CUDA_HOME=/usr/local/cuda
export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"
export PATH="${CUDA_HOME}/bin:$PATH"
```

## 已知问题

- 在 `v0.5.2`、`v0.5.3` 和 `v0.5.3.post1` 中，存在一个由 [zmq](https://github.com/zeromq/pyzmq/issues/2000) 引起的错误，该错误可能会根据机器配置偶尔导致 vLLM 挂起。解决方案是升级到最新版本的 `vllm`，其中包含[修复](https://github.com/vllm-project/vllm/pull/6759)。
- 为了解决旧版本 NCCL 中的内存开销问题（参见[错误](https://github.com/NVIDIA/nccl/issues/1234)），vLLM 版本 `>= 0.4.3, <= 0.10.1.1` 会设置环境变量 `NCCL_CUMEM_ENABLE=0`。连接到 vLLM 的外部进程也需要设置此变量以防止挂起或崩溃。由于底层 NCCL 错误在 NCCL 2.22.3 中已修复，此覆盖已在较新的 vLLM 版本中移除，以允许 NCCL 性能优化。
- 在某些 PCIe 机器上（例如没有 NVLink 的机器），如果您看到类似 `transport/shm.cc:590 NCCL WARN Cuda failure 217 'peer access is not supported between these two devices'` 的错误，很可能是由驱动程序错误引起的。有关更多详细信息，请参见[此问题](https://github.com/NVIDIA/nccl/issues/1838)。在这种情况下，您可以尝试设置 `NCCL_CUMEM_HOST_ENABLE=0` 来禁用该功能，或将驱动程序升级到最新版本。
