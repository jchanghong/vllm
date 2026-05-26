# 性能分析 vLLM

!!! warning
    性能分析仅供 vLLM 开发者和维护者了解代码库不同部分的时间占比。**vLLM 最终用户绝不应开启性能分析**，因为它会显著降低推理速度。

!!! tip "选择性能分析工具"
    - 使用 **Nsight Systems** 进行低开销、性能关键的性能分析。
    - 使用 **PyTorch Profiler** 进行中等开销的性能分析，提供更丰富的调试信息（例如，堆栈跟踪、内存、形状）。请注意，启用这些功能会增加开销，不建议用于基准测试。

## 使用 PyTorch Profiler 进行性能分析

我们支持使用不同的性能分析工具对 vLLM 工作进程进行跟踪。您可以在启动服务器时设置 `--profiler-config` 标志来启用性能分析。

!!! note
    `--profiler-config` 标志在 vLLM v0.13.0 及更高版本中可用。如果您使用的是更早的版本，请升级以使用此功能。

要使用 `torch.profiler` 模块，请将 `profiler` 条目设置为 `'torch'`，并将 `torch_profiler_dir` 设置为要保存跟踪数据的目录。此外，您还可以通过在配置中指定以下额外参数来控制性能分析内容：

- `torch_profiler_record_shapes` 启用记录张量形状，默认关闭
- `torch_profiler_with_memory` 记录内存信息，默认关闭
- `torch_profiler_with_stack` 启用记录堆栈信息，默认开启
- `torch_profiler_with_flops` 启用记录 FLOPs，默认关闭
- `torch_profiler_use_gzip` 控制是否使用 gzip 压缩性能分析文件，默认开启
- `torch_profiler_dump_cuda_time_total` 控制是否转储和打印聚合的 CUDA 自身时间表，默认开启

使用 `vllm bench serve` 时，您可以通过传递 `--profile` 标志来启用性能分析。

跟踪数据可以使用 <https://ui.perfetto.dev/> 进行可视化。

!!! tip
    您可以直接使用 `python -m vllm.entrypoints.cli.main bench` 调用 bench 模块，无需安装 vLLM。

!!! tip
    进行性能分析时，只向 vLLM 发送少量请求，因为跟踪数据可能会变得非常大。另外，无需解压跟踪文件，它们可以直接查看。

!!! tip
    要停止性能分析器——它会将所有性能分析跟踪文件刷新到目录中。这需要时间，例如对于大约 100 个请求的 llama 70b 数据，在 H100 上刷新需要大约 10 分钟。
    在启动服务器之前，将环境变量 VLLM_RPC_TIMEOUT 设置为较大的数值。比如设置为 30 分钟。
    `export VLLM_RPC_TIMEOUT=1800000`

### 示例命令和用法

#### 离线推理

请参考 [examples/features/profiling/simple_profiling_offline.py](../../examples/features/profiling/simple_profiling_offline.py) 获取示例。

#### OpenAI 服务器

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile"}'
```

vllm bench 命令：

```bash
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --profile \
    --num-prompts 2
```

或使用 HTTP 请求：

```shell
# 我们首先需要调用 /start_profile API 来开始性能分析。
$ curl -X POST http://localhost:8000/start_profile

# 调用模型生成。
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [
                        {
                                "role": "user",
                                "content": "San Francisco is a"
                        }
                ]
    }'

# 之后需要调用 /stop_profile API 来停止性能分析。
$ curl -X POST http://localhost:8000/stop_profile
```

## 使用 NVIDIA Nsight Systems 进行性能分析

Nsight Systems 是一款高级工具，可展示更多性能分析细节，例如寄存器和共享内存使用情况、带注释的代码区域以及底层 CUDA API 和事件。

使用包管理器[安装 nsight-systems](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html)。
以下块是 Ubuntu 的示例。

```bash
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli
```

!!! tip
    使用 `nsys` 进行性能分析时，建议设置环境变量 `VLLM_WORKER_MULTIPROC_METHOD=spawn`。默认使用 `fork` 方法而非 `spawn`。有关此主题的更多信息，请参见 [Nsight Systems 发布说明](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html#general-issues)。

Nsight Systems 性能分析器可以使用 `nsys profile ...` 启动，为 vLLM 提供了一些推荐的标志：`--trace-fork-before-exec=true --cuda-graph-trace=node`。

### 示例命令和用法

#### 离线推理

对于基本用法，您可以在运行离线推理的任何现有脚本前附加性能分析命令。

以下是使用 `vllm bench latency` 脚本的示例：

```bash
nsys profile  \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
vllm bench latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-iters-warmup 5 \
    --num-iters 1 \
    --batch-size 16 \
    --input-len 512 \
    --output-len 8
```

#### OpenAI 服务器

要对服务器进行性能分析，您需要像离线推理一样在 `vllm serve` 命令前加上 `nsys profile`，但需要指定一些其他参数以实现类似于 Torch Profiler 的动态捕获：

```bash
# 服务器
nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end repeat \
    vllm serve meta-llama/Llama-3.1-8B-Instruct --profiler-config.profiler cuda

# 客户端
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --profile \
    --num-prompts 2
```

使用 `--profile` 时，vLLM 会为每次 `vllm bench serve` 运行捕获一份性能分析数据。服务器被终止后，所有性能分析数据将被保存。

#### 分析

您可以在 CLI 中使用 `nsys stats [profile-file]` 查看摘要形式的性能分析数据，或者通过[按照此处的说明在本地安装 Nsight](https://developer.nvidia.com/nsight-systems/get-started) 在 GUI 中查看。

??? console "CLI 示例"

    ```bash
    nsys stats report1.nsys-rep
    ...
    ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

    Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)  Max (ns)   StdDev (ns)                                                  Name
    --------  ---------------  ---------  -----------  -----------  --------  ---------  -----------  ----------------------------------------------------------------------------------------------------
        46.3   10,327,352,338     17,505    589,965.9    144,383.0    27,040  3,126,460    944,263.8  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_of…
        14.8    3,305,114,764      5,152    641,520.7    293,408.0   287,296  2,822,716    867,124.9  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_of…
        12.1    2,692,284,876     14,280    188,535.4     83,904.0    19,328  2,862,237    497,999.9  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_off…
         9.5    2,116,600,578     33,920     62,399.8     21,504.0    15,326  2,532,285    290,954.1  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x64x64_warpgroupsize1x1x1_execute_segment_k_off_…
         5.0    1,119,749,165     18,912     59,208.4      9,056.0     6,784  2,578,366    271,581.7  void vllm::act_and_mul_kernel<c10::BFloat16, &vllm::silu_kernel<c10::BFloat16>, (bool)1>(T1 *, cons…
         4.1      916,662,515     21,312     43,011.6     19,776.0     8,928  2,586,205    199,790.1  void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMa…
         2.6      587,283,113     37,824     15,526.7      3,008.0     2,719  2,517,756    139,091.1  std::enable_if<T2>(int)0&&vllm::_typeConvert<T1>::exists, void>::type vllm::fused_add_rms_norm_kern…
         1.9      418,362,605     18,912     22,121.5      3,871.0     3,328  2,523,870    175,248.2  void vllm::rotary_embedding_kernel<c10::BFloat16, (bool)1>(const long *, T1 *, T1 *, const T1 *, in…
         0.7      167,083,069     18,880      8,849.7      2,240.0     1,471  2,499,996    101,436.1  void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0…
    ...
    ```

GUI 示例：

<img width="1799" alt="Screenshot 2025-03-05 at 11 48 42 AM" src="https://github.com/user-attachments/assets/c7cff1ae-6d6f-477d-a342-bd13c4fc424c" />

## 持续性能分析

在 PyTorch 基础架构仓库中有一个 [GitHub CI 工作流程](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-profiling.yml)，为 vLLM 上不同模型提供持续性能分析。这种自动化性能分析有助于跟踪随时间和不同模型配置的性能特征。

### 工作原理

该工作流程目前每周为选定的模型运行性能分析会话，生成详细的性能跟踪数据，可使用不同工具进行分析，以识别性能回归或优化机会。但是，也可以使用 GitHub Actions 工具手动触发。

### 添加新模型

要扩展持续性能分析到更多模型，您可以修改 PyTorch 集成测试仓库中的 [profiling-tests.json](https://github.com/pytorch/pytorch-integration-testing/blob/main/vllm-profiling/cuda/profiling-tests.json) 配置文件。只需在文件中添加您的模型规格，即可将其包含在自动化性能分析运行中。

### 查看性能分析结果

持续性能分析工作流程生成的性能分析跟踪数据可在 [vLLM 性能仪表板](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm)上公开获取。查找 **Profiling traces** 表以访问和下载不同模型和运行的分析跟踪数据。

## 性能分析 vLLM Python 代码

Python 标准库包括
[cProfile](https://docs.python.org/3/library/profile.html) 用于对 Python 代码进行性能分析。vLLM 包含几个辅助函数，可以方便地将其应用于 vLLM 的某个部分。
`vllm.utils.profiling.cprofile` 和 `vllm.utils.profiling.cprofile_context` 函数都可以
用于对代码的某个部分进行性能分析。

!!! note
    `vllm.utils.profiling` 辅助函数已弃用，将在
    `v0.21` 中移除。请直接使用 Python 的 `cProfile` 模块。

### 示例用法 - 装饰器

第一个辅助函数是一个 Python 装饰器，可用于对函数进行性能分析。
如果指定了文件名，性能分析数据将保存到该文件。如果未指定文件名，
性能分析数据将打印到标准输出。

```python
from vllm.utils.profiling import cprofile

@cprofile("expensive_function.prof")
def expensive_function():
    # 一些昂贵的代码
    pass
```

### 示例用法 - 上下文管理器

第二个辅助函数是一个上下文管理器，可用于对代码块进行性能分析。
与装饰器类似，文件名是可选的。

```python
from vllm.utils.profiling import cprofile_context

def another_function():
    # 更多昂贵的代码
    pass

with cprofile_context("another_function.prof"):
    another_function()
```

### 分析性能分析结果

有多种工具可帮助分析性能分析结果。
一个示例是 [snakeviz](https://jiffyclub.github.io/snakeviz/)。

```bash
pip install snakeviz
snakeviz expensive_function.prof
```

### 分析垃圾回收开销

利用 VLLM_GC_DEBUG 环境变量调试 GC 开销。

- VLLM_GC_DEBUG=1：启用 GC 调试器，记录 gc.collect 耗时
- VLLM_GC_DEBUG='{"top_objects":5}'：启用 GC 调试器，记录每次 gc.collect 中
  收集的前 5 个对象
