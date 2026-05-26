<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

vLLM 最初支持在 Intel GPU 平台上进行基本的模型推理和服务。

--8<-- [end:installation]
--8<-- [start:requirements]

- 支持的硬件：Intel Data Center GPU、Intel ARC GPU
- 依赖项：[vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels)：一个提供所有必要的 vLLM 自定义内核的包，用于在 Intel GPU 平台上运行 vLLM
- Python：3.12
!!! warning
    提供的 vllm-xpu-kernels whl 包是 Python 3.12 特定的，因此必须使用此版本。

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

此设备没有关于创建新 Python 环境的额外信息。

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

目前，没有预构建的 XPU wheel 包。

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

- 首先，安装所需的[驱动程序](https://dgpu-docs.intel.com/driver/installation.html#installing-gpu-drivers)。
- 其次，安装 vLLM XPU 后端构建所需的 Python 包（Intel OneAPI 依赖项将作为 `torch-xpu` 的一部分自动安装，请参阅 [PyTorch XPU 入门](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)）：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install --upgrade pip
pip install -v -r requirements/xpu.txt
```

- 然后，安装适用于 Intel XPU 的正确 Triton 包。

    默认的 `triton` 包（用于 NVIDIA GPU）可能会作为传递依赖项安装（例如，通过 `xgrammar`）。对于 Intel XPU，您必须将其替换为 `triton-xpu`：

    ```bash
    pip uninstall -y triton triton-xpu
    pip install triton-xpu==3.6.0 --extra-index-url https://download.pytorch.org/whl/xpu
    ```

    !!! note
        - `triton`（无后缀）仅适用于 NVIDIA GPU。在 XPU 上使用它而不是 `triton-xpu` 可能导致正确性或运行时问题。
        - 对于 torch 2.11（`requirements/xpu.txt` 中使用的版本），匹配的包是 `triton-xpu==3.7.0`。如果您使用不同版本的 torch，请在 [docker/Dockerfile.xpu](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.xpu) 中检查相应的 `triton-xpu` 版本。

- 最后，构建并安装 vLLM XPU 后端：

```bash
VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v
```

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

目前，我们在 Docker [hub](https://hub.docker.com/r/intel/vllm/tags) 上发布了基于 vLLM 发布版本的预构建 XPU 镜像。更多信息请参考发布[说明](https://github.com/intel/ai-containers/blob/main/vllm)。

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

```bash
docker build -f docker/Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .
docker run -it \
             --rm \
             --network=host \
             --device /dev/dri:/dev/dri \
             -v /dev/dri/by-path:/dev/dri/by-path \
             --ipc=host \
             --privileged \
             vllm-xpu-env
```

--8<-- [end:build-image-from-source]
--8<-- [start:supported-features]

XPU 平台支持**张量并行**推理/服务，并且还支持**流水线并行**作为在线服务的测试版功能。对于**流水线并行**，我们在单节点上支持以 mp 作为后端。例如，参考执行如下：

```bash
vllm serve facebook/opt-13b \
     --dtype=bfloat16 \
     --max_model_len=1024 \
     --distributed-executor-backend=mp \
     --pipeline-parallel-size=2 \
     -tp=8
```

默认情况下，如果系统中未检测到现有的 ray 实例，将自动启动一个 ray 实例，其 `num-gpus` 等于 `parallel_config.world_size`。建议在执行前正确启动 ray 集群，请参考辅助脚本 [examples/ray_serving/run_cluster.sh](https://github.com/vllm-project/vllm/blob/main/examples/ray_serving/run_cluster.sh)。

--8<-- [end:supported-features]
--8<-- [start:distributed-backend]

XPU 平台使用 **torch-ccl**（torch<2.8）和 **xccl**（torch>=2.8）作为分布式后端，因为 torch 2.8 支持 **xccl** 作为 XPU 的内置后端。

--8<-- [end:distributed-backend]
