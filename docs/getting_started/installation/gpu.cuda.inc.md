<!-- markdownlint-disable MD041 MD051 -->
--8<-- [start:installation]

vLLM 包含预编译的 C++ 和 CUDA（12.9）二进制文件。

--8<-- [end:installation]
--8<-- [start:requirements]

- GPU：计算能力 7.5 或更高（例如，T4、RTX20xx、A100、L4、H100、B200 等）

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

!!! note
    通过 `conda` 安装的 PyTorch 会静态链接 `NCCL` 库，这可能在 vLLM 尝试使用 `NCCL` 时导致问题。请参阅 <https://github.com/vllm-project/vllm/issues/8420> 了解更多详情。

为了获得高性能，vLLM 必须编译许多 CUDA 内核。不幸的是，这种编译引入了与其他 CUDA 版本和 PyTorch 版本的二进制不兼容性，即使对于具有不同构建配置的相同 PyTorch 版本也是如此。

因此，建议使用**全新的**环境安装 vLLM。如果您使用不同的 CUDA 版本或想要使用现有的 PyTorch 安装，则需要从源码构建 vLLM。更多详情请参见[下文](#build-wheel-from-source)。

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

```bash
uv pip install vllm --torch-backend=auto
```

??? console "pip"
    ```bash
    # 使用 CUDA 12.9 安装 vLLM。
    pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129
    ```

我们建议利用 `uv` 通过检查已安装的 CUDA 驱动程序版本，[在运行时自动选择适当的 PyTorch 索引](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)，使用 `--torch-backend=auto`（或 `UV_TORCH_BACKEND=auto`）。要选择特定的后端（例如 `cu130`），请设置 `--torch-backend=cu130`（或 `UV_TORCH_BACKEND=cu130`）。如果这不起作用，请先尝试运行 `uv self update` 来更新 `uv`。

!!! note
    NVIDIA Blackwell GPU（B200、GB200）至少需要 CUDA 12.8，因此请确保您安装的 PyTorch wheel 包版本至少为该版本。PyTorch 本身提供[专用界面](https://pytorch.org/get-started/locally/)来确定针对给定目标配置应运行的适当 pip 命令。

目前，vLLM 的二进制文件默认使用 CUDA 12.9 和公共 PyTorch 发布版本编译。我们也提供使用 CUDA 12.8、13.0 和公共 PyTorch 发布版本编译的 vLLM 二进制文件：

```bash
# 使用特定的 CUDA 版本安装 vLLM（例如 13.0）。
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
export CUDA_VERSION=130 # 或其他版本
export CPU_ARCH=$(uname -m) # x86_64 或 aarch64
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_35_${CPU_ARCH}.whl --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
```

#### 安装最新代码

LLM 推理是一个快速发展的领域，最新代码可能包含尚未发布的错误修复、性能改进和新功能。为了让用户无需等待下一个版本即可尝试最新代码，vLLM 自 `v0.5.3` 起在 <https://wheels.vllm.ai/nightly> 上为每次提交提供 wheel 包。有多个索引可供使用：

- `https://wheels.vllm.ai/nightly`：默认变体（CUDA，版本由 `VLLM_MAIN_CUDA_VERSION` 指定），使用 `main` 分支的最新提交构建。目前为 CUDA 12.9。
- `https://wheels.vllm.ai/nightly/<variant>`：所有其他变体。目前包括 `cu130` 和 `cpu`。默认变体（`cu129`）也有一个子目录以保持一致性。

要从 nightly 索引安装，请运行：

```bash
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly # 如果需要，在此处添加变体子目录
```

!!! warning "`pip` 注意事项"

    使用 `pip` 从 nightly 索引安装是_不支持的_，因为 `pip` 会合并 `--extra-index-url` 和默认索引中的包，仅选择最新版本，这使得安装已发布版本之前的开发版本变得困难。相比之下，`uv` 给予额外索引[比默认索引更高的优先级](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)。

    如果您坚持使用 `pip`，则必须指定 wheel 文件的完整 URL（可从网页获取）。

    ```bash
    pip install -U https://wheels.vllm.ai/nightly/vllm-0.11.2.dev399%2Bg3c7461c18-cp38-abi3-manylinux_2_31_x86_64.whl # 当前的 nightly 构建（文件名会变化！）
    pip install -U https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-0.11.2.dev399%2Bg3c7461c18-cp38-abi3-manylinux_2_31_x86_64.whl # 来自特定提交
    ```

##### 安装特定修订版本

如果您需要访问之前提交的 wheel 包（例如，用于二分查找行为变更、性能回归），可以在 URL 中指定提交哈希：

```bash
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069 # 使用主分支的完整提交哈希
uv pip install vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT} # 如果需要，在此处添加变体子目录
```

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

#### 仅使用 Python 构建（无需编译） {#python-only-build}

如果您只需要更改 Python 代码，可以在不编译的情况下构建和安装 vLLM。使用 `uv pip` 的 [`--editable` 标志](https://docs.astral.sh/uv/pip/packages/#editable-packages)，您对代码所做的更改将在运行 vLLM 时生效：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

此命令将执行以下操作：

1. 在您的 vLLM 克隆中查找当前分支。
2. 识别主分支中对应的基准提交。
3. 下载基准提交的预构建 wheel 包。
4. 在安装中使用其编译的库和 `vllm-rs` 二进制文件。

!!! note
    1. 如果您更改了 C++ 或内核代码，则无法使用 Python-only 构建；否则您将看到关于找不到库或未定义符号的导入错误。
    2. 如果您重新基于开发分支，建议卸载 vllm 并重新运行上述命令以确保您的库是最新的。

!!! tip "重建 Rust 前端"
如果需要重新编译 `vllm-rs` Rust 前端二进制文件，可以在不重新运行完整 pip 安装的情况下重建并安装它：

    ```bash
    ./build_rust.sh          # 发布构建
    ./build_rust.sh --debug  # 更快的开发构建
    ```

    这将安装所需的 Rust 工具链（如果需要），构建二进制文件，并将其放置在 `vllm/vllm-rs` 中。

如果在运行上述命令时看到关于找不到 wheel 包的错误，可能是因为您在 `main` 分支中基于的提交刚刚合并，其预编译 wheel 包尚不可用。您可以等待大约一小时后重试，或设置 `VLLM_PRECOMPILED_WHEEL_COMMIT=nightly` 以自动选择 `main` 上最近已构建的提交。

```bash
export VLLM_PRECOMPILED_WHEEL_COMMIT=nightly
export VLLM_USE_PRECOMPILED=1
uv pip install --editable .
```

还有更多环境变量用于控制 Python-only 构建的行为：

- `VLLM_PRECOMPILED_WHEEL_LOCATION`：指定要使用的预编译 wheel 包的精确 URL 或本地文件路径。所有其他查找 wheel 包的逻辑将被跳过。
- `VLLM_PRECOMPILED_WHEEL_COMMIT`：覆盖用于下载预编译 wheel 包的提交哈希。可以设置为 `nightly` 以使用主分支上最后一个**已构建**的提交。
- `VLLM_PRECOMPILED_WHEEL_VARIANT`：指定要在 nightly 索引上使用的变体子目录，例如 `cu129`、`cu130`、`cpu`。如果未指定，则根据系统 CUDA 版本（来自 PyTorch 或 nvidia-smi）自动检测变体。您还可以设置 `VLLM_MAIN_CUDA_VERSION` 以覆盖自动检测。

您可以在[安装最新代码](#install-the-latest-code)中找到有关 vLLM wheel 包的更多信息。

!!! note
    您的源代码可能与最新的 vLLM wheel 包具有不同的提交 ID，这可能导致未知错误。
    建议为源代码和已安装的 vLLM wheel 包使用相同的提交 ID。请参考[安装最新代码](#install-the-latest-code)了解如何安装指定 wheel 包的说明。

#### 完整构建（需编译） {#full-build}

如果您想修改 C++ 或 CUDA 代码，则需要从源码构建 vLLM。这可能需要几分钟：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -e . --torch-backend=auto
```

!!! tip
    从源码构建需要大量编译。如果您需要反复从源码构建，缓存编译结果会更高效。

    例如，您可以使用 `conda install ccache` 或 `apt install ccache` 安装 [ccache](https://github.com/ccache/ccache)。
    只要 `which ccache` 命令能找到 `ccache` 二进制文件，构建系统将自动使用它。首次构建后，后续构建将快得多。

    将 `ccache` 与 `pip install -e .` 一起使用时，应运行 `CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .`。这是因为 `pip` 为每次构建创建一个具有随机名称的新文件夹，阻止 `ccache` 识别正在构建相同的文件。

    [sccache](https://github.com/mozilla/sccache) 的工作方式与 `ccache` 类似，但具有在远程存储环境中利用缓存的能力。
    可以设置以下环境变量来配置 vLLM `sccache` 远程：`SCCACHE_BUCKET=vllm-build-sccache SCCACHE_REGION=us-west-2 SCCACHE_S3_NO_CREDENTIALS=1`。我们还建议设置 `SCCACHE_IDLE_TIMEOUT=0`。

!!! note "更快的内核开发"
    对于频繁的 C++/CUDA 内核更改，在完成初始的 `uv pip install -e .` 设置后，请考虑使用[增量编译工作流](../../contributing/incremental_build.md)以显著加快仅修改内核代码的重建速度。

##### 使用现有的 PyTorch 安装

在某些情况下，PyTorch 依赖项无法通过 `uv` 轻松安装，例如使用非默认的 PyTorch 构建（如 nightly 或自定义构建）构建 vLLM 时。

要使用现有的 PyTorch 安装构建 vLLM：

```bash
# 首先安装 PyTorch，可以从 PyPI 或从源码安装
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
uv pip install -r requirements/build/cuda.txt
uv pip install --no-build-isolation -e .
```

或者：如果您专门使用 `uv` 创建和管理虚拟环境，它有[独特的机制](https://docs.astral.sh/uv/concepts/projects/config/#disabling-build-isolation)用于禁用特定包的构建隔离。vLLM 可以利用此机制将 `torch` 指定为禁用构建隔离的包：

```bash
# 首先安装 PyTorch，可以从 PyPI 或从源码安装
git clone https://github.com/vllm-project/vllm.git
cd vllm
# pip install -e . 不能直接工作，只有 uv 可以做到这一点
uv pip install -e .
```

##### 使用本地 cutlass 进行编译

目前，在开始构建过程之前，vLLM 会从 GitHub 获取 cutlass 代码。但是，在某些情况下，您可能想使用本地版本的 cutlass。
为此，您可以设置环境变量 `VLLM_CUTLASS_SRC_DIR` 指向您的本地 cutlass 目录。

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_CUTLASS_SRC_DIR=/path/to/cutlass uv pip install -e . --torch-backend=auto
```

##### 故障排除

为了避免系统过载，您可以通过环境变量 `MAX_JOBS` 限制同时运行的编译任务数量。例如：

```bash
export MAX_JOBS=6
uv pip install -e .
```

这在性能较低的机器上构建时特别有用。例如，当您使用 WSL 时，它默认仅[分配总内存的 50%](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#main-wsl-settings)，因此使用 `export MAX_JOBS=1` 可以避免同时编译多个文件而导致内存不足。
副作用是构建过程会慢得多。

此外，如果您在构建 vLLM 时遇到问题，我们建议使用 NVIDIA PyTorch Docker 镜像。

```bash
# 使用 `--ipc=host` 确保共享内存足够大。
docker run \
    --gpus all \
    -it \
    --rm \
    --ipc=host nvcr.io/nvidia/pytorch:23.10-py3
```

如果您不想使用 Docker，建议完整安装 CUDA Toolkit。您可以从[官方网站](https://developer.nvidia.com/cuda-toolkit-archive)下载并安装它。安装完成后，设置环境变量 `CUDA_HOME` 指向 CUDA Toolkit 的安装路径，并确保 `nvcc` 编译器在您的 `PATH` 中，例如：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
```

以下是一个验证 CUDA Toolkit 是否正确安装的检查：

```bash
nvcc --version # 验证 nvcc 在您的 PATH 中
${CUDA_HOME}/bin/nvcc --version # 验证 nvcc 在您的 CUDA_HOME 中
```

#### 不支持的操作系统构建

vLLM 只能完全在 Linux 上运行，但出于开发目的，您仍然可以在其他系统上构建它（例如 macOS），以便进行导入和更便捷的开发环境。二进制文件将不会被编译，并且在非 Linux 系统上无法工作。

只需在安装前禁用 `VLLM_TARGET_DEVICE` 环境变量：

```bash
export VLLM_TARGET_DEVICE=empty
uv pip install -e .
```

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

vLLM 提供用于部署的官方 Docker 镜像。
该镜像可用于运行兼容 OpenAI 的服务器，并在 Docker Hub 上以 [vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai/tags) 提供。

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-0.6B
```

该镜像也可以与其他容器引擎（如 [Podman](https://podman.io/)）一起使用。

```bash
podman run --device nvidia.com/gpu=all \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HF_TOKEN=$HF_TOKEN" \
-p 8000:8000 \
--ipc=host \
docker.io/vllm/vllm-openai:latest \
--model Qwen/Qwen3-0.6B
```

您可以在镜像标签（`vllm/vllm-openai:latest`）后添加任何其他所需的 [engine-args](https://docs.vllm.ai/en/latest/configuration/engine_args/)。

!!! note
    您可以使用 `ipc=host` 标志或 `--shm-size` 标志来允许容器访问主机的共享内存。vLLM 使用 PyTorch，而 PyTorch 在底层使用共享内存在进程之间共享数据，特别是在张量并行推理中。

!!! note
    可选依赖项未包含在内以避免许可问题（例如 <https://github.com/vllm-project/vllm/issues/8030>）。

    如果您需要使用这些依赖项（已接受许可条款），
    请在基础镜像之上创建一个自定义 Dockerfile，添加一个安装它们的额外层：

    ```Dockerfile
    FROM vllm/vllm-openai:v0.11.0

    # 例如，安装 `audio` 可选依赖项
    # 注意：确保 vLLM 版本与基础镜像匹配！
    RUN uv pip install --system vllm[audio]==0.11.0
    ```

!!! tip
    某些新模型可能仅在 [HF Transformers](https://github.com/huggingface/transformers) 的主分支上可用。

    要使用 `transformers` 的开发版本，请在基础镜像之上创建一个自定义 Dockerfile，添加一个从源码安装其代码的额外层：

    ```Dockerfile
    FROM vllm/vllm-openai:latest

    RUN uv pip install --system git+https://github.com/huggingface/transformers.git
    ```

#### 在具有较旧 CUDA 驱动程序的系统上运行

vLLM 的 Docker 镜像预装了 [CUDA 兼容性库](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)。这允许您在具有比镜像中使用的 CUDA Toolkit 版本更旧的 NVIDIA 驱动程序的系统上运行 vLLM，但仅支持部分专业和数据中心 NVIDIA GPU。

要启用此功能，请在运行容器时将 `VLLM_ENABLE_CUDA_COMPATIBILITY` 环境变量设置为 `1` 或 `true`：

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<secret>" \
    --env "VLLM_ENABLE_CUDA_COMPATIBILITY=1" \
    vllm/vllm-openai <args...>
```

这将自动配置 `LD_LIBRARY_PATH` 在加载 PyTorch 和其他依赖项之前指向兼容性库。

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

您可以通过提供的 [docker/Dockerfile](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile) 从源码构建并运行 vLLM。要构建 vLLM：

```bash
# 可选指定：--build-arg max_jobs=8 --build-arg nvcc_threads=2
DOCKER_BUILDKIT=1 docker build . \
    --target vllm-openai \
    --tag vllm/vllm-openai \
    --file docker/Dockerfile
```

!!! note
    默认情况下，vLLM 将针对所有 GPU 类型进行构建以实现最广泛的分布。如果您只为机器当前运行的 GPU 类型构建，可以添加参数 `--build-arg torch_cuda_arch_list=""` 让 vLLM 查找当前 GPU 类型并为其构建。

    如果您使用 Podman 而不是 Docker，在运行 `podman build` 命令时可能需要通过添加 `--security-opt label=disable` 来禁用 SELinux 标签，以避免某些[现有问题](https://github.com/containers/buildah/discussions/4184)。

!!! note
    如果您没有更改任何 C++ 或 CUDA 内核代码，可以使用预编译 wheel 包显著减少 Docker 构建时间。

    *   **启用该功能**：添加构建参数 `--build-arg VLLM_USE_PRECOMPILED="1"`。
    *   **工作原理**：默认情况下，vLLM 通过与上游 `main` 分支的合并基准提交，自动从我们的[ nightly 构建](https://docs.vllm.ai/en/latest/contributing/ci/nightly_builds/)中找到正确的 wheel 包。
    *   **覆盖提交**：要使用来自特定提交的 wheel 包，请提供 `--build-arg VLLM_PRECOMPILED_WHEEL_COMMIT=<commit_hash>` 参数。

    有关详细说明，请参考 'Set up using Python-only build (without compilation)' 部分的文档，这些参数是类似的。

#### 为 Arm64/aarch64 从源码构建 vLLM 的 Docker 镜像

可以为 aarch64 系统（如 Nvidia Grace-Hopper 和 Grace-Blackwell）构建 Docker 容器。使用 `--platform "linux/arm64"` 标志将构建 arm64 版本。

!!! note
    需要编译多个模块，因此此过程可能需要一段时间。建议使用 `--build-arg max_jobs=` 和 `--build-arg nvcc_threads=` 标志来加速构建过程。但是，请确保您的 `max_jobs` 明显大于 `nvcc_threads` 以获得最大收益。注意并行作业的内存使用可能很大（请参阅下面的示例）。

??? console "命令"

    ```bash
    # 在 Nvidia GH200 服务器上构建的示例。（内存使用：~15GB，构建时间：~1475s / ~25 min，镜像大小：6.93GB）
    DOCKER_BUILDKIT=1 docker build . \
    --file docker/Dockerfile \
    --target vllm-openai \
    --platform "linux/arm64" \
    -t vllm/vllm-gh200-openai:latest \
    --build-arg max_jobs=66 \
    --build-arg nvcc_threads=2 \
    --build-arg torch_cuda_arch_list="9.0 10.0+PTX" \
    --build-arg RUN_WHEEL_CHECK=false
    ```

对于 (G)B300，我们建议使用 CUDA 13，如下所示。

??? console "命令"

    ```bash
    DOCKER_BUILDKIT=1 docker build \
    --build-arg CUDA_VERSION=13.0.2 \
    --build-arg BUILD_BASE_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu22.04 \
    --build-arg max_jobs=256 \
    --build-arg nvcc_threads=2 \
    --build-arg RUN_WHEEL_CHECK=false \
    --build-arg torch_cuda_arch_list='9.0 10.0+PTX' \
    --platform "linux/arm64" \
    --tag vllm/vllm-gb300-openai:latest \
    --target vllm-openai \
    -f docker/Dockerfile \
    .
    ```

!!! note
    如果您在非 ARM 主机（例如 x86_64 机器）上构建 `linux/arm64` 镜像，您需要确保您的系统已设置好使用 QEMU 进行交叉编译。这允许您的主机模拟 ARM64 执行。

    在主机上运行以下命令以注册 QEMU 用户静态处理程序：

    ```bash
    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
    ```

    设置好 QEMU 后，您可以在 `docker build` 命令中使用 `--platform "linux/arm64"` 标志。

#### 使用自定义构建的 vLLM Docker 镜像**

要使用自定义构建的 Docker 镜像运行 vLLM：

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<secret>" \
    vllm/vllm-openai <args...>
```

参数 `vllm/vllm-openai` 指定要运行的镜像，应替换为自定义构建镜像的名称（构建命令中的 `-t` 标签）。

!!! note
    **仅适用于版本 0.4.1 和 0.4.2** - 这些版本下的 vLLM Docker 镜像应以 root 用户运行，因为运行时需要加载位于 root 用户主目录下的库，即 `/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1`。如果您以不同用户运行容器，可能需要先更改该库（以及所有父目录）的权限以允许用户访问，然后使用环境变量 `VLLM_NCCL_SO_PATH=/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1` 运行 vLLM。

--8<-- [end:build-image-from-source]
--8<-- [start:supported-features]

有关功能支持信息，请参阅[功能 x 硬件](../../features/README.md#feature-x-hardware)兼容性矩阵。

--8<-- [end:supported-features]
