<!-- markdownlint-disable MD041 MD051 -->
--8<-- [start:installation]

vLLM 支持使用 ROCm 6.3 或更高版本的 AMD GPU。提供针对 ROCm 7.0 和 ROCm 7.2.1 的预构建 wheel 包。

#### 预构建的 Wheel 包

| ROCm 变体 | Python 版本 | ROCm 版本 | glibc 要求 | 支持的版本 |
| ------------ | -------------- | ------------ | ----------------- | ------------------ |
| `rocm700` | 3.12 | 7.0 | >= 2.35 | `0.14.0` 至 `0.18.0` |
| `rocm721` | 3.12 | 7.2.1 | >= 2.35 | 提交 `171775f306a333a9cf105bfd533bf3e113d401d9` 之后的 nightly 版本 |

--8<-- [end:installation]
--8<-- [start:requirements]

- GPU：MI200s (gfx90a)、MI300 (gfx942)、MI350 (gfx950)、Radeon RX 7900 系列 (gfx1100/1101)、Radeon RX 9000 系列 (gfx1200/1201)、Ryzen AI MAX / AI 300 系列 (gfx1151/1150)
- ROCm 6.3 或更高版本
    - MI350 需要 ROCm 7.0 或更高版本
    - Ryzen AI MAX / AI 300 系列需要 ROCm 7.0.2 或更高版本

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

vLLM wheel 包捆绑了 PyTorch 和所有必需的依赖项，您应该使用附带的 PyTorch 以确保兼容性。由于 vLLM 编译了许多 ROCm 内核以确保经过验证的高性能堆栈，生成的二进制文件可能与其他 ROCm 或 PyTorch 构建不兼容。
如果您需要不同的 ROCm 版本或想使用现有的 PyTorch 安装，则需要从源码构建 vLLM。更多详情请参见[下文](#build-wheel-from-source)。

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

要安装适用于 Python 3.12、ROCm 7.0 和 `glibc >= 2.35` 的最新版本 vLLM。

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/ --upgrade
```

!!! tip
    您可以通过检查 extra-index-url 中的 `vllm` 包来了解最新 vLLM 支持哪个 ROCm 版本，查看 <https://wheels.vllm.ai/rocm/> 下的 [https://wheels.vllm.ai/rocm/vllm](https://wheels.vllm.ai/rocm/vllm)。

    另一种方法是使用以下命令自动提取 wheel 变体：

    ```bash
    # 自动提取可用的 rocm 变体
    export VLLM_ROCM_VARIANT=$(curl -s https://wheels.vllm.ai/rocm/vllm | grep -oP 'rocm\d+' | head -1)

    # 自动提取 vLLM 版本
    export VLLM_VERSION=$(curl -s https://wheels.vllm.ai/rocm/vllm | grep -oP 'vllm-\K[0-9.]+' | head -1)

    # 检查 ROCm 版本是否与您的环境兼容
    echo $VLLM_ROCM_VARIANT
    echo $VLLM_VERSION
    ```

要安装特定版本和 ROCm 变体的 vLLM wheel 包。

```bash
# 版本不带 `v`
uv pip install vllm==${VLLM_VERSION} --extra-index-url https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VLLM_ROCM_VARIANT}

# 示例
uv pip install vllm==0.18.0 --extra-index-url https://wheels.vllm.ai/rocm/0.18.0/rocm700
```

!!! warning "使用 `pip` 的注意事项"

    我们建议使用 `uv` 安装 vLLM wheel 包。使用 `pip` 从自定义索引安装很麻烦，因为 `pip` 会合并 `--extra-index-url` 和默认索引中的包，仅选择最新版本。这使得从自定义索引安装 wheel 包变得困难，除非指定了所有包的确切版本。相比之下，`uv` 给予额外索引[比默认索引更高的优先级](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)。

    如果您坚持使用 `pip`，则需要在包名称中指定确切的 vLLM 版本，并通过 `--extra-index-url` 提供自定义索引 URL `https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VLLM_ROCM_VARIANT}`。

    ```bash
    pip install vllm==0.18.0+rocm700 --extra-index-url https://wheels.vllm.ai/rocm/0.18.0/rocm700
    ```

#### 安装最新代码

LLM 推理是一个快速发展的领域，最新代码可能包含尚未发布的错误修复、性能改进和新功能。为了让用户无需等待下一个版本即可尝试最新代码，vLLM 自提交 `171775f306a333a9cf105bfd533bf3e113d401d9` 起在 <https://wheels.vllm.ai/rocm/nightly/> 上为每次提交提供 wheel 包。要使用的自定义索引是 `https://wheels.vllm.ai/rocm/nightly/${VLLM_ROCM_VARIANT}`

**注意：** 第一个支持 nightly wheel 的 ROCm 变体是 ROCm 7.2.1

要从最新的 nightly 索引安装，请运行：

```bash
# 自动提取可用的 rocm 变体
export VLLM_ROCM_VARIANT=$(curl -s https://wheels.vllm.ai/rocm/nightly | \
    grep -oP 'rocm\d+' | head -1  | sed 's/%2B/+/g')

# 检查 ROCm 版本是否与您的环境兼容
echo $VLLM_ROCM_VARIANT

uv pip install --pre vllm \
    --extra-index-url https://wheels.vllm.ai/rocm/nightly/${VLLM_ROCM_VARIANT} \
    --index-strategy unsafe-best-match
```

##### 安装特定修订版本

如果您需要访问之前提交的 wheel 包（例如，用于二分查找行为变更、性能回归），可以在 URL 中指定提交哈希，示例：

```bash
export VLLM_COMMIT=5b8c30d62b754b575e043ce2fc0dcbf8a64f6306

export VLLM_ROCM_VARIANT=$(curl -s https://wheels.vllm.ai/rocm/${VLLM_COMMIT} | \
    grep -oP 'rocm\d+' | head -1  | sed 's/%2B/+/g')

# 从 wheel URL 提取版本
export VLLM_VERSION=$(curl -s https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}/vllm/ | \
    grep -oP 'vllm-\K[^-]+' | head -1  | sed 's/%2B/+/g')

# 检查版本是否与您环境的 ROCm 版本兼容
echo $VLLM_ROCM_VARIANT
echo $VLLM_VERSION

uv pip install vllm==${VLLM_VERSION} \
  --extra-index-url https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT} \
  --index-strategy unsafe-best-match
```

!!! warning "`pip` 注意事项"

    使用 `pip` 从 nightly 索引安装是_不支持的_，因为 `pip` 会合并 `--extra-index-url` 和默认索引中的包，仅选择最新版本，这使得安装已发布版本之前的开发版本变得困难。相比之下，`uv` 给予额外索引[比默认索引更高的优先级](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)。

    如果您坚持使用 `pip`，则需要在包名称中指定确切的 vLLM 版本，并提供自定义索引 URL（可从网页获取）。

    ```bash
    export VLLM_COMMIT=5b8c30d62b754b575e043ce2fc0dcbf8a64f6306

    export VLLM_ROCM_VARIANT=$(curl -s https://wheels.vllm.ai/rocm/${VLLM_COMMIT} | \
        grep -oP 'rocm\d+' | head -1  | sed 's/%2B/+/g')

    # 从 wheel URL 提取版本
    export VLLM_VERSION=$(curl -s https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}/vllm/ | \
        grep -oP 'vllm-\K[^-]+' | head -1  | sed 's/%2B/+/g')

    # 检查版本是否与您环境的 ROCm 版本兼容
    echo $VLLM_ROCM_VARIANT
    echo $VLLM_VERSION

    pip install vllm==${VLLM_VERSION} \
    --extra-index-url https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}
    ```

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

!!! tip
    - 如果您发现以下安装步骤不适用于您，请参考 [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base)。Dockerfile 是一种安装步骤的形式。

0. 安装先决条件（如果您已经在已安装以下内容的环境/docker 中，请跳过此步骤）：

    - [ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
    - [PyTorch](https://pytorch.org/)

    要安装 PyTorch，您可以从一个全新的 docker 镜像开始，例如 `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0`、`rocm/pytorch-nightly`。如果您使用 docker 镜像，可以跳到步骤 3。

    或者，您可以使用 PyTorch wheel 包安装 PyTorch。您可以在 PyTorch [入门指南](https://pytorch.org/get-started/locally/)中查看 PyTorch 安装指南。示例：

    ```bash
    # 安装 PyTorch
    pip uninstall torch -y
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0
    ```

1. 安装 [Triton for ROCm](https://github.com/ROCm/triton.git)

    按照 [ROCm/triton](https://github.com/ROCm/triton.git) 的说明安装 ROCm 的 Triton

    ```bash
    python3 -m pip install ninja cmake wheel pybind11
    pip uninstall -y triton
    git clone https://github.com/ROCm/triton.git
    cd triton
    # git checkout $TRITON_BRANCH
    git checkout f9e5bf54
    if [ ! -f setup.py ]; then cd python; fi
    python3 setup.py install
    cd ../..
    ```

    !!! note
        - 已验证的 `$TRITON_BRANCH` 可在 [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) 中找到。
        - 如果您在构建 triton 期间看到与下载包相关的 HTTP 问题，请重试，因为 HTTP 错误是间歇性的。

2. 可选，如果您选择使用 CK flash attention，可以安装 [flash attention for ROCm](https://github.com/Dao-AILab/flash-attention.git)

    按照 [ROCm/flash-attention](https://github.com/Dao-AILab/flash-attention#amd-rocm-support) 的说明安装 ROCm 的 flash attention (v2.8.0)

    例如，对于 ROCm 7.0，假设您的 gfx 架构是 `gfx942`。要获取您的 gfx 架构，请运行 `rocminfo |grep gfx`。

    ```bash
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    # git checkout $FA_BRANCH
    git checkout 0e60e394
    git submodule update --init
    GPU_ARCHS="gfx942" python3 setup.py install
    cd ..
    ```

    !!! note
        - 已验证的 `$FA_BRANCH` 可在 [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) 中找到。

3. 可选，如果您选择自行构建 AITER 以使用特定分支或提交，可以使用以下步骤构建 AITER：

    ```bash
    python3 -m pip uninstall -y aiter
    git clone --recursive https://github.com/ROCm/aiter.git
    cd aiter
    git checkout $AITER_BRANCH_OR_COMMIT
    git submodule sync; git submodule update --init --recursive
    python3 setup.py develop
    ```

    !!! note
        - 您需要根据您的目的配置 `$AITER_BRANCH_OR_COMMIT`。
        - 已验证的 `$AITER_BRANCH_OR_COMMIT` 可在 [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) 中找到。

4. 可选，如果您想使用 MORI 进行 EP 或 PD 分离，可以使用以下步骤安装 [MORI](https://github.com/ROCm/mori)：

    ```bash
    git clone https://github.com/ROCm/mori.git
    cd mori
    git checkout $MORI_BRANCH_OR_COMMIT
    git submodule sync; git submodule update --init --recursive
    MORI_GPU_ARCHS="gfx942;gfx950" python3 setup.py install
    ```

    !!! note
        - 您需要根据您的目的配置 `$MORI_BRANCH_OR_COMMIT`。
        - 已验证的 `$MORI_BRANCH_OR_COMMIT` 可在 [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) 中找到。

5. 构建 vLLM。例如，ROCm 7.0 上的 vLLM 可以通过以下步骤构建：

    ???+ console "命令"

        ```bash
        pip install --upgrade pip

        # 构建并安装 AMD SMI
        pip install /opt/rocm/share/amd_smi

        # 安装依赖项
        pip install --upgrade numba \
            scipy \
            huggingface-hub[cli] \
            setuptools_scm
        pip install -r requirements/rocm.txt

        # 为单一架构构建（例如 MI300）以加快安装速度（推荐）：
        export PYTORCH_ROCM_ARCH="gfx942"

        # 要为多个架构构建 vLLM（MI210/MI250/MI300），请使用此命令
        # export PYTORCH_ROCM_ARCH="gfx90a;gfx942"

        python3 setup.py develop
        ```

    这可能需要 5-10 分钟。目前，从源码安装 vLLM 时，`pip install .` 不适用于 ROCm。

    !!! tip
        - ROCm 版本的 PyTorch 理想情况下应与 ROCm 驱动程序版本匹配。

!!! tip
    - 对于 MI300x (gfx942) 用户，为获得最佳性能，请参考 [MI300x 调优指南](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html)了解系统和工作负载级别的性能优化和调优提示。
      对于 vLLM，请参考 [vLLM 性能优化](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)。

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

vLLM 提供用于部署的官方 Docker 镜像。
这些镜像可用于运行兼容 OpenAI 的服务器，并在 Docker Hub 上以 [vllm/vllm-openai-rocm](https://hub.docker.com/r/vllm/vllm-openai-rocm/tags) 提供。

- `vllm/vllm-openai-rocm:latest` — 稳定版本
- `vllm/vllm-openai-rocm:nightly` — 来自最新开发分支的预览版本，如果您想要最新功能和修复，请使用此版本

```bash
docker run --rm \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai-rocm:<tag> \
    --model Qwen/Qwen3-0.6B
```

要将 docker 镜像作为开发基础使用，您可以通过覆盖入口点以交互式会话方式启动它。

???+ console "命令"
    ```bash
    docker run --rm -it \
        --group-add=video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --device /dev/kfd \
        --device /dev/dri \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HF_TOKEN=$HF_TOKEN" \
        --network=host \
        --ipc=host \
        --entrypoint /bin/bash \
        vllm/vllm-openai-rocm:<tag>
    ```

#### 使用 AMD 的 Docker 镜像（已弃用）

!!! warning "已弃用"
    AMD 的 Docker 镜像（`rocm/vllm` 和 `rocm/vllm-dev`）已弃用，建议使用上述官方 vLLM Docker 镜像（`vllm/vllm-openai-rocm`）。请迁移到官方镜像。

在 2026 年 1 月 20 日官方 docker 镜像在[上游 vLLM docker hub](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/) 上可用之前，[AMD Infinity hub for vLLM](https://hub.docker.com/r/rocm/vllm/tags) 提供了一个预构建的优化 docker 镜像，用于验证 AMD Instinct MI300X™ 加速器上的推理性能。
AMD 还从 [Docker Hub](https://hub.docker.com/r/rocm/vllm-dev) 提供了 nightly 预构建 docker 镜像，其中包含 vLLM 及其所有依赖项。此 docker 镜像的入口点是 `/bin/bash`（与 vLLM 的官方 Docker 镜像不同）。

!!! tip
    请查看 [AMD Instinct MI300X 上的 LLM 推理性能验证](https://rocm.docs.amd.com/en/latest/how-to/performance-validation/mi300x/vllm-benchmark.html)
    了解如何使用此预构建 docker 镜像的说明。

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

您可以通过提供的 [docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) 从源码构建并运行 vLLM。

??? info "（可选）构建带有 ROCm 软件栈的镜像"

    从 [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) 构建一个 docker 镜像，该镜像设置 vLLM 所需的 ROCm 软件栈。
    **此步骤是可选的，因为此 rocm_base 镜像通常是预构建的，并存储在 [Docker Hub](https://hub.docker.com/r/rocm/vllm-dev) 标签 `rocm/vllm-dev:base` 下，以加快用户体验。**
    如果您选择自行构建此 rocm_base 镜像，步骤如下。

    用户使用 buildkit 启动 docker 构建非常重要。用户可以在调用 docker build 命令时将 `DOCKER_BUILDKIT=1` 作为环境变量，或者需要在 docker 守护进程配置 `/etc/docker/daemon.json` 中设置 buildkit 如下，然后重启守护进程：

    ```json
    {
        "features": {
            "buildkit": true
        }
    }
    ```

    要为 MI200 和 MI300 系列在 ROCm 7.0 上构建 vllm，您可以使用默认设置：

    ```bash
    DOCKER_BUILDKIT=1 docker build \
        -f docker/Dockerfile.rocm_base \
        -t rocm/vllm-dev:base .
    ```

首先，从 [docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) 构建一个 docker 镜像，然后从该镜像启动一个 docker 容器。
用户使用 buildkit 启动 docker 构建非常重要。用户可以在调用 docker build 命令时将 `DOCKER_BUILDKIT=1` 作为环境变量，或者需要在 docker 守护进程配置 /etc/docker/daemon.json 中设置 buildkit 如下，然后重启守护进程：

```json
{
    "features": {
        "buildkit": true
    }
}
```

[docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) 默认使用 ROCm 7.0，但在较旧的 vLLM 分支中也支持 ROCm 5.7、6.0、6.1、6.2、6.3 和 6.4。
它提供了使用以下参数自定义 docker 镜像构建的灵活性：

- `BASE_IMAGE`：指定运行 `docker build` 时使用的基础镜像。默认值 `rocm/vllm-dev:base` 是由 AMD 发布和维护的镜像。它使用 [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) 构建
- `ARG_PYTORCH_ROCM_ARCH`：允许覆盖基础 docker 镜像中的 gfx 架构值

这些值可以在运行 `docker build` 时通过 `--build-arg` 选项传入。

要为 MI200 和 MI300 系列在 ROCm 7.0 上构建 vllm，您可以使用默认设置（构建一个以 `vllm serve` 为入口点的 docker 镜像）：

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm/vllm-openai-rocm .
```

要使用自定义构建的 Docker 镜像运行 vLLM：

```bash
docker run --rm \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai-rocm <args...>
```

参数 `vllm/vllm-openai-rocm` 指定要运行的镜像，应替换为自定义构建镜像的名称（构建命令中的 `-t` 标签）。

要将 docker 镜像作为开发基础使用，您可以通过覆盖入口点以交互式会话方式启动它。

???+ console "命令"
    ```bash
    docker run --rm -it \
        --group-add=video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --device /dev/kfd \
        --device /dev/dri \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HF_TOKEN=$HF_TOKEN" \
        --network=host \
        --ipc=host \
        --entrypoint bash \
        vllm/vllm-openai-rocm
    ```

--8<-- [end:build-image-from-source]
--8<-- [start:supported-features]

有关功能支持信息，请参阅[功能 x 硬件](../../features/README.md#feature-x-hardware)兼容性矩阵。

--8<-- [end:supported-features]
