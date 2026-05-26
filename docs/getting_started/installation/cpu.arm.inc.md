<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

vLLM 在 ARM CPU 平台上提供基本的模型推理和服务，支持 NEON 以及 FP32、FP16 和 BF16 数据类型。

--8<-- [end:installation]
--8<-- [start:requirements]

- 操作系统：Linux
- 编译器：`gcc/g++ >= 12.3.0`（可选，推荐）
- 指令集架构 (ISA)：需要 NEON 支持

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

自 0.11.2 版本起，提供 Arm 的预构建 vLLM wheel 包。这些 wheel 包包含预编译的 C++ 二进制文件。

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_aarch64.whl --torch-backend cpu
```

??? console "pip"
    ```bash
    pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_aarch64.whl --extra-index-url https://download.pytorch.org/whl/cpu
    ```

!!! warning "设置 `LD_PRELOAD`"
    使用通过 wheel 包安装的 vLLM CPU 之前，请确保已安装 TCMalloc 并将其添加到 `LD_PRELOAD`：
    ```bash
    # 安装 TCMalloc
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # 手动查找路径
    sudo find / -iname *libtcmalloc_minimal.so.4
    TC_PATH=...

    # 将其添加到 LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$LD_PRELOAD"
    ```

`uv` 方法适用于 vLLM `v0.6.6` 及更高版本。`uv` 的一个独特功能是，`--extra-index-url` 中的包具有[比默认索引更高的优先级](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)。如果最新的公开发布版是 `v0.6.6.post1`，`uv` 的行为允许通过指定 `--extra-index-url` 来安装 `v0.6.6.post1` 之前的提交。相比之下，`pip` 会合并 `--extra-index-url` 和默认索引中的包，仅选择最新版本，这使得安装已发布版本之前的开发版本变得困难。

#### 安装最新代码

LLM 推理是一个快速发展的领域，最新代码可能包含尚未发布的错误修复、性能改进和新功能。为了让用户无需等待下一个版本即可尝试最新代码，vLLM 自 `v0.11.2` 起在 <https://wheels.vllm.ai/nightly> 上为每次提交提供预构建的 Arm CPU wheel 包。对于原生 CPU wheel 包，应使用以下索引：

- `https://wheels.vllm.ai/nightly/cpu/vllm`

要从 nightly 索引安装，请运行：

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cpu --index-strategy first-index --torch-backend cpu
```

??? console "pip（有注意事项）"

    使用 `pip` 从 nightly 索引安装是_不支持的_，因为 `pip` 会合并 `--extra-index-url` 和默认索引中的包，仅选择最新版本，这使得安装已发布版本之前的开发版本变得困难。相比之下，`uv` 给予额外索引[比默认索引更高的优先级](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)。

    如果您坚持使用 `pip`，则必须指定 wheel 文件的完整 URL（可从 https://wheels.vllm.ai/nightly/cpu/vllm 获取）。

    ```bash
    pip install https://wheels.vllm.ai/4fa7ce46f31cbd97b4651694caf9991cc395a259/vllm-0.13.0rc2.dev104%2Bg4fa7ce46f.cpu-cp38-abi3-manylinux_2_35_aarch64.whl --extra-index-url https://download.pytorch.org/whl/cpu # 当前的 nightly 构建（文件名会变化！）
    ```

#### 安装特定修订版本

如果您需要访问之前提交的 wheel 包（例如，用于二分查找行为变更、性能回归），可以在 URL 中指定提交哈希：

```bash
export VLLM_COMMIT=730bd35378bf2a5b56b6d3a45be28b3092d26519 # 使用主分支的完整提交哈希
uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/cpu --index-strategy first-index --torch-backend cpu
```

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

首先，安装推荐的编译器。建议使用 `gcc/g++ >= 12.3.0` 作为默认编译器以避免潜在问题。例如，在 Ubuntu 22.4 上，您可以运行：

```bash
sudo apt-get update  -y
sudo apt-get install -y --no-install-recommends ccache git curl wget ca-certificates gcc-12 g++-12 libtcmalloc-minimal4 libnuma-dev ffmpeg libsm6 libxext6 libgl1 jq lsof
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

其次，克隆 vLLM 项目：

```bash
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
```

第三，安装所需的依赖项：

```bash
uv pip install -r requirements/build/cpu.txt --torch-backend cpu
uv pip install -r requirements/cpu.txt --torch-backend cpu
```

??? console "pip"
    ```bash
    pip install --upgrade pip
    pip install -v -r requirements/build/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
    pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
    ```

最后，构建并安装 vLLM：

```bash
VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation
```

如果您想开发 vLLM，请改用可编辑模式安装。

```bash
VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
```

兼容性测试已在 AWS Graviton3 实例上进行。

!!! warning "设置 `LD_PRELOAD`"
    使用通过 wheel 包安装的 vLLM CPU 之前，请确保已安装 TCMalloc 并将其添加到 `LD_PRELOAD`：
    ```bash
    # 安装 TCMalloc
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # 手动查找路径
    sudo find / -iname *libtcmalloc_minimal.so.4
    TC_PATH=...

    # 将其添加到 LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$LD_PRELOAD"
    ```

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

从 Docker Hub 拉取最新镜像：

```bash
docker pull vllm/vllm-openai-cpu:latest-arm64
```

要拉取特定 vLLM 版本的镜像：

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
docker pull vllm/vllm-openai-cpu:v${VLLM_VERSION}-arm64
```

所有可用的镜像标签在这里：[https://hub.docker.com/r/vllm/vllm-openai-cpu/tags](https://hub.docker.com/r/vllm/vllm-openai-cpu/tags)。

您可以通过以下方式运行这些镜像：

```bash
docker run \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<secret>" \
    vllm/vllm-openai-cpu:latest-arm64 <args...>
```

您还可以通过 Docker 镜像访问最新代码。这些镜像不适用于生产环境，仅用于 CI 和测试。它们将在几天后过期。

最新代码可能包含错误且可能不稳定，请谨慎使用。

```bash
export VLLM_COMMIT=6299628d326f429eba78736acb44e76749b281f5 # 使用主分支的完整提交哈希
docker pull public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:${VLLM_COMMIT}-arm64-cpu
```

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

#### 为您的目标 ARM CPU 构建

```bash
docker build -f docker/Dockerfile.cpu \
        --platform=linux/arm64 \
        --build-arg VLLM_CPU_ARM_BF16=<false (default)|true> \
        --tag vllm-cpu-env \
        --target vllm-openai .
```

!!! note "默认自动检测"
    默认情况下，ARM CPU 指令集（BF16、NEON 等）会从构建系统的 CPU 标志自动检测。`VLLM_CPU_ARM_BF16` 构建参数用于交叉编译：

    - `VLLM_CPU_ARM_BF16=true` - 强制启用 ARM BF16 支持（无论构建系统能力如何，都使用 BF16 构建）
    - `VLLM_CPU_ARM_BF16=false` - 依赖自动检测（默认）

##### 示例

###### 自动检测构建（原生 ARM）

```bash
# 在 ARM64 系统上构建 - 平台自动检测
docker build -f docker/Dockerfile.cpu \
        --tag vllm-cpu-arm64 \
        --target vllm-openai .
```

###### 为支持 BF16 的 ARM 交叉编译

```bash
# 在 ARM64 上为更新支持 BF16 的 ARM CPU 构建
docker build -f docker/Dockerfile.cpu \
        --build-arg VLLM_CPU_ARM_BF16=true \
        --tag vllm-cpu-arm64-bf16 \
        --target vllm-openai .
```

###### 从 x86_64 向 ARM64 交叉编译（带 BF16）

```bash
# 需要 Docker buildx 及 ARM 仿真（QEMU）
docker buildx build -f docker/Dockerfile.cpu \
        --platform=linux/arm64 \
        --build-arg VLLM_CPU_ARM_BF16=true \
        --build-arg max_jobs=4 \
        --tag vllm-cpu-arm64-bf16 \
        --target vllm-openai \
        --load .
```

!!! note "ARM BF16 要求"
    ARM BF16 支持需要 ARMv8.6-A 或更高版本（FEAT_BF16）。在 AWS Graviton3/4、AmpereOne 及其他较新的 ARM 处理器上支持。

#### 启动 OpenAI 服务器

```bash
docker run --rm \
            --security-opt seccomp=unconfined \
            --cap-add SYS_NICE \
            --shm-size=4g \
            -p 8000:8000 \
            -e VLLM_CPU_KVCACHE_SPACE=<KV cache space> \
            -e VLLM_CPU_OMP_THREADS_BIND=<CPU cores for inference> \
            vllm-cpu-arm64 \
            meta-llama/Llama-3.2-1B-Instruct \
            --dtype=bfloat16 \
            other vLLM OpenAI server arguments
```

!!! tip "`--privileged` 的替代方案"
    使用 `--cap-add SYS_NICE --security-opt seccomp=unconfined` 替代 `--privileged=true` 以获得更好的安全性。

--8<-- [end:build-image-from-source]
--8<-- [start:extra-information]
--8<-- [end:extra-information]
