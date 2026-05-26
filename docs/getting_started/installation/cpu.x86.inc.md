<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

vLLM 支持在 x86 CPU 平台上进行基本的模型推理和服务，支持 FP32、FP16 和 BF16 数据类型。

--8<-- [end:installation]
--8<-- [start:requirements]

- 操作系统：Linux
- CPU 标志：`avx512f`（推荐）、`avx2`（功能有限）

!!! tip
    使用 `lscpu` 检查 CPU 标志。

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

自 0.17.0 版本起，提供支持 AVX512/AVX2 的 x86 预构建 vLLM wheel 包。要安装发布版的 wheel 包：

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')

# 使用 uv
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl --torch-backend cpu
```

??? console "pip"
    ```bash
    # 使用 pip
    pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cpu
    ```
!!! warning "设置 `LD_PRELOAD`"
    使用通过 wheel 包安装的 vLLM CPU 之前，请确保已安装 TCMalloc 和 Intel OpenMP 并将其添加到 `LD_PRELOAD`：
    ```bash
    # 安装 TCMalloc，Intel OpenMP 随 vLLM CPU 一起安装
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # 手动查找路径
    sudo find / -iname *libtcmalloc_minimal.so.4
    sudo find / -iname *libiomp5.so
    TC_PATH=...
    IOMP_PATH=...

    # 将其添加到 LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$IOMP_PATH:$LD_PRELOAD"
    ```

#### 安装最新代码

要安装从最新主分支构建的 wheel 包：

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cpu --index-strategy first-index --torch-backend cpu
```

#### 安装特定修订版本

如果您需要访问之前提交的 wheel 包（例如，用于二分查找行为变更、性能回归），可以在 URL 中指定提交哈希：

```bash
export VLLM_COMMIT=730bd35378bf2a5b56b6d3a45be28b3092d26519 # 使用主分支的完整提交哈希
uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/cpu --index-strategy first-index --torch-backend cpu
```

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

安装推荐的编译器。建议使用 `gcc/g++ >= 12.3.0` 作为默认编译器，以避免潜在问题。例如，在 Ubuntu 22.4 上，您可以运行：

```bash
sudo apt-get update -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

克隆 vLLM 项目：

```bash
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
```

安装所需的依赖项：

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

构建并安装 vLLM：

```bash
VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation
```

如果您想开发 vLLM，请改用可编辑模式安装。

```bash
VLLM_TARGET_DEVICE=cpu python3 setup.py develop
```

可选地，构建一个可移植的 wheel 包以便在其他地方安装：

```bash
VLLM_TARGET_DEVICE=cpu uv build --wheel --no-build-isolation
```

```bash
uv pip install dist/*.whl
```

??? console "pip"
    ```bash
    VLLM_TARGET_DEVICE=cpu python -m build --wheel --no-isolation
    ```

    ```bash
    pip install dist/*.whl
    ```

!!! warning "设置 `LD_PRELOAD`"
    使用通过 wheel 包安装的 vLLM CPU 之前，请确保已安装 TCMalloc 和 Intel OpenMP 并将其添加到 `LD_PRELOAD`：
    ```bash
    # 安装 TCMalloc，Intel OpenMP 随 vLLM CPU 一起安装
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # 手动查找路径
    sudo find / -iname *libtcmalloc_minimal.so.4
    sudo find / -iname *libiomp5.so
    TC_PATH=...
    IOMP_PATH=...

    # 将其添加到 LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$IOMP_PATH:$LD_PRELOAD"
    ```

!!! example "故障排除"
    - **NumPy ≥2.0 错误**：使用 `pip install "numpy<2.0"` 降级。
    - **CMake 检测到 CUDA**：添加 `CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON` 以在 CPU 构建期间阻止 CUDA 检测，即使已安装 CUDA。
    - `AMD` 需要至少第 4 代处理器（Zen 4/Genoa）或更高版本以支持 [AVX512](https://www.phoronix.com/review/amd-zen4-avx512) 来运行 vLLM CPU。
    - 如果您收到类似错误：`Could not find a version that satisfies the requirement torch==X.Y.Z+cpu+cpu`，请考虑更新 [pyproject.toml](https://github.com/vllm-project/vllm/blob/main/pyproject.toml) 以帮助 pip 解析依赖项。
    ```toml title="pyproject.toml"
    [build-system]
    requires = [
      "cmake>=3.26.1",
      ...
      "torch==X.Y.Z+cpu"   # <-------
    ]
    ```

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

您可以从 Docker Hub 拉取最新的可用 CPU 镜像：

```bash
docker pull vllm/vllm-openai-cpu:latest-x86_64
```

要拉取特定 vLLM 版本的镜像：

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
docker pull vllm/vllm-openai-cpu:v${VLLM_VERSION}-x86_64
```

所有可用的镜像标签在这里：[https://hub.docker.com/r/vllm/vllm-openai-cpu/tags](https://hub.docker.com/r/vllm/vllm-openai-cpu/tags)

您可以通过以下方式运行这些镜像：

```bash
docker run \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<secret>" \
    vllm/vllm-openai-cpu:latest-x86_64 <args...>
```

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

#### 为您的目标 CPU 构建

```bash
docker build -f docker/Dockerfile.cpu \
        --build-arg VLLM_CPU_X86=<false (default)|true> \ # 用于交叉编译
        --tag vllm-cpu-env \
        --target vllm-openai .
```

#### 启动 OpenAI 服务器

```bash
docker run --rm \
            --security-opt seccomp=unconfined \
            --cap-add SYS_NICE \
            --shm-size=4g \
            -p 8000:8000 \
            -e VLLM_CPU_KVCACHE_SPACE=<KV cache space> \
            vllm-cpu-env \
            meta-llama/Llama-3.2-1B-Instruct \
            --dtype=bfloat16 \
            other vLLM OpenAI server arguments
```

--8<-- [end:build-image-from-source]
--8<-- [start:extra-information]
--8<-- [end:extra-information]
