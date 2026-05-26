# 增量编译工作流程

当处理位于 `csrc/` 目录中的 vLLM C++/CUDA 内核时，每次更改都使用 `uv pip install -e .` 重新编译整个项目可能非常耗时。使用 CMake 的增量编译工作流程允许在初始设置后仅重新编译必要的组件，从而加快迭代速度。本指南详细说明了如何设置和使用此类工作流程，该工作流程是对您的可编辑 Python 安装的补充。

## 前提条件

在设置增量构建之前：

1. **vLLM 可编辑安装：** 确保您已从源码以可编辑模式安装了 vLLM。初始可编辑设置使用预编译 wheel 可以更快，因为 CMake 工作流程将处理后续的内核重新编译。

    ```console
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
    ```

2. **CUDA 工具包：** 确认 NVIDIA CUDA 工具包已正确安装，并且 `nvcc` 可在 `PATH` 中访问。CMake 依赖 `nvcc` 来编译 CUDA 代码。您通常可以在 `$CUDA_HOME/bin/nvcc` 或通过运行 `which nvcc` 找到 `nvcc`。如果遇到问题，请参考[官方 CUDA 工具包安装指南](https://developer.nvidia.com/cuda-toolkit-archive)和 vLLM 的主要 [GPU 安装文档](../getting_started/installation/gpu.md#troubleshooting)进行故障排除。`CMakeUserPresets.json` 中的 `CMAKE_CUDA_COMPILER` 变量也应指向您的 `nvcc` 二进制文件。

3. **构建工具：** 强烈建议安装 `ccache`，通过缓存编译结果来加快重新构建速度（例如，`sudo apt install ccache` 或 `conda install ccache`）。同时，确保安装了 `cmake` 和 `ninja` 等核心构建依赖项。这些可以通过 `requirements/build/cuda.txt` 或系统的包管理器安装。

    ```console
    uv pip install -r requirements/build/cuda.txt --torch-backend=auto
    ```

## 设置 CMake 构建环境

增量构建过程通过 CMake 管理。您可以使用位于 vLLM 仓库根目录的 `CMakeUserPresets.json` 文件来配置构建设置。

### 使用辅助脚本生成 `CMakeUserPresets.json`

为简化设置，vLLM 提供了一个辅助脚本，该脚本尝试自动检测系统配置（如 CUDA 路径、Python 环境和 CPU 核心数），并为您生成 `CMakeUserPresets.json` 文件。

**运行脚本：**

导航到 vLLM 克隆的根目录并执行以下命令：

```console
python tools/generate_cmake_presets.py
```

如果脚本无法自动确定某些路径（例如 `nvcc` 或用于您 vLLM 开发环境的特定 Python 可执行文件），它将提示您输入信息。请按照屏幕提示操作。如果发现现有的 `CMakeUserPresets.json`，脚本将在覆盖前请求确认。

**强制覆盖现有文件：**

要自动覆盖现有的 `CMakeUserPresets.json` 而不提示，请使用 `--force-overwrite` 标志：

```console
python tools/generate_cmake_presets.py --force-overwrite
```

这在自动化脚本或 CI/CD 环境（不需要交互式提示）中特别有用。

运行脚本后，将在 vLLM 仓库的根目录创建一个 `CMakeUserPresets.json` 文件。

### `CMakeUserPresets.json` 示例

以下是生成的 `CMakeUserPresets.json` 可能的样子示例。脚本将根据您的系统和您提供的任何输入来定制这些值。

```json
{
    "version": 6,
    "cmakeMinimumRequired": {
        "major": 3,
        "minor": 26,
        "patch": 1
    },
    "configurePresets": [
        {
            "name": "release",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/cmake-build-release",
            "cacheVariables": {
                "CMAKE_CUDA_COMPILER": "/usr/local/cuda/bin/nvcc",
                "CMAKE_C_COMPILER_LAUNCHER": "ccache",
                "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
                "CMAKE_CUDA_COMPILER_LAUNCHER": "ccache",
                "CMAKE_BUILD_TYPE": "Release",
                "VLLM_PYTHON_EXECUTABLE": "/home/user/venvs/vllm/bin/python",
                "CMAKE_INSTALL_PREFIX": "${sourceDir}",
                "CMAKE_CUDA_FLAGS": "",
                "NVCC_THREADS": "4",
                "CMAKE_JOB_POOLS": "compile=32"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "release",
            "configurePreset": "release",
            "jobs": 32
        }
    ]
}
```

**各种配置的含义是什么？**

- `CMAKE_CUDA_COMPILER`：指向您的 `nvcc` 二进制文件的路径。脚本会尝试自动找到此路径。
- `CMAKE_C_COMPILER_LAUNCHER`、`CMAKE_CXX_COMPILER_LAUNCHER`、`CMAKE_CUDA_COMPILER_LAUNCHER`：将这些设置为 `ccache`（或 `sccache`）可以通过缓存编译结果显著加快重新构建速度。确保已安装 `ccache`（例如，`sudo apt install ccache` 或 `conda install ccache`）。脚本默认设置这些值。
- `VLLM_PYTHON_EXECUTABLE`：您的 vLLM 开发环境中 Python 可执行文件的路径。脚本会提示输入此信息，如果合适，默认为当前 Python 环境。
- `CMAKE_INSTALL_PREFIX: "${sourceDir}"`：指定编译后的组件应安装回您的 vLLM 源码目录。这对于可编辑安装至关重要，因为它使新构建的内核立即可用于您的 Python 环境。
- `CMAKE_JOB_POOLS` 和构建预设中的 `jobs`：控制构建的并行度。脚本根据检测到的系统 CPU 核心数设置这些值。
- `binaryDir`：指定构建产物的存储位置（例如 `cmake-build-release`）。

## 使用 CMake 构建和安装

配置好 `CMakeUserPresets.json` 后：

1. **初始化 CMake 构建环境：**
   此步骤根据您选择的预设（例如 `release`）配置构建系统，并在 `binaryDir` 创建构建目录。

    ```console
    cmake --preset release
    ```

2. **构建并安装 vLLM 组件：**
   此命令编译代码并将生成的二进制文件安装到您的 vLLM 源码目录中，使其可用于您的可编辑 Python 安装。

    ```console
    cmake --build --preset release --target install
    ```

3. **进行更改并重复！**
   现在您可以使用 vLLM 的可编辑安装，根据需要进行测试和更改。如果需要重新构建以更新更改，只需再次运行 CMake 命令，它将仅编译受影响文件。

    ```console
    cmake --build --preset release --target install
    ```

## 验证构建

成功构建后，您将看到一个填充好的构建目录（例如，如果您使用了 `release` 预设和示例配置，则会在 `cmake-build-release/`）。

```console
> ls cmake-build-release/
bin             cmake_install.cmake      _deps                                machete_generation.log
build.ninja     CPackConfig.cmake        detect_cuda_compute_capabilities.cu  marlin_generation.log
_C.abi3.so      CPackSourceConfig.cmake  detect_cuda_version.cc               _moe_C.abi3.so
CMakeCache.txt  ctest                    _flashmla_C.abi3.so                  moe_marlin_generation.log
CMakeFiles      cumem_allocator.abi3.so  install_local_manifest.txt           vllm-flash-attn
```

`cmake --build ... --target install` 命令将编译后的共享库（如 `_C.abi3.so`、`_moe_C.abi3.so` 等）复制到源码树中相应的 `vllm` 包目录中。这将使用新编译的内核更新您的可编辑安装。

## 额外提示

- **调整并行度：** 在 `CMakeUserPresets.json` 中微调 `configurePresets` 的 `CMAKE_JOB_POOLS` 和 `buildPresets` 的 `jobs`。任务过多可能会使 RAM 或 CPU 核心有限的系统过载，导致构建速度变慢或系统不稳定。任务过少则无法充分利用可用资源。
- **必要时进行清理构建：** 如果您遇到持续存在或奇怪的构建错误，特别是在进行重大更改或切换分支后，请考虑删除 CMake 构建目录（例如 `rm -rf cmake-build-release`）并重新运行 `cmake --preset` 和 `cmake --build` 命令。
- **特定目标构建：** 为在特定模块上工作时实现更快的迭代，有时您可以构建特定目标而不是完整的 `install` 目标，不过 `install` 可确保所有必要组件在您的 Python 环境中更新。有关更高级的目标管理，请参考 CMake 文档。
