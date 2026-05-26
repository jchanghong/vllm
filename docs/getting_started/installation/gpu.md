---
toc_depth: 3
---

# GPU

vLLM 是一个 Python 库，支持以下 GPU 变体。选择您的 GPU 类型以查看特定于供应商的说明：

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:installation"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:installation"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:installation"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:installation"

## 要求

- 操作系统：Linux
- Python：3.10 -- 3.13

!!! note
    vLLM 不原生支持 Windows。要在 Windows 上运行 vLLM，您可以使用适用于 Linux 的 Windows 子系统 (WSL) 并配合兼容的 Linux 发行版，或者使用一些社区维护的分支，例如 [https://github.com/SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows)。

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:requirements"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:requirements"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:requirements"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:requirements"

## 使用 Python 设置

### 创建新的 Python 环境

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:set-up-using-python"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:set-up-using-python"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:set-up-using-python"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:set-up-using-python"

### 预构建的 Wheel 包 {#pre-built-wheels}

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:pre-built-wheels"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:pre-built-wheels"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:pre-built-wheels"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:pre-built-wheels"

### 从源码构建 Wheel 包

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:build-wheel-from-source"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:build-wheel-from-source"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:build-wheel-from-source"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:build-wheel-from-source"

## 使用 Docker 设置

### 预构建镜像

--8<-- [start:pre-built-images]

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:pre-built-images"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:pre-built-images"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:pre-built-images"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:pre-built-images"

--8<-- [end:pre-built-images]

### 从源码构建镜像

--8<-- [start:build-image-from-source]

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:build-image-from-source"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:build-image-from-source"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:build-image-from-source"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:build-image-from-source"

--8<-- [end:build-image-from-source]

## 支持的功能

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:supported-features"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:supported-features"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:supported-features"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:supported-features"
