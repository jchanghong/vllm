# 在 vLLM OSS CI/CD 上更新 PyTorch 版本

vLLM 的当前策略是在 CI/CD 中始终使用最新的 PyTorch 稳定版发布。标准做法是，当新的 [PyTorch 稳定版发布](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-cadence)可用时，尽早提交 PR 以更新 PyTorch 版本。由于 PyTorch 各版本之间存在差距，此过程并非微不足道。本文档以 <https://github.com/vllm-project/vllm/pull/16859> 为例，概述了实现此更新的常见步骤以及潜在问题列表及解决方法。

## 测试 PyTorch 发布候选版本（RC）

在官方发布后在 vLLM 中更新 PyTorch 并不理想，因为此时发现的任何问题只能通过等待下一个版本或在 vLLM 中实施 hacky 的变通方法来解决。更好的解决方案是测试 vLLM 与 PyTorch 发布候选版本（RC）的兼容性，以确保在每个版本发布前的兼容性。

PyTorch 发布候选版本可以从 [PyTorch 测试索引](https://download.pytorch.org/whl/test)下载。例如，`torch2.7.0+cu12.8` RC 可以使用以下命令安装：

```bash
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/test/cu128
```

当最终 RC 准备好进行测试时，它将在 [PyTorch dev-discuss 论坛](https://dev-discuss.pytorch.org/c/release-announcements)上向社区公告。在此公告之后，我们可以按照以下 3 步流程起草拉取请求开始测试 vLLM 集成：

1. 更新[requirements 文件](https://github.com/vllm-project/vllm/tree/main/requirements)
以指向 `torch`、`torchvision` 和 `torchaudio` 的新版本。

2. 使用以下选项获取最终发布候选版本的 wheel。一些常见平台是 `cpu`、`cu128` 和 `rocm6.2.4`。

    ```bash
    --extra-index-url https://download.pytorch.org/whl/test/<PLATFORM>
    ```

3. 由于 vLLM 使用 `uv`，请确保应用以下索引策略：

    - 通过环境变量：

    ```bash
    export UV_INDEX_STRATEGY=unsafe-best-match
    ```

    - 或通过 CLI 标志：

    ```bash
    --index-strategy unsafe-best-match
    ```

如果在拉取请求中发现失败，请在 vLLM 上将其提出为问题，并抄送 PyTorch 发布团队，以启动讨论解决这些问题。

## 更新 CUDA 版本

PyTorch 发布矩阵包含稳定版和实验版 [CUDA 版本](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)。由于限制，只有最新的稳定 CUDA 版本（例如 torch `2.7.1+cu126`）会上传到 PyPI。但是，vLLM 可能需要不同的 CUDA 版本，例如用于 Blackwell 支持的 12.8。这使过程变得复杂，因为我们无法直接使用 `pip install torch torchvision torchaudio` 命令。解决方案是在 vLLM 的 Dockerfile 中使用 `--extra-index-url`。

- 目前重要的索引包括：

| 平台 | `--extra-index-url` |
| -------- | ------------------- |
| CUDA 12.8 | [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128) |
| CPU | [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu) |
| ROCm 6.2 | [https://download.pytorch.org/whl/rocm6.2.4](https://download.pytorch.org/whl/rocm6.2.4) |
| ROCm 6.3 | [https://download.pytorch.org/whl/rocm6.3](https://download.pytorch.org/whl/rocm6.3) |
| XPU | [https://download.pytorch.org/whl/xpu](https://download.pytorch.org/whl/xpu) |

- 更新以下文件以匹配步骤 1 中的 CUDA 版本。这确保发布的 vLLM wheel 在 CI 中经过测试。
    - `.buildkite/release-pipeline.yaml`
    - `.buildkite/scripts/upload-wheels.sh`

## 在 BuildKite CI 上手动运行 vLLM 构建

当使用新的 PyTorch/CUDA 版本构建 vLLM 时，vLLM sccache S3 存储桶中不会有任何缓存产物，这可能导致 CI 构建作业超过 5 小时。此外，vLLM 的 fastcheck 流水线以只读模式运行，不会填充缓存，因此无法用于缓存预热。

为解决此问题，在 Buildkite 上手动触发一个构建以实现两个目标：

1. 通过设置环境变量 `RUN_ALL=1` 和 `NIGHTLY=1`，针对 PyTorch RC 构建运行完整的测试套件
2. 用编译的产物填充 vLLM sccache S3 存储桶，使后续构建更快

<p align="center" width="100%">
<img width="60%" alt="Buildkite 新建构建弹窗" src="https://github.com/user-attachments/assets/3b07f71b-bb18-4ca3-aeaf-da0fe79d315f" />
</p>

## 更新所有不同的 vLLM 平台

与其在单个拉取请求中尝试更新所有 vLLM 平台，更合理的做法是分别处理某些平台。vLLM CI/CD 中不同平台的 requirements 和 Dockerfile 是分开的，这使我们能够选择性地选择要更新的平台。例如，更新 XPU 需要 Intel 的 [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) 提供相应版本。虽然 <https://github.com/vllm-project/vllm/pull/16859> 在 CPU、CUDA 和 ROCm 上将 vLLM 更新到了 PyTorch 2.7.0，而 <https://github.com/vllm-project/vllm/pull/17444> 完成了 XPU 的更新。
