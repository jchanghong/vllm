# 安装

vLLM 支持以下硬件平台：

- [GPU](gpu.md)
    - [NVIDIA CUDA](gpu.md)
    - [AMD ROCm](gpu.md)
    - [Intel XPU](gpu.md)
    - [Apple Silicon](gpu.md)（通过 [vLLM-Metal](https://github.com/vllm-project/vllm-metal)）
- [CPU](cpu.md)
    - [Intel/AMD x86](cpu.md#intelamd-x86)
    - [ARM AArch64](cpu.md#arm-aarch64)
    - [Apple silicon](cpu.md#apple-silicon)
    - [IBM Z (S390X)](cpu.md#ibm-z-s390x)

## 硬件插件

vLLM 支持位于主 `vllm` 仓库**外部**的第三方硬件插件。这些插件遵循[硬件可插拔式 RFC](../../design/plugin_system.md)。

所有支持的硬件列表可在 vLLM 网站上找到，请参阅[通用兼容性 - 硬件](https://vllm.ai/#compatibility)。

如果您想添加新的硬件，请通过 [Slack](https://slack.vllm.ai/) 或[电子邮件](mailto:collaboration@vllm.ai)联系我们。
