# 专家并行内核

[DeepSeek-V3 技术报告](http://arxiv.org/abs/2412.19437) 中描述的大规模集群级专家并行，是部署具有大量专家的稀疏 MoE 模型的有效方式。然而，这种部署需要许多超出正常 Python 包的组件，包括系统包支持和系统驱动支持。不可能将所有组件都打包到一个 Python 包中。

这里我们将需求分解为 2 个步骤：

1. 构建并安装 Python 库（[DeepEP](https://github.com/deepseek-ai/DeepEP)），包括必要的依赖（如 NVSHMEM）。此步骤不需要任何特权访问。任何用户都可以执行。
2. 配置 NVIDIA 驱动以启用 IBGDA。此步骤需要 root 访问权限，并且必须在宿主机上执行。

步骤 2 对于多节点部署是必需的。

所有脚本接受一个位置参数作为构建暂存的工作空间路径，默认为 `$(pwd)/ep_kernels_workspace`。

## 使用方法

```bash
# 对于 hopper
TORCH_CUDA_ARCH_LIST="9.0" bash install_python_libraries.sh
# 对于 blackwell
TORCH_CUDA_ARCH_LIST="10.0" bash install_python_libraries.sh
```

多节点部署的额外步骤：

```bash
sudo bash configure_system_drivers.sh # update-initramfs 可能需要几分钟
sudo reboot # 需要重启以加载新驱动
```
