# 使用 InstantTensor 加载模型权重

InstantTensor 通过分布式加载、流水线预取和直接 I/O，加速在 CUDA 设备上加载 Safetensors 权重。InstantTensor 在可用时还支持 GDS（GPUDirect Storage）。
更多详情，请参见 [InstantTensor GitHub 仓库](https://github.com/scitix/InstantTensor)。

## 安装

```bash
pip install instanttensor
```

## 在 vLLM 中使用 InstantTensor

添加 `--load-format instanttensor` 作为命令行参数。

例如：

```bash
vllm serve Qwen/Qwen2.5-0.5B --load-format instanttensor
```

## 基准测试

| 模型 | GPU | 后端 | 加载时间（秒） | 吞吐量（GB/s） | 加速比 |
| --- | ---: | --- | ---: | ---: | --- |
| Qwen3-30B-A3B | 1*H200 | Safetensors | 57.4 | 1.1 | 1x |
| Qwen3-30B-A3B | 1*H200 | InstantTensor | 1.77 | 35 | <span style="color: green">**32.4x**</span> |
| DeepSeek-R1 | 8*H200 | Safetensors | 160 | 4.3 | 1x |
| DeepSeek-R1 | 8*H200 | InstantTensor | 15.3 | 45 | <span style="color: green">**10.5x**</span> |

完整的基准测试结果，请参见 <https://github.com/scitix/InstantTensor/blob/main/docs/benchmark.md>。
