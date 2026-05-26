# 使用统计信息收集

vLLM 默认会收集匿名使用数据，以帮助工程团队更好地了解哪些硬件和模型配置被广泛使用。这些数据使他们能够优先处理最常见的工作负载。收集的数据是透明的，不包含任何敏感信息。

部分数据在经过清理和聚合后，将公开发布以供社区使用。例如，您可以在此处查看 2024 年的使用报告[2024 年报告](https://2024.vllm.ai)。

## 收集哪些数据？

最新版本 vLLM 收集的数据列表示例如下：[vllm/usage/usage_lib.py](../../vllm/usage/usage_lib.py)

以下是 v0.4.0 版本的一个示例：

??? console "输出"

    ```json
    {
      "uuid": "fbe880e9-084d-4cab-a395-8984c50f1109",
      "provider": "GCP",
      "num_cpu": 24,
      "cpu_type": "Intel(R) Xeon(R) CPU @ 2.20GHz",
      "cpu_family_model_stepping": "6,85,7",
      "total_memory": 101261135872,
      "architecture": "x86_64",
      "platform": "Linux-5.10.0-28-cloud-amd64-x86_64-with-glibc2.31",
      "gpu_count": 2,
      "gpu_type": "NVIDIA L4",
      "gpu_memory_per_device": 23580639232,
      "model_architecture": "OPTForCausalLM",
      "vllm_version": "0.3.2+cu123",
      "context": "LLM_CLASS",
      "log_time": 1711663373492490000,
      "source": "production",
      "dtype": "torch.float16",
      "tensor_parallel_size": 1,
      "block_size": 16,
      "gpu_memory_utilization": 0.9,
      "quantization": null,
      "kv_cache_dtype": "auto",
      "enable_lora": false,
      "enable_prefix_caching": false,
      "enforce_eager": false,
      "disable_custom_all_reduce": true
    }
    ```

您可以通过运行以下命令预览收集的数据：

```bash
tail ~/.config/vllm/usage_stats.json
```

## 选择退出

您可以通过设置 `VLLM_NO_USAGE_STATS` 或 `DO_NOT_TRACK` 环境变量，或创建 `~/.config/vllm/do_not_track` 文件来选择退出使用统计信息收集：

```bash
# 以下任何一种方法都可以禁用使用统计信息收集
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p ~/.config/vllm && touch ~/.config/vllm/do_not_track
```
