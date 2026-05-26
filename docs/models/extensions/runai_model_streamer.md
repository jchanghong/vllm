# 使用 Run:ai Model Streamer 加载模型

Run:ai Model Streamer 是一个用于并发读取张量并将其流式传输到 GPU 内存的库。
更多信息请参见 [Run:ai Model Streamer 文档](https://github.com/run-ai/runai-model-streamer/blob/master/docs/README.md)。

vLLM 支持使用 Run:ai Model Streamer 以 Safetensors 格式加载权重。
您首先需要安装 vLLM 的 RunAI 可选依赖：

```bash
pip3 install vllm[runai]
```

要将其作为 OpenAI 兼容的服务器运行，请添加 `--load-format runai_streamer` 标志：

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer
```

要从 AWS S3 对象存储运行模型，请执行：

```bash
vllm serve s3://core-llm/Llama-3-8b \
    --load-format runai_streamer
```

要从 Google Cloud Storage 运行模型，请执行：

```bash
vllm serve gs://core-llm/Llama-3-8b \
    --load-format runai_streamer
```

要从 Azure Blob Storage 运行模型，请执行：

```bash
AZURE_STORAGE_ACCOUNT_NAME=<account> \
vllm serve az://<container>/<model-path> \
    --load-format runai_streamer
```

认证使用 `DefaultAzureCredential`，支持 `az login`、托管身份、环境变量（`AZURE_CLIENT_ID`、`AZURE_TENANT_ID`、`AZURE_CLIENT_SECRET`）以及其他方法。

要从兼容 S3 的对象存储运行模型，请执行：

```bash
RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 \
AWS_EC2_METADATA_DISABLED=true \
AWS_ENDPOINT_URL=https://storage.googleapis.com \
vllm serve s3://core-llm/Llama-3-8b \
    --load-format runai_streamer
```

## 可调参数

您可以使用 `--model-loader-extra-config` 来调整参数：

可以调整 `distributed` 参数，用于控制是否使用分布式流式加载。目前仅在 CUDA 和 ROCM 设备上可用。这可以显著改善从对象存储或高吞吐量网络文件共享的加载时间。
更多关于分布式流式加载的信息，请阅读[此处](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/usage.md#distributed-streaming)。

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer \
    --model-loader-extra-config '{"distributed":true}'
```

您可以调整 `concurrency` 参数，用于控制并发级别以及从文件读取张量到 CPU 缓冲区的 OS 线程数。
对于从 S3 读取，它将表示主机向 S3 服务器打开的客户端实例数量。

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer \
    --model-loader-extra-config '{"concurrency":16}'
```

您可以控制从文件读取张量到 CPU 内存缓冲区的大小，并限制该大小。
更多关于 CPU 缓冲区内存限制的信息，请阅读[此处](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/env-vars.md#runai_streamer_memory_limit)。

```bash
vllm serve /home/meta-llama/Llama-3.2-3B-Instruct \
    --load-format runai_streamer \
    --model-loader-extra-config '{"memory_limit":5368709120}'
```

!!! note
    有关可调参数及其他可通过环境变量配置的参数，请阅读[环境变量文档](https://github.com/run-ai/runai-model-streamer/blob/master/docs/src/env-vars.md)。

## 分片模型加载

vLLM 还支持使用 Run:ai Model Streamer 加载分片模型。这对于跨多个文件拆分的大型模型尤为有用。要使用此功能，请使用 `--load-format runai_streamer_sharded` 标志：

```bash
vllm serve /path/to/sharded/model --load-format runai_streamer_sharded
```

分片加载器期望模型文件遵循与常规分片状态加载器相同的命名模式：`model-rank-{rank}-part-{part}.safetensors`。您可以使用 `--model-loader-extra-config` 中的 `pattern` 参数自定义此模式：

```bash
vllm serve /path/to/sharded/model \
    --load-format runai_streamer_sharded \
    --model-loader-extra-config '{"pattern":"custom-model-rank-{rank}-part-{part}.safetensors"}'
```

要创建分片模型文件，您可以使用 [examples/features/sharded_state/save_sharded_state_offline.py](../../../examples/features/sharded_state/save_sharded_state_offline.py) 中提供的脚本。该脚本演示了如何以与 Run:ai Model Streamer 分片加载器兼容的分片格式保存模型。

分片加载器支持与常规 Run:ai Model Streamer 相同的所有可调参数，包括 `concurrency` 和 `memory_limit`。这些参数的配置方式相同：

```bash
vllm serve /path/to/sharded/model \
    --load-format runai_streamer_sharded \
    --model-loader-extra-config '{"concurrency":16, "memory_limit":5368709120}'
```

!!! note
    分片加载器对于张量或流水线并行模型尤其高效，因为每个工作节点只需读取自己的分片，而无需读取整个检查点。
