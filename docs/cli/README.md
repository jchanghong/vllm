# vLLM CLI 指南

vllm 命令行工具用于运行和管理 vLLM 模型。你可以通过查看帮助信息开始：

```bash
vllm --help
```

可用命令：

```bash
vllm {chat,complete,serve,launch,bench,collect-env,run-batch}
```

## serve

启动 vLLM OpenAI 兼容的 API 服务器。

使用模型启动：

```bash
vllm serve meta-llama/Llama-2-7b-hf
```

指定端口：

```bash
vllm serve meta-llama/Llama-2-7b-hf --port 8100
```

通过 Unix 域套接字提供服务：

```bash
vllm serve meta-llama/Llama-2-7b-hf --uds /tmp/vllm.sock
```

使用 --help 查看更多选项：

```bash
# 列出所有标志
vllm serve --help=all

# 查看参数组
vllm serve --help=ModelConfig

# 查看单个参数
vllm serve --help=max-num-seqs

# 按关键字或标志名称搜索
vllm serve --help=max
```

参见 [vllm serve](./serve.md) 获取所有可用参数的完整参考。

## launch

启动单个 vLLM 组件。

```bash
# 启动渲染服务器组件
vllm launch render meta-llama/Llama-3.2-1B-Instruct

# 查看渲染组件的所有可用标志
vllm launch render --help=all
```

参见 [vllm launch render](./launch/render.md) 获取当前启动组件参考。

## chat

通过运行的 API 服务器生成聊天补全。

```bash
# 直接连接本地 localhost API，无需参数
vllm chat

# 指定 API url
vllm chat --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# 使用单个提示快速聊天
vllm chat --quick "hi"
```

参见 [vllm chat](./chat.md) 获取所有可用参数的完整参考。

## complete

通过运行的 API 服务器，基于给定提示生成文本补全。

```bash
# 直接连接本地 localhost API，无需参数
vllm complete

# 指定 API url
vllm complete --url http://{vllm-serve-host}:{vllm-serve-port}/v1

# 使用单个提示快速补全
vllm complete --quick "The future of AI is"
```

参见 [vllm complete](./complete.md) 获取所有可用参数的完整参考。

## bench

运行延迟、在线服务吞吐量和离线推理吞吐量的基准测试。

要使用基准测试命令，请使用 `pip install vllm[bench]` 安装额外依赖。

可用命令：

```bash
vllm bench {latency, serve, throughput}
```

### latency

对单个请求批次的延迟进行基准测试。

```bash
vllm bench latency \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

参见 [vllm bench latency](./bench/latency.md) 获取所有可用参数的完整参考。

### serve

对在线服务吞吐量进行基准测试。

```bash
vllm bench serve \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host server-host \
    --port server-port \
    --random-input-len 32 \
    --random-output-len 4  \
    --num-prompts  5
```

参见 [vllm bench serve](./bench/serve.md) 获取所有可用参数的完整参考。

### throughput

对离线推理吞吐量进行基准测试。

```bash
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

参见 [vllm bench throughput](./bench/throughput.md) 获取所有可用参数的完整参考。

## collect-env

开始收集环境信息。

```bash
vllm collect-env
```

## run-batch

运行批量提示并将结果写入文件。

使用本地文件运行：

```bash
vllm run-batch \
    -i features/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

使用远程文件：

```bash
vllm run-batch \
    -i https://raw.githubusercontent.com/vllm-project/vllm/main/examples/features/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

参见 [vllm run-batch](./run-batch.md) 获取所有可用参数的完整参考。

## 更多帮助

有关任何子命令的详细选项，请使用：

```bash
vllm <subcommand> --help
```
