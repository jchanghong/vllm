# 服务端参数

`vllm serve` 命令用于启动兼容 OpenAI 的服务器。

## CLI 参数

`vllm serve` 命令用于启动兼容 OpenAI 的服务器。
要查看可用选项，请查阅 [CLI 参考](../cli/README.md)！

## 配置文件

您可以通过 [YAML](https://yaml.org/) 配置文件加载 CLI 参数。
参数名称必须是上述[服务端参数](serve_args.md)中概述的长格式。

例如：

```yaml
# config.yaml

model: meta-llama/Llama-3.1-8B-Instruct
host: "127.0.0.1"
port: 6379
uvicorn-log-level: "info"
```

要使用上述配置文件：

```bash
vllm serve --config config.yaml
```

!!! note
    如果同时通过命令行和配置文件提供了某个参数，则命令行中的值优先。
    优先级顺序为 `命令行 > 配置文件值 > 默认值`。
    例如 `vllm serve SOME_MODEL --config config.yaml`，SOME_MODEL 的优先级高于配置文件中的 `model`。
