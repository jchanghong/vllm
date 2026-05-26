# 日志配置

vLLM 利用 Python 的 `logging.config.dictConfig` 功能来实现
对 vLLM 使用的各种日志记录器进行健壮且灵活的配置。

vLLM 提供了两个环境变量，可用于适应从简单且不灵活到
更复杂且更灵活的各种日志配置。

- 无 vLLM 日志记录（简单且不灵活）
    - 设置 `VLLM_CONFIGURE_LOGGING=0`（保持 `VLLM_LOGGING_CONFIG_PATH` 未设置）
- vLLM 的默认日志配置（简单且不灵活）
    - 保持 `VLLM_CONFIGURE_LOGGING` 未设置或设置 `VLLM_CONFIGURE_LOGGING=1`
- 细粒度的自定义日志配置（更复杂，更灵活）
    - 保持 `VLLM_CONFIGURE_LOGGING` 未设置或设置 `VLLM_CONFIGURE_LOGGING=1` 并
    设置 `VLLM_LOGGING_CONFIG_PATH=<path-to-logging-config.json>`

## 日志配置环境变量

### `VLLM_CONFIGURE_LOGGING`

`VLLM_CONFIGURE_LOGGING` 控制 vLLM 是否对
vLLM 使用的日志记录器进行任何配置操作。此功能默认启用，
但可以通过在运行 vLLM 时设置 `VLLM_CONFIGURE_LOGGING=0` 来禁用。

如果 `VLLM_CONFIGURE_LOGGING` 已启用且没有为
`VLLM_LOGGING_CONFIG_PATH` 提供值，vLLM 将使用内置的默认配置来
配置根 vLLM 日志记录器。默认情况下，不会配置其他 vLLM 日志记录器，
因此所有 vLLM 日志记录器都委托给根 vLLM 日志记录器来做出
所有日志记录决策。

如果 `VLLM_CONFIGURE_LOGGING` 被禁用且为
`VLLM_LOGGING_CONFIG_PATH` 提供了值，则启动 vLLM 时会报错。

### `VLLM_LOGGING_CONFIG_PATH`

`VLLM_LOGGING_CONFIG_PATH` 允许用户指定一个 JSON 文件路径，该文件包含
替代的自定义日志配置，将用于替代 vLLM 的
内置默认日志配置。日志配置应
按照 Python 的[日志配置字典架构](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details)
以 JSON 格式提供。

如果指定了 `VLLM_LOGGING_CONFIG_PATH`，但 `VLLM_CONFIGURE_LOGGING` 被
禁用，则启动 vLLM 时会报错。

## 示例

### 示例 1：自定义 vLLM 根日志记录器

在本示例中，我们将自定义 vLLM 根日志记录器，使用
[`python-json-logger`](https://github.com/nhairs/python-json-logger)
（它是容器镜像的一部分）以日志级别 `INFO`
将 JSON 格式的日志输出到控制台的 STDOUT。

首先，创建一个适当的 JSON 日志配置文件：

??? note "/path/to/logging_config.json"

    ```json
    {
      "formatters": {
        "json": {
          "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
      },
      "handlers": {
        "console": {
          "class" : "logging.StreamHandler",
          "formatter": "json",
          "level": "INFO",
          "stream": "ext://sys.stdout"
        }
      },
      "loggers": {
        "vllm": {
          "handlers": ["console"],
          "level": "INFO",
          "propagate": false
        }
      },
      "version": 1
    }
    ```

最后，将 `VLLM_LOGGING_CONFIG_PATH` 环境变量设置为
自定义日志配置 JSON 文件的路径后运行 vLLM：

```bash
VLLM_LOGGING_CONFIG_PATH=/path/to/logging_config.json \
    vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

### 示例 2：静默特定 vLLM 日志记录器

要静默特定的 vLLM 日志记录器，需要为目标日志记录器提供自定义日志
配置，将其配置为不
将其日志消息传播到根 vLLM 日志记录器。

为任何日志记录器提供自定义配置时，还需要
为根 vLLM 日志记录器提供配置，因为任何自定义日志记录器
配置都会覆盖 vLLM 使用的内置默认日志配置。

首先，创建一个适当的 JSON 日志配置文件，其中包含
根 vLLM 日志记录器和您希望静默的日志记录器的配置：

??? note "/path/to/logging_config.json"

    ```json
    {
      "formatters": {
        "vllm": {
          "class": "vllm.logging_utils.NewLineFormatter",
          "datefmt": "%m-%d %H:%M:%S",
          "format": "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
        }
      },
      "handlers": {
        "vllm": {
          "class" : "logging.StreamHandler",
          "formatter": "vllm",
          "level": "INFO",
          "stream": "ext://sys.stdout"
        }
      },
      "loggers": {
        "vllm": {
          "handlers": ["vllm"],
          "level": "DEBUG",
          "propagate": false
        },
        "vllm.example_noisy_logger": {
          "propagate": false
        }
      },
      "version": 1
    }
    ```

最后，将 `VLLM_LOGGING_CONFIG_PATH` 环境变量设置为
自定义日志配置 JSON 文件的路径后运行 vLLM：

```bash
VLLM_LOGGING_CONFIG_PATH=/path/to/logging_config.json \
    vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

### 示例 3：禁用 vLLM 默认日志配置

要禁用 vLLM 的默认日志配置并静默所有 vLLM 日志记录器，
只需在运行 vLLM 时设置 `VLLM_CONFIGURE_LOGGING=0`。这将阻止 vLLM
配置根 vLLM 日志记录器，从而静默所有其他 vLLM
日志记录器。

```bash
VLLM_CONFIGURE_LOGGING=0 \
    vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

### 示例 4：禁用健康检查端点的访问日志

在生产环境中，像 `/health`、`/metrics`
和 `/ping` 这样的健康检查端点会被负载均衡器和监控系统频繁调用，
产生大量重复的访问日志。为了减少日志噪音，同时
保留其他端点的日志，请使用 `--disable-access-log-for-endpoints`
选项。

**禁用健康检查和指标端点的访问日志：**

```bash
vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048 \
    --disable-access-log-for-endpoints /health,/metrics,/ping
```

**常见可考虑过滤的端点：**

| 端点 | 描述 | 典型调用方 |
| ---------- | ---------------------- | ---------------------------------------------------- |
| `/health`  | 健康检查 | Kubernetes 存活/就绪探针、负载均衡器 |
| `/metrics` | Prometheus 指标 | Prometheus 采集器（每 15-60 秒） |
| `/ping`    | SageMaker 健康检查 | SageMaker 基础设施 |
| `/load`    | 服务器负载指标 | 自定义监控 |

**注意：**

- 此选项仅影响 uvicorn 访问日志，不影响 vLLM 应用日志
- 指定多个端点时，用逗号分隔（无空格）
- 过滤器使用精确路径匹配，忽略查询参数（例如，`/health?verbose=true` 匹配 `/health`）
- 如果需要完全禁用所有访问日志，请使用 `--disable-uvicorn-access-log`

## 其他资源

- [`logging.config` 字典架构详情](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details)
