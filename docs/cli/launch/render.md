# vllm launch render

## 概述

`vllm launch render` 启动一个无 GPU 的渲染服务器，仅用于预处理和后处理。

```bash
vllm launch render meta-llama/Llama-3.2-1B-Instruct --port 8100
```

该命令复用标准服务解析器，因此模型、前端、网络和相关的 CLI 选项遵循与 [`vllm serve`](../serve.md) 相同的约定。

## JSON CLI 参数

--8<-- "docs/cli/json_tip.inc.md"

## 参数

--8<-- "docs/generated/argparse/launch_render.inc.md"
