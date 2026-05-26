# Codex

[Codex](https://github.com/openai/codex) 是 OpenAI 的官方自主编码工具，运行在终端中。它可以理解你的代码库、编辑文件、运行命令，并帮助你更高效地编写代码。

通过将 Codex 指向 vLLM 服务器，你可以使用自己的模型作为后端，而不是 OpenAI API。这对于以下场景非常有用：

- 运行完全本地/私有的编码辅助
- 使用具有工具调用能力的开放权重模型
- 使用自定义模型进行测试和开发

## 工作原理

vLLM 实现了 OpenAI-Responses API，这正是 Codex 用来与 OpenAI 服务器通信的同一 API。通过配置 Codex 指向你的 vLLM 服务器，Codex 将其请求发送到 vLLM 而不是 OpenAI。然后 vLLM 将这些请求转换以适配你的本地模型，并以 Codex 期望的格式返回响应。

这意味着任何由 vLLM 提供且具有适当工具调用支持的模型都可以在 Codex 中作为 OpenAI 模型的即插即用替代品。

## 要求

Codex 需要一个具有强大工具调用能力的模型。模型必须支持 OpenAI-Responses 工具调用 API。有关为模型启用工具调用的详细信息，请参见[工具调用](../../features/tool_calling.md)。

## 安装

首先，按照[官方安装指南](https://github.com/openai/codex)安装 Codex。

## 启动 vLLM 服务器

启动 vLLM 时使用支持工具调用的模型——以下是使用 `Qwen/Qwen3-27B` 的示例：

```bash
vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder

```

对于其他模型，你需要使用 `--enable-auto-tool-choice` 和正确的 `--tool-call-parser` 显式启用工具调用。有关你模型正确的标志，请参考[工具调用文档](../../features/tool_calling.md)。

## 配置 Codex

Codex 通过位于 `~/.codex/config.toml` 的 TOML 文件进行配置。创建或编辑此文件，将 Codex 指向你的 vLLM 服务器：

```toml
model = "my-model"
model_provider = "vllm"

[model_providers.vllm]
name = "vLLM"
env_key = "VLLM_API_KEY"
base_url = "http://localhost:8000/v1"
wire_api = "responses"
```

配置字段：

| 字段 | 描述 |
| ----- | ----------- |
| `model` | 要使用的模型名称。必须与你传递给 vLLM 的 `--served-model-name` 匹配。 |
| `model_provider` | 设置为 `"vllm"` 以使用本地 vLLM 服务器。 |
| `[model_providers.vllm]` | vLLM 提供者的配置部分。 |
| `name` | vLLM 提供者的显示名称。 |
| `env_key` | Codex 将读取以获取 API 密钥的环境变量名称。vLLM 默认不要求身份验证，因此可以是任何值。 |
| `base_url` | vLLM 服务器的 OpenAI 兼容 API 端点 URL（默认为 `http://localhost:8000/v1`）。 |
| `wire_api` | 要使用的 API 风格。设置为 `"responses"` 以使用 OpenAI Responses API |

!!! tip
    由于 vLLM 默认不要求身份验证，你可以将 `env_key` 设置为任何虚拟环境变量：
    ```bash
    export VLLM_API_KEY=dummy
    ```

!!! warning
    使用 `responses` API 时，请确保你的 vLLM 版本支持 OpenAI Responses API。

## 测试设置

一旦 Codex 配置完成，在你的项目目录中启动它：

```bash
codex
```

尝试一个简单的提示来验证连接，例如要求它解释项目中的某个文件。如果模型正确响应，你的设置就完成了。你现在可以将 Codex 与你通过 vLLM 服务的模型一起用于编码任务。

## 故障排除

**连接被拒绝**：确保 vLLM 正在运行并且可以通过指定的 URL 访问。检查端口是否匹配，并且 `base_url` 包含 `/v1` 路径后缀。

**工具调用不起作用**：验证你的模型是否支持工具调用，并且是否已使用正确的 `--tool-call-parser` 标志启用。请参见[工具调用](../../features/tool_calling.md)。

**模型未找到**：确保 `~/.codex/config.toml` 中的 `model` 字段与你传递给 vLLM 的 `--served-model-name` 匹配。
