# Claude Code

[Claude Code](https://code.claude.com/docs/en/quickstart) 是 Anthropic 的官方自主编码工具，运行在终端中。它可以理解你的代码库、编辑文件、运行命令，并帮助你更高效地编写代码。

通过将 Claude Code 指向 vLLM 服务器，你可以使用自己的模型作为后端，而不是 Anthropic API。这对于以下场景非常有用：

- 运行完全本地/私有的编码辅助
- 使用具有工具调用能力的开放权重模型
- 使用自定义模型进行测试和开发

## 工作原理

vLLM 实现了 Anthropic Messages API，这正是 Claude Code 用来与 Anthropic 服务器通信的同一 API。通过设置 `ANTHROPIC_BASE_URL` 指向你的 vLLM 服务器，Claude Code 将其请求发送到 vLLM 而不是 Anthropic。然后 vLLM 将这些请求转换以适配你的本地模型，并以 Claude Code 期望的格式返回响应。

这意味着任何由 vLLM 提供且具有适当工具调用支持的模型都可以在 Claude Code 中作为 Claude 模型的即插即用替代品。

## 要求

Claude Code 需要一个具有强大工具调用能力的模型。模型必须支持 OpenAI 兼容的工具调用 API。有关为模型启用工具调用的详细信息，请参见[工具调用](../../features/tool_calling.md)。

## 安装

首先，按照[官方安装指南](https://docs.anthropic.com/en/docs/claude-code/getting-started)安装 Claude Code。

## 启动 vLLM 服务器

启动 vLLM 时使用支持工具调用的模型——以下是使用 `openai/gpt-oss-120b` 的示例：

```bash
vllm serve openai/gpt-oss-120b --served-model-name my-model --enable-auto-tool-choice --tool-call-parser openai
```

对于其他模型，你需要使用 `--enable-auto-tool-choice` 和正确的 `--tool-call-parser` 显式启用工具调用。有关你模型正确的标志，请参考[工具调用文档](../../features/tool_calling.md)。

## 配置 Claude Code

使用指向你的 vLLM 服务器的环境变量启动 Claude Code：

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_OPUS_MODEL=my-model \
ANTHROPIC_DEFAULT_SONNET_MODEL=my-model \
ANTHROPIC_DEFAULT_HAIKU_MODEL=my-model \
claude
```

环境变量：

| 变量 | 描述 |
| -------------------------------- | --------------------------------------------------------------------- |
| `ANTHROPIC_BASE_URL` | 指向你的 vLLM 服务器（默认端口为 8000） |
| `ANTHROPIC_API_KEY` | 可以是任何值，因为 vLLM 默认不要求身份验证 |
| `ANTHROPIC_AUTH_TOKEN` | 必需。可以是任何值。 |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | Opus 级别请求的模型名称 |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | Sonnet 级别请求的模型名称 |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Haiku 级别请求的模型名称 |

!!! tip
    你可以将这些环境变量添加到你的 shell 配置文件（例如 `.bashrc`、`.zshrc`）、Claude Code 配置文件（`~/.claude/settings.json`）中，或者创建一个方便的包装脚本。

!!! warning
    Claude Code 最近开始在系统提示中注入每个请求的哈希值，这可能会破坏[前缀缓存](../../design/prefix_caching.md)，因为提示在每次请求时都会变化，导致性能大幅降低。在 vLLM 版本 > 0.17.1 中，这会自动处理，但对于旧版本，应将 `"CLAUDE_CODE_ATTRIBUTION_HEADER": "0"` 添加到 `~/.claude/settings.json` 的 `"env"` 部分（请参见 Unsloth 的[这篇博客文章](https://unsloth.ai/docs/basics/claude-code#fixing-90-slower-inference-in-claude-code)）。

## 测试设置

一旦 Claude Code 启动，尝试一个简单的提示来验证连接：

![Claude Code 示例聊天](../../assets/deployment/claude-code-example.png)

如果模型正确响应，你的设置就完成了。你现在可以将 Claude Code 与你通过 vLLM 服务的模型一起用于编码任务。

## 故障排除

**连接被拒绝**：确保 vLLM 正在运行并且可以通过指定的 URL 访问。检查端口是否匹配。

**工具调用不起作用**：验证你的模型是否支持工具调用，并且是否已使用正确的 `--tool-call-parser` 标志启用。请参见[工具调用](../../features/tool_calling.md)。

**模型未找到**：确保 `--served-model-name` 与环境变量中的模型名称匹配。你无法在模型名称中使用包含 `/` 的路径，例如直接从 Huggingface 使用 `openai/gpt-oss-120b`，请注意 Claude Code 的这个限制。
