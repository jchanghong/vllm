<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

对于 Apple Silicon 上的 GPU 加速推理，请使用 [vLLM-Metal](https://github.com/vllm-project/vllm-metal)，这是一个社区维护的硬件插件，使用 MLX 作为计算后端，并通过 Apple 的 Metal 框架提供原生 GPU 加速。

vLLM-Metal 与 Hugging Face 上 [mlx-community](https://huggingface.co/mlx-community) 组织的 MLX 优化模型兼容，该组织提供针对 Apple Silicon 优化的流行模型的量化版本。

!!! tip
    有关安装和使用说明，请参阅下面的[使用 vLLM-Metal 设置](#set-up-using-vllm-metal)部分。

--8<-- [end:installation]
--8<-- [start:requirements]

- 操作系统：macOS Sonoma 或更高版本
- 硬件：Apple Silicon
- 已启用 Metal 支持

!!! note
    安装说明请参阅下面的[使用 vLLM-Metal 设置](#set-up-using-vllm-metal)部分。

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

## 使用 vLLM-Metal 设置

vLLM-Metal 作为一个独立的包分发，为 Apple Silicon 提供原生 GPU 加速。

要安装 vLLM-Metal，请按照 [vLLM-Metal 文档](https://github.com/vllm-project/vllm-metal#installation)中的安装说明操作。

安装过程将：

1. 设置适当的 Python 环境
2. 安装 MLX 及所需的依赖项
3. 安装 vLLM-Metal 包

安装完成后，您就可以开始使用带有 Metal GPU 加速的 vLLM。

!!! tip
    使用 vLLM-Metal 时，请使用 Hugging Face 上 [mlx-community](https://huggingface.co/mlx-community) 的模型以获得最佳性能。这些模型针对 MLX 进行了优化，通常包含可在 Apple Silicon 上高效运行的量化版本（4 位、8 位）。

    示例模型：`mlx-community/Qwen2.5-0.5B-Instruct-4bit`

### 使用 vLLM-Metal

安装后，vLLM-Metal 提供了一个易于使用的 CLI 来运行兼容 OpenAI 的 API 服务器：

```bash
# 激活 vLLM-Metal 环境
source ~/.venv-vllm-metal/bin/activate

# 启动 API 服务器（指定您的 mlx-community 模型，否则将使用默认模型）
vllm serve
```

服务器启动后，您有多个选项与之交互：

#### 选项 1：交互式聊天

打开一个新终端并启动交互式聊天会话：

```bash
source ~/.venv-vllm-metal/bin/activate
vllm chat
```

#### 选项 2：使用 curl 进行 API 请求

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

#### 选项 3：使用 OpenAI SDK 的 Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # 本地服务器无需认证
)

response = client.chat.completions.create(
    model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

有关 `vllm` CLI 命令的更多详细信息，请参阅[兼容 OpenAI 的服务器文档](../../serving/online_serving/openai_compatible_server.md)。

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

vLLM-Metal 通过 vLLM-Metal 包安装。请参阅上面的[使用 vLLM-Metal 设置](#set-up-using-vllm-metal)部分。

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

关于从源码构建的说明，请参考 [vLLM-Metal 文档](https://github.com/vllm-project/vllm-metal#installation)。

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

--8<-- [end:build-image-from-source]
--8<-- [start:supported-features]

vLLM-Metal 提供：

- 使用 Metal 的原生 GPU 加速
- 针对 Apple Silicon 优化的基于 MLX 的计算后端
- 兼容 OpenAI 的 API 服务器
- 支持流行的模型架构

有关特定功能支持和限制，请参考 [vLLM-Metal 文档](https://github.com/vllm-project/vllm-metal)。

--8<-- [end:supported-features]
