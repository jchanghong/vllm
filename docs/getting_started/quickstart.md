# 快速入门

本指南将帮助您快速上手 vLLM，以执行：

- [离线批处理推理](#offline-batched-inference)
- [在线服务](#online-serving)

## 先决条件

- 操作系统：Linux
- Python：3.10 -- 3.13

!!! note
    vLLM 也可在 macOS 上通过 [vLLM-Metal](https://github.com/vllm-project/vllm-metal) 实现 Apple Silicon GPU 加速。请参阅 [GPU 安装指南](installation/gpu.md)并选择 "Apple Silicon" 选项卡。

## 安装

=== "NVIDIA CUDA"

    如果您使用 NVIDIA GPU，可以直接使用 [pip](https://pypi.org/project/vllm/) 安装 vLLM。

    建议使用 [uv](https://docs.astral.sh/uv/)（一个非常快速的 Python 环境管理器）来创建和管理 Python 环境。请按照[文档](https://docs.astral.sh/uv/#getting-started)安装 `uv`。安装 `uv` 后，您可以使用以下命令创建新的 Python 环境并安装 vLLM：

    ```bash
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    uv pip install vllm --torch-backend=auto
    ```

    `uv` 可以通过 `--torch-backend=auto`（或 `UV_TORCH_BACKEND=auto`）检查已安装的 CUDA 驱动程序版本，[在运行时自动选择适当的 PyTorch 索引](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)。要选择特定的后端（例如 `cu126`），请设置 `--torch-backend=cu126`（或 `UV_TORCH_BACKEND=cu126`）。

    另一个便捷的方法是使用 `uv run` 配合 `--with [dependency]` 选项，它允许您无需创建任何永久环境即可运行诸如 `vllm serve` 等命令：

    ```bash
    uv run --with vllm vllm --help
    ```

    您也可以使用 [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) 来创建和管理 Python 环境。如果您想在 conda 环境中管理 `uv`，可以通过 `pip` 将其安装到 conda 环境中。

    ```bash
    conda create -n myenv python=3.12 -y
    conda activate myenv
    pip install --upgrade uv
    uv pip install vllm --torch-backend=auto
    ```

=== "AMD ROCm"

    如果您使用 AMD GPU，可以使用 `uv` 安装 vLLM。

    建议使用 [uv](https://docs.astral.sh/uv/)，因为它给予额外索引[比默认索引更高的优先级](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)。`uv` 也是一个非常快速的 Python 环境管理器，用于创建和管理 Python 环境。请按照[文档](https://docs.astral.sh/uv/#getting-started)安装 `uv`。安装 `uv` 后，您可以使用以下命令创建新的 Python 环境并安装 vLLM：

    ```bash
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
    ```

    !!! note
        目前支持 Python 3.12、ROCm 7.0 和 `glibc >= 2.35`。

    !!! note
        请注意，以前 docker 镜像是通过 AMD 的 docker 发布流水线发布并位于 `rocm/vllm-dev` 下。这正在被 vLLM 的 docker 发布流水线取代。

    !!! tip
        也提供 nightly Docker 镜像 [vllm/vllm-openai-rocm:nightly](https://hub.docker.com/r/vllm/vllm-openai-rocm/tags) 用于测试最新的开发版本。

=== "Google TPU"

    要在 Google TPU 上运行 vLLM，您需要安装 `vllm-tpu` 包。
    
    ```bash
    uv pip install vllm-tpu
    ```

    !!! note
        有关更多详细说明，包括 Docker、从源码安装和故障排除，请参阅 [vLLM on TPU 文档](https://docs.vllm.ai/projects/tpu/en/latest/)。

=== "Apple Silicon (Mac)"

    如果您使用 Apple Silicon Mac，可以通过 vLLM-Metal 使用 Apple 的 Metal 框架进行 GPU 加速推理。

    请按照 [vLLM-Metal 文档](https://github.com/vllm-project/vllm-metal#installation)中的安装说明操作。

    !!! note
        vLLM-Metal 使用 MLX 而非 PyTorch 作为计算后端，需要来自 Hugging Face 上 [mlx-community](https://huggingface.co/mlx-community) 的 MLX 优化模型。

    !!! tip
        有关更多详细说明，请参阅 [GPU 安装指南](installation/gpu.md)并选择 "Apple Silicon" 选项卡。

!!! note
    有关更多详细信息和非 CUDA 平台，请参阅[安装指南](installation/README.md)了解如何安装 vLLM 的具体说明。

## 离线批处理推理

安装 vLLM 后，您可以开始为输入提示列表生成文本（即离线批处理推理）。请参阅示例脚本：[examples/basic/offline_inference/basic.py](../../examples/basic/offline_inference/basic.py)

此示例的第一行导入了类 [LLM][vllm.LLM] 和 [SamplingParams][vllm.SamplingParams]：

- [LLM][vllm.LLM] 是使用 vLLM 引擎运行离线推理的主类。
- [SamplingParams][vllm.SamplingParams] 指定采样过程的参数。

```python
from vllm import LLM, SamplingParams
```

下一部分定义了一系列输入提示和用于文本生成的采样参数。[采样温度](https://arxiv.org/html/2402.05201v1)设置为 `0.8`，[核心采样概率](https://en.wikipedia.org/wiki/Top-p_sampling)设置为 `0.95`。您可以在此处找到有关采样参数的更多信息：[推理参数](../api/README.md#inference-parameters)。

!!! important
    默认情况下，vLLM 将应用 Hugging Face 模型仓库中的 `generation_config.json`（如果存在），使用模型创建者推荐的采样参数。在大多数情况下，如果未指定 [SamplingParams][vllm.SamplingParams]，这将为您提供最佳结果。

    但是，如果您希望使用 vLLM 的默认采样参数，请在创建 [LLM][vllm.LLM] 实例时设置 `generation_config="vllm"`。

```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```

[LLM][vllm.LLM] 类初始化 vLLM 引擎和 [OPT-125M 模型](https://arxiv.org/abs/2205.01068)以进行离线推理。支持的模型列表可在[此处](../models/supported_models.md)找到。

```python
llm = LLM(model="facebook/opt-125m")
```

!!! note
    默认情况下，vLLM 从 [Hugging Face](https://huggingface.co/) 下载模型。如果您想使用 [ModelScope](https://www.modelscope.cn) 的模型，请在初始化引擎前设置环境变量 `VLLM_USE_MODELSCOPE`。

    ```shell
    export VLLM_USE_MODELSCOPE=True
    ```

现在，最有趣的部分！使用 `llm.generate` 生成输出。它将输入提示添加到 vLLM 引擎的等待队列中，并执行 vLLM 引擎以高吞吐量生成输出。输出作为 `RequestOutput` 对象列表返回，其中包含所有输出 token。

```python
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

!!! note
    `llm.generate` 方法不会自动将模型的聊天模板应用于输入提示。因此，如果您使用的是 Instruct 模型或 Chat 模型，应手动应用相应的聊天模板以确保预期行为。或者，您可以使用 `llm.chat` 方法并传递消息列表，其格式与传递给 OpenAI 的 `client.chat.completions` 的消息格式相同：

    ??? code
    
        ```python
        # 使用 tokenizer 应用聊天模板
        from transformers import AutoTokenizer
    
        tokenizer = AutoTokenizer.from_pretrained("/path/to/chat_model")
        messages_list = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        texts = tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 生成输出
        outputs = llm.generate(texts, sampling_params)
        
        # 打印输出。
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
        # 使用聊天接口。
        outputs = llm.chat(messages_list, sampling_params)
        for idx, output in enumerate(outputs):
            prompt = prompts[idx]
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        ```

## 在线服务

vLLM 可以部署为实现 OpenAI API 协议的服务器。这使得 vLLM 可以作为使用 OpenAI API 的应用程序的直接替代品。
默认情况下，它在 `http://localhost:8000` 启动服务器。您可以使用 `--host` 和 `--port` 参数指定地址。该服务器一次托管一个模型，并实现诸如 [list models](https://platform.openai.com/docs/api-reference/models/list)、[create chat completion](https://platform.openai.com/docs/api-reference/chat/completions/create) 和 [create completion](https://platform.openai.com/docs/api-reference/completions/create) 等端点。

运行以下命令以使用 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 模型启动 vLLM 服务器：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

!!! note
    默认情况下，服务器使用 tokenizer 中预定义的聊天模板。
    您可以了解如何覆盖它：[此处](../serving/online_serving/README.md#chat-template)。
!!! important
    默认情况下，服务器应用 Hugging Face 模型仓库中的 `generation_config.json`（如果存在）。这意味着某些采样参数的默认值可以被模型创建者推荐的值覆盖。

    要禁用此行为，请在启动服务器时传递 `--generation-config vllm`。

此服务器可以使用与 OpenAI API 相同的格式进行查询。例如，列出模型：

```bash
curl http://localhost:8000/v1/models
```

您可以传入 `--api-key` 参数或环境变量 `VLLM_API_KEY`，使服务器检查请求头中的 API 密钥。
您可以在 `--api-key` 后传递多个密钥，服务器将接受传递的任何密钥，这对于密钥轮换非常有用。

### 使用 vLLM 的 OpenAI Completions API

服务器启动后，您可以使用输入提示查询模型：

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

由于此服务器与 OpenAI API 兼容，您可以将其用作使用 OpenAI API 的任何应用程序的直接替代品。例如，通过 `openai` Python 包查询服务器的另一种方式：

??? code

    ```python
    from openai import OpenAI

    # 修改 OpenAI 的 API 密钥和 API 基础 URL 以使用 vLLM 的 API 服务器。
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    completion = client.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        prompt="San Francisco is a",
    )
    print("Completion result:", completion)
    ```

更详细的客户端示例可在此处找到：[examples/basic/offline_inference/basic.py](../../examples/basic/offline_inference/basic.py)

### 使用 vLLM 的 OpenAI Chat Completions API

vLLM 也被设计为支持 OpenAI Chat Completions API。聊天界面是一种更动态、更交互的与模型通信的方式，允许来回交流，这些交流可以存储在聊天历史中。这对于需要上下文或更详细解释的任务非常有用。

您可以使用 [create chat completion](https://platform.openai.com/docs/api-reference/chat/completions/create) 端点与模型交互：

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```

或者，您可以使用 `openai` Python 包：

??? code

    ```python
    from openai import OpenAI
    # 设置 OpenAI 的 API 密钥和 API 基础 URL 以使用 vLLM 的 API 服务器。
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ],
    )
    print("Chat response:", chat_response)
    ```

## 关于注意力后端

目前，vLLM 支持多个后端，用于跨不同平台和加速器架构的高效注意力计算。它会自动选择与您的系统和模型规范兼容的性能最佳的后端。

如果需要，您也可以使用 `--attention-backend` CLI 参数手动设置您选择的后端：

```bash
# 在线服务
vllm serve Qwen/Qwen2.5-1.5B-Instruct --attention-backend FLASH_ATTN

# 离线推理
python script.py --attention-backend FLASHINFER
```

一些可用的后端选项包括：

- 在 NVIDIA CUDA 上：`FLASH_ATTN` 或 `FLASHINFER`。
- 在 AMD ROCm 上：`TRITON_ATTN`、`ROCM_ATTN`、`ROCM_AITER_FA`、`ROCM_AITER_UNIFIED_ATTN`、`TRITON_MLA`、`ROCM_AITER_MLA` 或 `ROCM_AITER_TRITON_MLA`。

!!! warning
    没有包含 Flash Infer 的预构建 vllm wheel 包，因此您必须先在环境中安装它。请参考 [Flash Infer 官方文档](https://docs.flashinfer.ai/)或查看 [docker/Dockerfile](../../docker/Dockerfile) 了解如何安装的说明。
