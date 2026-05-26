# 推理输出

vLLM 支持像 [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) 这样的推理模型，这些模型旨在生成包含推理步骤和最终结论的输出。

推理模型在其输出中返回一个额外的 `reasoning` 字段，其中包含得出最终结论的推理步骤。其他模型的输出中不存在此字段。

!!! warning
    `reasoning` 以前被称为 `reasoning_content`。要迁移，请直接将 `reasoning_content` 替换为 `reasoning`。

## 支持的模型

vLLM 目前支持以下推理模型：

| 模型系列 | 解析器名称 | 结构化输出支持 | 工具调用 |
| ------------ | ----------- | ---------------- | ----------- |
| [Cohere Command A Reasoning](https://huggingface.co/CohereLabs/command-a-reasoning-08-2025) | `cohere_command3` | `json`、`regex` | ✅ |
| [DeepSeek R1 系列](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) | `deepseek_r1` | `json`、`regex` | ❌ |
| [DeepSeek-V3.1](https://huggingface.co/collections/deepseek-ai/deepseek-v31-68a491bed32bd77e7fca048f) | `deepseek_v3` | `json`、`regex` | ❌ |
| [ERNIE-4.5-VL 系列](https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-PT) | `ernie45` | `json`、`regex` | ❌ |
| [ERNIE-4.5-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking) | `ernie45` | `json`、`regex` | ✅ |
| [GLM-4.5 系列](https://huggingface.co/collections/zai-org/glm-45-687c621d34bda8c9e4bf503b) | `glm45` | `json`、`regex` | ✅ |
| [Holo2 系列](https://huggingface.co/collections/Hcompany/holo2) | `holo2` | `json`、`regex` | ✅ |
| [Hunyuan A13B 系列](https://huggingface.co/collections/tencent/hunyuan-a13b-685ec38e5b46321e3ea7c4be) | `hunyuan_a13b` | `json`、`regex` | ✅ |
| [IBM Granite 3.2 语言模型](https://huggingface.co/collections/ibm-granite/granite-32-language-models-67b3bc8c13508f6d064cff9a) | `granite` | ❌ | ❌ |
| [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) | `minimax_m2_append_think` | `json`、`regex` | ✅ |
| [Qwen3 系列](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | `qwen3` | `json`、`regex` | ✅ |
| [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) | `deepseek_r1` | `json`、`regex` | ✅ |

!!! note
    IBM Granite 3.2 和 DeepSeek-V3.1 的推理默认为禁用；要启用它，您还必须在 `chat_template_kwargs` 中传递 `thinking=True`。
    Qwen3 系列的推理功能默认启用。要禁用它，您必须在 `chat_template_kwargs` 中传递 `enable_thinking=False`。
    DeepSeek-V3.1 的工具调用在非思考模式下受支持。
    Holo2 的推理默认启用。要禁用它，您还必须在 `chat_template_kwargs` 中传递 `thinking=False`。

## 快速开始

要使用推理模型，您需要在向聊天补全端点发送请求时指定 `--reasoning-parser` 标志。`--reasoning-parser` 标志指定用于从模型输出中提取推理内容的推理解析器。

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --reasoning-parser deepseek_r1
```

接下来，向模型发送一个请求，该请求应在响应中返回推理内容。

??? code

    ```python
    from openai import OpenAI

    # 修改 OpenAI 的 API key 和 API base 以使用 vLLM 的 API 服务器。
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    # 第 1 轮
    messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
    # 对于 granite，添加：`extra_body={"chat_template_kwargs": {"thinking": True}}`
    # 对于 Qwen3 系列，如果要在推理模式下禁用思考，请添加：
    # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    response = client.chat.completions.create(model=model, messages=messages)

    reasoning = response.choices[0].message.reasoning
    content = response.choices[0].message.content

    print("推理:", reasoning)
    print("内容:", content)
    ```

`reasoning` 字段包含得出最终结论的推理步骤，而 `content` 字段包含最终结论。

## 流式聊天补全

推理模型也支持流式聊天补全。在[聊天补全响应块](https://platform.openai.com/docs/api-reference/chat/streaming)的 `delta` 字段中可以使用 `reasoning` 字段。

??? console "JSON"

    ```json
    {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1694268190,
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "reasoning": "is",
                },
                "logprobs": null,
                "finish_reason": null
            }
        ]
    }
    ```

OpenAI Python 客户端库官方不支持流式输出的 `reasoning` 属性。但客户端支持响应中的额外属性。您可以使用 `hasattr` 检查响应中是否存在 `reasoning` 属性。例如：

??? code

    ```python
    from openai import OpenAI

    # 修改 OpenAI 的 API key 和 API base 以使用 vLLM 的 API 服务器。
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
    # 对于 granite，添加：`extra_body={"chat_template_kwargs": {"thinking": True}}`
    # 对于 Qwen3 系列，如果要在推理模式下禁用思考，请添加：
    # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    print("client: 开始流式聊天补全...")
    printed_reasoning = False
    printed_content = False

    for chunk in stream:
        # 安全地从 delta 中提取推理和内容，
        # 如果属性不存在或为空字符串，则默认为 None
        reasoning = (
            getattr(chunk.choices[0].delta, "reasoning", None) or None
        )
        content = getattr(chunk.choices[0].delta, "content", None) or None

        if reasoning is not None:
            if not printed_reasoning:
                printed_reasoning = True
                print("推理:", end="", flush=True)
            print(reasoning, end="", flush=True)
        elif content is not None:
            if not printed_content:
                printed_content = True
                print("\n内容:", end="", flush=True)
            # 提取并打印内容
            print(content, end="", flush=True)
    ```

请记住在访问 `reasoning` 之前检查它是否存在于响应中。您可以查看[示例](https://github.com/vllm-project/vllm/blob/main/examples/reasoning/openai_chat_completion_with_reasoning_streaming.py)。

## 工具调用

当工具调用和推理解析器都启用时，推理内容也可用。此外，工具调用仅从 `content` 字段解析函数，而不是从 `reasoning` 中解析。

??? code

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                }
            },
        }
    ]

    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
        tools=tools,
        tool_choice="auto",
    )

    print(response)
    tool_call = response.choices[0].message.tool_calls[0].function

    print(f"推理: {response.choices[0].message.reasoning}")
    print(f"调用的函数: {tool_call.name}")
    print(f"参数: {tool_call.arguments}")
    ```

更多示例，请参考 [examples/reasoning/openai_chat_completion_tool_calls_with_reasoning.py](../../examples/reasoning/openai_chat_completion_tool_calls_with_reasoning.py)。

## 服务器级默认对话模板参数

您可以使用 `--default-chat-template-kwargs` CLI 参数在服务器级别设置默认的 `chat_template_kwargs`。这对于配置所有请求的推理行为很有用，无需客户端在每个请求中指定。

### 默认禁用思考模式

对于像 Qwen3 这样默认启用思考的模型，您可以在服务器范围内禁用它：

```bash
vllm serve Qwen/Qwen3-8B \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

### 默认启用思考模式

对于像 IBM Granite 3.2 或 DeepSeek-V3.1 这样默认禁用思考的模型，您可以在服务器范围内启用它：

```bash
vllm serve ibm-granite/granite-3.2-2b-instruct \
    --reasoning-parser granite \
    --default-chat-template-kwargs '{"thinking": true}'
```

### 请求级覆盖

请求级别的 `chat_template_kwargs` 始终优先于服务器默认值。例如，如果服务器以 `enable_thinking=false` 启动，客户端仍然可以为特定请求启用它：

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}}  # 覆盖服务器默认值
)
```

## 思考预算控制

一些模型，如 [Qwen3](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html#thinking-budget)、[DeepSeek](https://www.alibabacloud.com/help/en/model-studio/deep-thinking) 和 [Nemotron3](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)，支持限制用于推理的最大 token 数量的思考预算。

Token 计数从 `reasoning_start_str` 开始。一旦推理 token 计数达到配置的 `thinking_token_budget`，vLLM 强制模型生成 `reasoning_end_str`，从而有效终止推理块。

要使用此功能：

- `--reasoning-parser` 启用推理提取。
- `--reasoning-config` 定义推理边界 token（例如，`reasoning_start_str`、`reasoning_end_str`）。如果未设置，vLLM 将尝试从推理解析器自动初始化这些 token。
- `thinking_token_budget`（一个采样参数）设置每个请求的推理 token 限制。

如果未指定 `thinking_token_budget`，则不应用显式的推理限制，仅受 `max_tokens` 等正常生成约束限制。

`--reasoning-config` 接受一个 JSON 对象，对应 [ReasoningConfig][vllm.config.ReasoningConfig] 的以下字段：

| 字段 | 类型 | 描述 |
|-----------------------|----------------|--------------------------------------------------|
| `reasoning_start_str` | `str \| null`  | 标记推理内容开始的字符串 |
| `reasoning_end_str`   | `str \| null`  | 标记推理内容结束的字符串 |

!!! note
    `reasoning_end_str` 可以包含推理结束标记之前的过渡短语。例如，将 `reasoning_end_str` 设置为 `"I have to give the solution based on the reasoning directly now.</think>"` 指示模型在预算耗尽时发出该短语，使推理终止更加自然。

### 在线服务

```bash
vllm serve Qwen/Qwen3-0.6B \
    --reasoning-parser qwen3 \
    --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.</think>"}'
```

然后使用 `thinking_token_budget` 发起请求以限制推理 token：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      { "role": "user", "content": "9.11 and 9.8, which is greater?" }
    ],
    "thinking_token_budget": 10
  }'
```

### 离线推理

```python
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    reasoning_config=ReasoningConfig(
        reasoning_start_str="<think>",
        reasoning_end_str="I have to give the solution based on the thinking directly now.</think>",
    ),
)

sampling_params = SamplingParams(thinking_token_budget=10)

messages = [
    {"role": "user", "content": "9.11 and 9.8, which is greater?"},
]

outputs = llm.chat(messages, sampling_params=sampling_params)

for output in outputs:
    print("text:", output.outputs[0].text)
```

## 限制

- 推理内容仅适用于在线服务的聊天补全端点（`/v1/chat/completions`）。

## 如何支持新的推理模型

您可以添加一个新的 `ReasoningParser`，类似于 [vllm/reasoning/deepseek_r1_reasoning_parser.py](../../vllm/reasoning/deepseek_r1_reasoning_parser.py)。

??? code

    ```python
    # 导入所需的包

    from vllm.reasoning import ReasoningParser, ReasoningParserManager
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage

    # 定义一个推理解析器并将其注册到 vllm
    # register_module 中的名称列表可用于
    # --reasoning-parser。
    class ExampleParser(ReasoningParser):
        def __init__(self, tokenizer: TokenizerLike):
            super().__init__(tokenizer)

        def extract_reasoning_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
        ) -> DeltaMessage | None:
            """
            应用于从不完整响应中提取推理的实例方法；
            用于处理推理调用和流式传输。必须是实例方法，
            因为它需要状态——当前 tokens/差异，以及关于已解析和提取内容的
            信息（参见构造函数）
            """

        def extract_reasoning(
            self,
            model_output: str,
            request: ChatCompletionRequest | ResponsesRequest,
        ) -> tuple[str | None, str | None]:
            """
            从完整的模型生成字符串中提取推理内容。

            用于非流式响应，在这种响应中，我们可以在发送给客户端之前
            获得完整的模型响应。

            参数：
            model_output: str
                用于提取推理内容的模型生成字符串。

            request: ChatCompletionRequest
                用于生成 model_output 的请求对象。

            返回：
            tuple[Optional[str], Optional[str]]
                包含推理内容和内容的元组。
            """
    # 注册推理解析器
    ReasoningParserManager.register_lazy_module(
        name="example",
        module_path="vllm.reasoning.example_reasoning_parser",
        class_name="ExampleParser",
    )
    ```

此外，要启用结构化输出，您需要创建一个新的 `Reasoner`，类似于 [vllm/reasoning/deepseek_r1_reasoning_parser.py](../../vllm/reasoning/deepseek_r1_reasoning_parser.py) 中的那个。

??? code

    ```python
    @dataclass
    class DeepSeekReasoner(Reasoner):
        """
        DeepSeek R 系列模型的 Reasoner。
        """
        start_token_id: int
        end_token_id: int

        start_token: str = "<think>"
        end_token: str = "</think>"

        @classmethod
        def from_tokenizer(cls, tokenizer: PreTrainedTokenizer) -> Reasoner:
            return cls(
                start_token_id=tokenizer.encode("<think>", add_special_tokens=False)[0],
                end_token_id=tokenizer.encode("</think>", add_special_tokens=False)[0],
            )

        def is_reasoning_end(self, input_ids: list[int]) -> bool:
            return self.end_token_id in input_ids

        def is_reasoning_end_streaming(self, input_ids: list[int], delta_ids: list[int]) -> bool:
            return self.end_token_id in delta_token_ids
        ...
    ```

像 [xgrammar](https://github.com/mlc-ai/xgrammar) 这样的结构化输出引擎将使用 `end_token_id` 来检查模型输出中是否存在推理内容，并在存在时跳过结构化输出。

最后，您可以通过使用 `--reasoning-parser` 标志为模型启用推理。

```bash
vllm serve <model_tag> --reasoning-parser example
```
