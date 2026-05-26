# 工具调用

vLLM 目前支持命名函数调用，以及 Chat Completion API 中 `tool_choice` 字段的 `auto`、`required`（自 `vllm>=0.8.3` 起）和 `none` 选项。

## 快速开始

启动服务器并启用工具调用。此示例使用 Meta 的 Llama 3.1 8B 模型，因此我们需要使用 vLLM 示例目录中的 `llama3_json` 工具调用对话模板：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template examples/tool_chat_template_llama3.1_json.jinja
```

接下来，发送一个触发模型使用可用工具的请求：

??? code

    ```python
    from openai import OpenAI
    import json

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

    def get_weather(location: str, unit: str):
        return f"Getting the weather for {location} in {unit}..."
    tool_functions = {"get_weather": get_weather}

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
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location", "unit"],
                },
            },
        },
    ]

    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
        tools=tools,
        tool_choice="auto",
    )

    tool_call = response.choices[0].message.tool_calls[0].function
    print(f"调用的函数: {tool_call.name}")
    print(f"参数: {tool_call.arguments}")
    print(f"结果: {tool_functions[tool_call.name](**json.loads(tool_call.arguments))}")
    ```

示例输出：

```text
调用的函数: get_weather
参数: {"location": "San Francisco, CA", "unit": "fahrenheit"}
结果: Getting the weather for San Francisco, CA in fahrenheit...
```

此示例演示了：

* 设置启用工具调用的服务器
* 定义处理工具调用的实际函数
* 使用 `tool_choice="auto"` 发起请求
* 处理结构化响应并执行相应函数

您还可以通过设置 `tool_choice={"type": "function", "function": {"name": "get_weather"}}` 来使用命名函数调用指定特定函数。请注意，这将使用结构化输出后端——因此首次使用时，在 FSM 编译完成并缓存以供后续请求之前，会有几秒（或更长时间）的延迟。

请记住，调用方有责任：

1. 在请求中定义适当的工具
2. 在聊天消息中包含相关上下文
3. 在应用程序逻辑中处理工具调用

有关更高级的用法，包括并行工具调用和不同模型特定的解析器，请参见以下章节。

## 命名函数调用

vLLM 默认在 chat completion API 中支持命名函数调用。这应该适用于 vLLM 支持的大多数结构化输出后端。保证返回有效可解析的函数调用——但不保证高质量。

vLLM 将使用结构化输出确保响应符合 `tools` 参数中 JSON schema 定义的工具参数对象。
为获得最佳结果，我们建议确保在提示词中指定预期的输出格式/schema，以使模型的预期生成与其被迫通过结构化输出后端生成的 schema 保持一致。

要使用命名函数，您需要在 chat completion 请求的 `tools` 参数中定义函数，并在 chat completion 请求的 `tool_choice` 参数中指定其中一个工具的 `name`。

## 必需函数调用

vLLM 支持 chat completion API 中的 `tool_choice='required'` 选项。与命名函数调用类似，它也使用结构化输出，因此默认启用并适用于任何支持的模型。然而，对替代解码后端的支持已列入 V1 引擎的[路线图](../usage/v1_guide.md#features)。

当设置 `tool_choice='required'` 时，模型保证根据 `tools` 参数中指定的工具列表生成一个或多个工具调用。工具调用的数量取决于用户的查询。输出格式严格遵循 `tools` 参数中定义的 schema。

## 无函数调用

vLLM 支持 chat completion API 中的 `tool_choice='none'` 选项。设置此选项后，即使请求中定义了工具，模型也不会生成任何工具调用，仅响应常规文本内容。

!!! note
    当在请求中指定工具时，vLLM 默认会在提示词中包含工具定义，无论 `tool_choice` 设置如何。要在 `tool_choice='none'` 时排除工具定义，请使用 `--exclude-tools-when-tool-choice-none` 选项。

## 约束解码行为

vLLM 是否在生成期间强制执行工具参数 schema 取决于 `tool_choice` 模式：

| `tool_choice` 值 | Schema 约束解码 | 行为 |
| --- | --- | --- |
| 命名函数 | 是（通过结构化输出后端） | 参数保证是符合函数参数 schema 的有效 JSON。 |
| `"required"` | 是（通过结构化输出后端） | 与命名函数相同。模型必须生成至少一个工具调用。 |
| `"auto"` | 否 | 模型自由生成。工具调用解析器从原始文本中提取工具调用。参数可能格式错误或不匹配 schema。 |
| `"none"` | 不适用 | 不生成任何工具调用。 |

当 schema 一致性很重要时，优先选择 `tool_choice="required"` 或命名函数调用，而不是 `"auto"`。

### 严格模式（`strict` 参数）

[OpenAI API](https://platform.openai.com/docs/guides/function-calling#strict-mode) 支持函数定义上的 `strict` 字段。当设置为 `true` 时，OpenAI 使用约束解码来保证工具调用参数匹配函数 schema，即使在 `tool_choice="auto"` 模式下也是如此。

vLLM **目前未实现** `strict` 模式。请求中接受 `strict` 字段（以避免客户端设置时出错），但它对解码行为没有影响。在 auto 模式下，参数有效性完全取决于模型的输出质量和解析器的提取逻辑。

跟踪问题：[#15526](https://github.com/vllm-project/vllm/issues/15526)，[#16313](https://github.com/vllm-project/vllm/issues/16313)。

## 自动函数调用

要启用此功能，您应设置以下标志：

* `--enable-auto-tool-choice` — **必需**。自动工具选择。它告诉 vLLM 您希望模型在认为适当时自行生成工具调用。
* `--tool-call-parser` — 选择要使用的工具解析器（如下所列）。未来将继续添加更多工具解析器。您也可以在 `--tool-parser-plugin` 中注册自己的工具解析器。
* `--tool-parser-plugin` — **可选**工具解析器插件，用于将用户定义的工具解析器注册到 vLLM，注册的工具解析器名称可以在 `--tool-call-parser` 中指定。
* `--chat-template` — **可选**用于自动工具选择。这是处理 `tool` 角色消息和包含先前生成的工具调用的 `assistant` 角色消息的对话模板的路径。Hermes、Mistral 和 Llama 模型在其 `tokenizer_config.json` 文件中具有工具兼容的对话模板，但您可以指定自定义模板。如果您的模型在 `tokenizer_config.json` 中配置了特定于工具使用的对话模板，则此参数可以设置为 `tool_use`。在这种情况下，它将根据 `transformers` 规范使用。更多信息请参阅 HuggingFace 的[说明](https://huggingface.co/docs/transformers/en/chat_templating#why-do-some-models-have-multiple-templates)；您可以在 [此处](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/blob/main/tokenizer_config.json) 找到 `tokenizer_config.json` 中的示例。

如果您喜欢的工具调用模型不受支持，欢迎贡献解析器及工具使用的对话模板！

!!! note
    使用 `tool_choice="auto"` 时，工具调用参数通过所选解析器从模型的原始文本输出中提取。解码过程中不应用 schema 级别的约束，因此参数偶尔可能格式错误或违反函数的参数 schema。详情请参见[约束解码行为](#constrained-decoding-behavior)。

### Hermes 模型（`hermes`）

所有比 Hermes 2 Pro 更新的 Nous Research Hermes 系列模型都应支持。

* `NousResearch/Hermes-2-Pro-*`
* `NousResearch/Hermes-2-Theta-*`
* `NousResearch/Hermes-3-*`

_请注意，Hermes 2 **Theta** 模型由于创建过程中的合并步骤，已知工具调用质量和能力有所下降。_

标志：`--tool-call-parser hermes`

### Mistral 模型（`mistral`）

支持的模型：

* `mistralai/Mistral-7B-Instruct-v0.3`（已确认）
* 其他 Mistral 函数调用模型也兼容。

已知问题：

1. Mistral 7B 难以正确生成并行工具调用。
2. **仅适用于 Transformers 分词后端**：Mistral 的 `tokenizer_config.json` 对话模板要求工具调用 ID 恰好为 9 位数字，这比 vLLM 生成的短得多。由于不满足此条件时会抛出异常，因此提供了以下额外的对话模板：

    * [examples/tool_chat_template_mistral.jinja](../../examples/tool_chat_template_mistral.jinja) - 这是"官方"Mistral 对话模板，但经过调整以适用于 vLLM 的工具调用 ID（提供的 `tool_call_id` 字段被截断为最后 9 位数字）
    * [examples/tool_chat_template_mistral_parallel.jinja](../../examples/tool_chat_template_mistral_parallel.jinja) - 这是一个"更好"的版本，当提供工具时添加了工具使用系统提示词，从而在处理并行工具调用时获得更好的可靠性。

推荐标志：

1. 使用官方 Mistral AI 格式：

    `--tool-call-parser mistral`

2. 使用 Transformers 格式（可用时）：

    `--tokenizer_mode hf --config_format hf --load_format hf --tool-call-parser mistral --chat-template examples/tool_chat_template_mistral_parallel.jinja`

!!! note
    Mistral AI 官方发布的模型有两种可能的格式：

    1. 使用 `auto` 或 `mistral` 参数时默认使用的官方格式：

        `--tokenizer_mode mistral --config_format mistral --load_format mistral`
        此格式使用 [mistral-common](https://github.com/mistralai/mistral-common)，即 Mistral AI 的分词器后端。

    2. Transformers 格式（可用时），使用 `hf` 参数：

        `--tokenizer_mode hf --config_format hf --load_format hf --chat-template examples/tool_chat_template_mistral_parallel.jinja`

### Llama 模型（`llama3_json`）

支持的模型：

所有 Llama 3.1、3.2 和 4 模型都应支持。

* `meta-llama/Llama-3.1-*`
* `meta-llama/Llama-3.2-*`
* `meta-llama/Llama-4-*`

所支持的工具调用是[基于 JSON 的工具调用](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling)。对于 Llama-3.2 模型引入的 [python 风格工具调用](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md#zero-shot-function-calling)，请参阅下面的 `pythonic` 工具解析器。至于 Llama 4 模型，建议使用 `llama4_pythonic` 工具解析器。

其他工具调用格式，如内置的 python 工具调用或自定义工具调用，不受支持。

已知问题：

1. Llama 3 不支持并行工具调用，但 Llama 4 模型支持。
2. 模型可能生成格式不正确的参数，例如生成序列化为字符串的数组而不是数组。

VLLM 为 Llama 3.1 和 3.2 提供了两个基于 JSON 的对话模板：

* [examples/tool_chat_template_llama3.1_json.jinja](../../examples/tool_chat_template_llama3.1_json.jinja) - 这是 Llama 3.1 模型的"官方"对话模板，但经过调整以更好地与 vLLM 配合使用。
* [examples/tool_chat_template_llama3.2_json.jinja](../../examples/tool_chat_template_llama3.2_json.jinja) - 在 Llama 3.1 对话模板的基础上扩展，增加了对图像的支持。

推荐标志：`--tool-call-parser llama3_json --chat-template {see_above}`

VLLM 还为 Llama 4 提供了 python 风格和基于 JSON 的对话模板，但建议使用 python 风格工具调用：

* [examples/tool_chat_template_llama4_pythonic.jinja](../../examples/tool_chat_template_llama4_pythonic.jinja) - 基于 Llama 4 模型的[官方对话模板](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/)。

对于 Llama 4 模型，请使用 `--tool-call-parser llama4_pythonic --chat-template examples/tool_chat_template_llama4_pythonic.jinja`。

### IBM Granite

支持的模型：

* `ibm-granite/granite-4.0-h-small` 及其他 Granite 4.0 模型

    推荐标志：`--tool-call-parser granite4`

* `ibm-granite/granite-3.0-8b-instruct`

    推荐标志：`--tool-call-parser granite --chat-template examples/tool_chat_template_granite.jinja`

    [examples/tool_chat_template_granite.jinja](../../examples/tool_chat_template_granite.jinja)：这是从 Hugging Face 原始模板修改而来的对话模板。支持并行函数调用。

* `ibm-granite/granite-3.1-8b-instruct`

    推荐标志：`--tool-call-parser granite`

    可以直接使用 Huggingface 的对话模板。支持并行函数调用。

* `ibm-granite/granite-20b-functioncalling`

    推荐标志：`--tool-call-parser granite-20b-fc --chat-template examples/tool_chat_template_granite_20b_fc.jinja`

    [examples/tool_chat_template_granite_20b_fc.jinja](../../examples/tool_chat_template_granite_20b_fc.jinja)：这是从 Hugging Face 原始模板修改而来的对话模板，原模板与 vLLM 不兼容。它融合了 Hermes 模板中的函数描述元素，并遵循[论文](https://arxiv.org/abs/2407.00121)中"响应生成"模式的相同系统提示词。支持并行函数调用。

### InternLM 模型（`internlm`）

支持的模型：

* `internlm/internlm2_5-7b-chat`（已确认）
* 其他 internlm2.5 函数调用模型也兼容。

已知问题：

* 尽管此实现也支持 InternLM2，但在使用 `internlm/internlm2-chat-7b` 模型测试时，工具调用结果不稳定。

推荐标志：`--tool-call-parser internlm --chat-template examples/tool_chat_template_internlm2_tool.jinja`

### Jamba 模型（`jamba`）

支持 AI21 的 Jamba-1.5 模型。

* `ai21labs/AI21-Jamba-1.5-Mini`
* `ai21labs/AI21-Jamba-1.5-Large`

标志：`--tool-call-parser jamba`

### xLAM 模型（`xlam`）

xLAM 工具解析器旨在支持以各种 JSON 格式生成工具调用的模型。它检测几种不同输出样式中的函数调用：

1. 直接 JSON 数组：以 `[` 开头、以 `]` 结尾的 JSON 数组格式输出字符串
2. 思考标签：使用 `<think>...</think>` 标签包含 JSON 数组
3. 代码块：代码块中的 JSON（```json ...```）
4. 工具调用标签：使用 `[TOOL_CALLS]` 或 `<tool_call>...</tool_call>` 标签

支持并行函数调用，解析器可以有效分离文本内容和工具调用。

支持的模型：

* Salesforce Llama-xLAM 模型：`Salesforce/Llama-xLAM-2-8B-fc-r`、`Salesforce/Llama-xLAM-2-70B-fc-r`
* Qwen-xLAM 模型：`Salesforce/xLAM-1B-fc-r`、`Salesforce/xLAM-3B-fc-r`、`Salesforce/Qwen-xLAM-32B-fc-r`

标志：

* 对于基于 Llama 的 xLAM 模型：`--tool-call-parser xlam --chat-template examples/tool_chat_template_xlam_llama.jinja`
* 对于基于 Qwen 的 xLAM 模型：`--tool-call-parser xlam --chat-template examples/tool_chat_template_xlam_qwen.jinja`

### Qwen 模型

对于 Qwen2.5，`tokenizer_config.json` 中的对话模板已经包含了 Hermes 风格的工具使用支持。因此，您可以使用 `hermes` 解析器为 Qwen 模型启用工具调用。更多详细信息，请参阅官方 [Qwen 文档](https://qwen.readthedocs.io/en/latest/framework/function_call.html#vllm)

* `Qwen/Qwen2.5-*`
* `Qwen/QwQ-32B`

标志：`--tool-call-parser hermes`

### MiniMax 模型（`minimax_m1`）

支持的模型：

* `MiniMaxAi/MiniMax-M1-40k`（与 [examples/tool_chat_template_minimax_m1.jinja](../../examples/tool_chat_template_minimax_m1.jinja) 一起使用）
* `MiniMaxAi/MiniMax-M1-80k`（与 [examples/tool_chat_template_minimax_m1.jinja](../../examples/tool_chat_template_minimax_m1.jinja) 一起使用）

标志：`--tool-call-parser minimax --chat-template examples/tool_chat_template_minimax_m1.jinja`

### DeepSeek-V3 模型（`deepseek_v3`）

支持的模型：

* `deepseek-ai/DeepSeek-V3-0324`（与 [examples/tool_chat_template_deepseekv3.jinja](../../examples/tool_chat_template_deepseekv3.jinja) 一起使用）
* `deepseek-ai/DeepSeek-R1-0528`（与 [examples/tool_chat_template_deepseekr1.jinja](../../examples/tool_chat_template_deepseekr1.jinja) 一起使用）

标志：`--tool-call-parser deepseek_v3 --chat-template {see_above}`

### DeepSeek-V3.1 模型（`deepseek_v31`）

支持的模型：

* `deepseek-ai/DeepSeek-V3.1`（与 [examples/tool_chat_template_deepseekv31.jinja](../../examples/tool_chat_template_deepseekv31.jinja) 一起使用）

标志：`--tool-call-parser deepseek_v31 --chat-template {see_above}`

### OpenAI OSS 模型（'openai'）

支持的模型：

* `openai/gpt-oss-20b`
* `openai/gpt-oss-120b`

标志：`--tool-call-parser openai`

### Kimi-K2 模型（`kimi_k2`）

支持的模型：

* `moonshotai/Kimi-K2-Instruct`

标志：`--tool-call-parser kimi_k2`

### Hunyuan 模型（`hunyuan_a13b`）

支持的模型：

* `tencent/Hunyuan-A13B-Instruct`（对话模板已包含在 Hugging Face 模型文件中。）

标志：

* 非推理模式：`--tool-call-parser hunyuan_a13b`
* 推理模式：`--tool-call-parser hunyuan_a13b --reasoning-parser hunyuan_a13b`

### Cohere Command A Reasoning（`cohere_command3`）

支持的模型：

* [`CohereLabs/command-a-reasoning-08-2025`](https://huggingface.co/CohereLabs/command-a-reasoning-08-2025)

标志：`--tool-call-parser cohere_command3 --reasoning-parser cohere_command3`

注意：Cohere 工具解析器需要 `cohere_melody` 包，该包默认未安装。使用此解析器前请安装 [cohere_melody](https://pypi.org/project/cohere-melody/) 包。

### LongCat-Flash-Chat 模型（`longcat`）

支持的模型：

* `meituan-longcat/LongCat-Flash-Chat`
* `meituan-longcat/LongCat-Flash-Chat-FP8`

标志：`--tool-call-parser longcat`

### GLM-4.5 模型（`glm45`）

支持的模型：

* `zai-org/GLM-4.5`
* `zai-org/GLM-4.5-Air`
* `zai-org/GLM-4.6`

标志：`--tool-call-parser glm45`

### GLM-4.7 模型（`glm47`）

支持的模型：

* `zai-org/GLM-4.7`
* `zai-org/GLM-4.7-Flash`

标志：`--tool-call-parser glm47`

### FunctionGemma 模型（`functiongemma`）

Google 的 FunctionGemma 是一个轻量级（270M 参数）模型，专门为函数调用设计。它基于 Gemma 3 构建，针对笔记本电脑和手机等设备上的边缘部署进行了优化。

支持的模型：

* `google/functiongemma-270m-it`

FunctionGemma 使用独特的输出格式，带有 `<start_function_call>` 和 `<end_function_call>` 标签：

```text
<start_function_call>call:get_weather{location:<escape>London<escape>}<end_function_call>
```

该模型设计为针对特定函数调用任务进行微调，以获得最佳效果。

标志：`--tool-call-parser functiongemma --chat-template examples/tool_chat_template_functiongemma.jinja`

!!! note
    FunctionGemma 旨在针对您的特定函数调用任务进行微调。
    基础模型提供通用函数调用能力，但最佳效果需通过针对特定任务的微调实现。请参阅 Google 的 [FunctionGemma 文档](https://ai.google.dev/gemma/docs/functiongemma)了解微调指南。

### Qwen3-Coder 模型（`qwen3_xml`）

支持的模型：

* `Qwen/Qwen3-Coder-480B-A35B-Instruct`
* `Qwen/Qwen3-Coder-30B-A3B-Instruct`

标志：`--tool-call-parser qwen3_xml`

### Olmo 3 模型（`olmo3`）

Olmo 3 模型以与 `pythonic` 解析器（见下文）预期格式非常相似的格式输出工具调用，但有一些差异。每个工具调用是一个 python 风格字符串，但并行工具调用以换行符分隔，并且调用包裹在 XML 标签内，格式为 `<function_calls>..</function_calls>`。此外，解析器还允许 JSON 布尔值和 null 字面量（`true`、`false` 和 `null`），以及 python 风格的（`True`、`False` 和 `None`）。

支持的模型：

* `allenai/Olmo-3-7B-Instruct`
* `allenai/Olmo-3-32B-Think`

标志：`--tool-call-parser olmo3`

### Gigachat 3 模型（`gigachat3`）

使用 Hugging Face 模型文件中的对话模板。

支持的模型：

* `ai-sage/GigaChat3-702B-A36B-preview`
* `ai-sage/GigaChat3-702B-A36B-preview-bf16`
* `ai-sage/GigaChat3-10B-A1.8B`
* `ai-sage/GigaChat3-10B-A1.8B-bf16`

标志：`--tool-call-parser gigachat3`

### Apertus 模型（`apertus`）

使用示例文件夹中的对话模板；它修复了几个 OpenAI 兼容性问题：`--chat-template /vllm-workspace/examples/tool_chat_template_apertus.jinja`

支持的模型：

* `swiss-ai/Apertus-8B-Instruct-2509`
* `swiss-ai/Apertus-70B-Instruct-2509`

标志：`--tool-call-parser apertus`

### 支持 Python 风格工具调用的模型（`pythonic``

越来越多的模型使用 python 列表表示工具调用，而不是使用 JSON。这样做的好处是天然支持并行工具调用，并消除了工具调用所需 JSON schema 的歧义。`pythonic` 工具解析器可以支持此类模型。

具体来说，这些模型可能通过生成以下内容来查询旧金山和西雅图的天气：

```python
[get_weather(city='San Francisco', metric='celsius'), get_weather(city='Seattle', metric='celsius')]
```

限制：

* 模型不能在同一个生成中同时生成文本和工具调用。这对于特定模型可能不难改变，但社区目前对于开始和结束工具调用时应生成哪些标记缺乏共识。（特别是，Llama 3.2 模型不生成此类标记。）
* Llama 较小的模型难以有效使用工具。

支持的示例模型：

* `meta-llama/Llama-3.2-1B-Instruct` ⚠️（与 [examples/tool_chat_template_llama3.2_pythonic.jinja](../../examples/tool_chat_template_llama3.2_pythonic.jinja) 一起使用）
* `meta-llama/Llama-3.2-3B-Instruct` ⚠️（与 [examples/tool_chat_template_llama3.2_pythonic.jinja](../../examples/tool_chat_template_llama3.2_pythonic.jinja) 一起使用）
* `Team-ACE/ToolACE-8B`（与 [examples/tool_chat_template_toolace.jinja](../../examples/tool_chat_template_toolace.jinja) 一起使用）
* `fixie-ai/ultravox-v0_4-ToolACE-8B`（与 [examples/tool_chat_template_toolace.jinja](../../examples/tool_chat_template_toolace.jinja) 一起使用）
* `meta-llama/Llama-4-Scout-17B-16E-Instruct` ⚠️（与 [examples/tool_chat_template_llama4_pythonic.jinja](../../examples/tool_chat_template_llama4_pythonic.jinja) 一起使用）
* `meta-llama/Llama-4-Maverick-17B-128E-Instruct` ⚠️（与 [examples/tool_chat_template_llama4_pythonic.jinja](../../examples/tool_chat_template_llama4_pythonic.jinja) 一起使用）

标志：`--tool-call-parser pythonic --chat-template {see_above}`

!!! warning
    Llama 较小的模型经常无法以正确的格式发出工具调用。结果可能因模型而异。

## 如何编写工具解析器插件

工具解析器插件是一个 Python 文件，包含一个或多个 ToolParser 实现。您可以编写类似于 [vllm/tool_parsers/hermes_tool_parser.py](../../vllm/tool_parsers/hermes_tool_parser.py) 中的 `Hermes2ProToolParser` 的 ToolParser。

以下是插件文件的摘要：

??? code

    ```python

    # 导入所需的包

    # 定义一个工具解析器并将其注册到 vllm
    # register_module 中的名称列表可用于
    # --tool-call-parser。您可以在此处定义任意数量的
    # 工具解析器。
    class ExampleToolParser(ToolParser):
        def __init__(self, tokenizer: TokenizerLike):
            super().__init__(tokenizer)

        # 调整请求。例如：将跳过特殊标记
        # 设置为 False 以获取工具调用输出。
        def adjust_request(self, request: ChatCompletionRequest | ResponsesRequest) -> ChatCompletionRequest | ResponsesRequest:
            return request

        # 实现流式调用的工具调用解析
        def extract_tool_calls_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
            request: ChatCompletionRequest,
        ) -> DeltaMessage | None:
            return delta

        # 实现非流式调用的工具解析
        def extract_tool_calls(
            self,
            model_output: str,
            request: ChatCompletionRequest,
        ) -> ExtractedToolCallInformation:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=text)
    # 将工具解析器注册到 ToolParserManager
    ToolParserManager.register_lazy_module(
        name="example",
        module_path="vllm.tool_parsers.example",
        class_name="ExampleToolParser",
    )

    ```

然后您可以在命令行中使用此插件，如下所示：

```bash
    --enable-auto-tool-choice \
    --tool-parser-plugin <插件文件的绝对路径>
    --tool-call-parser example \
    --chat-template <您的对话模板> \
```
