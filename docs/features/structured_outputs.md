# 结构化输出

vLLM 支持使用 [xgrammar](https://github.com/mlc-ai/xgrammar) 或
[guidance](https://github.com/guidance-ai/llguidance) 作为后端生成结构化输出。
本文档展示了可用于生成结构化输出的一些不同选项的示例。

!!! warning
    如果您仍在使用以下在 v0.12.0 中已移除的已弃用 API 字段，请更新您的代码以使用本文档其余部分演示的 `structured_outputs`：

    - `guided_json` -> `{"structured_outputs": {"json": ...}}` 或 `StructuredOutputsParams(json=...)`
    - `guided_regex` -> `{"structured_outputs": {"regex": ...}}` 或 `StructuredOutputsParams(regex=...)`
    - `guided_choice` -> `{"structured_outputs": {"choice": ...}}` 或 `StructuredOutputsParams(choice=...)`
    - `guided_grammar` -> `{"structured_outputs": {"grammar": ...}}` 或 `StructuredOutputsParams(grammar=...)`
    - `guided_whitespace_pattern` -> `{"structured_outputs": {"whitespace_pattern": ...}}` 或 `StructuredOutputsParams(whitespace_pattern=...)`
    - `structural_tag` -> `{"structured_outputs": {"structural_tag": ...}}` 或 `StructuredOutputsParams(structural_tag=...)`
    - `guided_decoding_backend` -> 从您的请求中移除此字段

## 在线服务（OpenAI API）

您可以使用 OpenAI 的 [Completions](https://platform.openai.com/docs/api-reference/completions) 和 [Chat](https://platform.openai.com/docs/api-reference/chat) API 生成结构化输出。

支持以下参数，必须作为额外参数添加：

- `choice`：输出将恰好是选项之一。
- `regex`：输出将遵循正则表达式模式。
- `json`：输出将遵循 JSON schema。
- `grammar`：输出将遵循上下文无关文法。
- `structural_tag`：在生成文本中的一组指定标签内遵循 JSON schema。

您可以在 [OpenAI 兼容服务器](../serving/online_serving/openai_compatible_server.md) 页面上查看支持的完整参数列表。

结构化输出默认在 OpenAI 兼容服务器中支持。您可以通过设置 `--structured-outputs-config.backend` 标志来指定要使用的后端。默认后端是 `auto`，它会根据请求的详细信息尝试选择适当的后端。您也可以选择特定的后端以及一些选项。完整的选项集可在 `vllm serve --help` 文本中找到。

现在让我们从 `choice` 开始，看看每种情况的示例，因为它是最简单的：

??? code

    ```python
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="-",
    )
    model = client.models.list().data[0].id

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
        ],
        extra_body={"structured_outputs": {"choice": ["positive", "negative"]}},
    )
    print(completion.choices[0].message.content)
    ```

下一个示例演示如何使用 `regex`。支持的正则表达式语法取决于结构化输出后端。例如，`xgrammar`、`guidance` 和 `outlines` 使用 Rust 风格的正则表达式，而 `lm-format-enforcer` 使用 Python 的 `re` 模块。思路是根据一个简单的正则表达式模板生成电子邮件地址：

??? code

    ```python
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate an example email address for Alan Turing, who works in Enigma. End in .com and new line. Example result: alan.turing@enigma.com\n",
            }
        ],
        extra_body={"structured_outputs": {"regex": r"\w+@\w+\.com\n"}, "stop": ["\n"]},
    )
    print(completion.choices[0].message.content)
    ```

结构化文本生成中最相关的功能之一是生成具有预定义字段和格式的有效 JSON 的选项。
为此，我们可以通过两种方式使用 `json` 参数：

- 直接使用 [JSON Schema](https://json-schema.org/)
- 定义 [Pydantic 模型](https://docs.pydantic.dev/latest/)，然后从其提取 JSON Schema（这通常是更简单的方式）。

下一个示例演示如何将 `response_format` 参数与 Pydantic 模型一起使用：

??? code

    ```python
    from pydantic import BaseModel
    from enum import Enum

    class CarType(str, Enum):
        sedan = "sedan"
        suv = "SUV"
        truck = "Truck"
        coupe = "Coupe"

    class CarDescription(BaseModel):
        brand: str
        model: str
        car_type: CarType

    json_schema = CarDescription.model_json_schema()

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "car-description",
                "schema": CarDescription.model_json_schema()
            },
        },
    )
    print(completion.choices[0].message.content)
    ```

!!! tip
    虽然并非严格必要，但通常在提示词中指明 JSON schema 以及如何填充字段效果更好。在大多数情况下，这可以显著改善结果。

最后，我们还有 `grammar` 选项，这可能最难使用，但非常强大。它允许我们定义像 SQL 查询这样的完整语言。它通过使用上下文无关的 EBNF 文法来工作。作为示例，我们可以使用它来定义简化 SQL 查询的特定格式：

??? code

    ```python
    simplified_sql_grammar = """
        root ::= select_statement

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate an SQL query to show the 'username' and 'email' from the 'users' table.",
            }
        ],
        extra_body={"structured_outputs": {"grammar": simplified_sql_grammar}},
    )
    print(completion.choices[0].message.content)
    ```

另请参见：[完整示例](../../examples/features/structured_outputs/README.md)

## 推理输出

您还可以将结构化输出与 <project:#reasoning-outputs> 一起用于推理模型。

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --reasoning-parser deepseek_r1
```

请注意，您可以将推理与任何提供的结构化输出功能一起使用。以下是一个与 JSON schema 一起使用的示例：

??? code

    ```python
    from pydantic import BaseModel


    class People(BaseModel):
        name: str
        age: int


    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate a JSON with the name and age of one random person.",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "people",
                "schema": People.model_json_schema()
            }
        },
    )
    print("推理: ", completion.choices[0].message.reasoning)
    print("内容: ", completion.choices[0].message.content)
    ```

另请参见：[完整示例](../../examples/features/structured_outputs/README.md)

!!! note
    使用启用推理的 Qwen3 Coder 模型时，如果推理内容未被单独解析到 `reasoning` 字段中（v0.11.2+），结构化输出可能会被禁用。
    要同时使用这两个功能，您必须显式启用推理模式下的结构化输出。
    为此，请在启动 vLLM 服务器时添加以下标志：`--structured-outputs-config.enable_in_reasoning=True`。
    另请参见：[推理输出](reasoning_outputs.md)文档。

## 实验性自动解析（OpenAI API）

本节介绍 OpenAI 对 `client.chat.completions.create()` 方法的 beta 包装器，它提供了与 Python 特定类型的更丰富集成。

在撰写本文时（`openai==1.54.4`），这是 OpenAI 客户端库中的一个"beta"功能。代码参考可在[此处](https://github.com/openai/openai-python/blob/52357cff50bee57ef442e94d78a0de38b4173fc2/src/openai/resources/beta/chat/completions.py#L100-L104)找到。

对于以下示例，vLLM 使用 `vllm serve meta-llama/Llama-3.1-8B-Instruct` 设置。

以下是一个使用 Pydantic 模型获取结构化输出的简单示例：

??? code

    ```python
    from pydantic import BaseModel
    from openai import OpenAI

    class Info(BaseModel):
        name: str
        age: int

    client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="dummy")
    model = client.models.list().data[0].id
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My name is Cameron, I'm 28. What's my name and age?"},
        ],
        response_format=Info,
    )

    message = completion.choices[0].message
    print(message)
    assert message.parsed
    print("Name:", message.parsed.name)
    print("Age:", message.parsed.age)
    ```

```console
ParsedChatCompletionMessage[Testing](content='{"name": "Cameron", "age": 28}', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=Testing(name='Cameron', age=28))
Name: Cameron
Age: 28
```

以下是一个使用嵌套 Pydantic 模型处理逐步数学解的更复杂示例：

??? code

    ```python
    from typing import List
    from pydantic import BaseModel
    from openai import OpenAI

    class Step(BaseModel):
        explanation: str
        output: str

    class MathResponse(BaseModel):
        steps: list[Step]
        final_answer: str

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful expert math tutor."},
            {"role": "user", "content": "Solve 8x + 31 = 2."},
        ],
        response_format=MathResponse,
    )

    message = completion.choices[0].message
    print(message)
    assert message.parsed
    for i, step in enumerate(message.parsed.steps):
        print(f"步骤 #{i}:", step)
    print("答案:", message.parsed.final_answer)
    ```

输出：

```console
ParsedChatCompletionMessage[MathResponse](content='{ "steps": [{ "explanation": "First, let\'s isolate the term with the variable \'x\'. To do this, we\'ll subtract 31 from both sides of the equation.", "output": "8x + 31 - 31 = 2 - 31"}, { "explanation": "By subtracting 31 from both sides, we simplify the equation to 8x = -29.", "output": "8x = -29"}, { "explanation": "Next, let\'s isolate \'x\' by dividing both sides of the equation by 8.", "output": "8x / 8 = -29 / 8"}], "final_answer": "x = -29/8" }', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=MathResponse(steps=[Step(explanation="First, let's isolate the term with the variable 'x'. To do this, we'll subtract 31 from both sides of the equation.", output='8x + 31 - 31 = 2 - 31'), Step(explanation='By subtracting 31 from both sides, we simplify the equation to 8x = -29.', output='8x = -29'), Step(explanation="Next, let's isolate 'x' by dividing both sides of the equation by 8.", output='8x / 8 = -29 / 8')], final_answer='x = -29/8'))
步骤 #0: explanation="First, let's isolate the term with the variable 'x'. To do this, we'll subtract 31 from both sides of the equation." output='8x + 31 - 31 = 2 - 31'
步骤 #1: explanation='By subtracting 31 from both sides, we simplify the equation to 8x = -29.' output='8x = -29'
步骤 #2: explanation="Next, let's isolate 'x' by dividing both sides of the equation by 8." output='8x / 8 = -29 / 8'
答案: x = -29/8
```

`structural_tag` 的使用示例可在此处找到：[examples/features/structured_outputs](../../examples/features/structured_outputs/README.md)

## 离线推理

离线推理支持相同类型的结构化输出。
要使用它，我们需要通过在 `SamplingParams` 中使用 `StructuredOutputsParams` 类来配置结构化输出。
`StructuredOutputsParams` 中的主要可用选项有：

- `json`
- `regex`
- `choice`
- `grammar`
- `structural_tag`

这些参数的使用方式与上述在线服务示例中的参数相同。以下展示了 `choice` 参数的使用示例：

??? code

    ```python
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    structured_outputs_params = StructuredOutputsParams(choice=["Positive", "Negative"])
    sampling_params = SamplingParams(structured_outputs=structured_outputs_params)
    outputs = llm.generate(
        prompts="Classify this sentiment: vLLM is wonderful!",
        sampling_params=sampling_params,
    )
    print(outputs[0].outputs[0].text)
    ```

另请参见：[完整示例](../../examples/features/structured_outputs/structured_outputs_offline.py)
