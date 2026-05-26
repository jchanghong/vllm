# 生成式模型

vLLM 对生成式模型提供了一流支持，涵盖了大多数大语言模型 (LLM)。

在 vLLM 中，生成式模型实现了 [VllmModelForTextGeneration][vllm.model_executor.models.VllmModelForTextGeneration] 接口。
基于输入的最终隐藏状态，这些模型输出要生成的 token 的对数概率，
然后通过 [Sampler][vllm.v1.sample.sampler.Sampler] 获得最终文本。

## 配置

### 模型运行器 (`--runner`)

通过 `--runner generate` 选项在生成模式下运行模型。

!!! tip
    绝大多数情况下无需设置此选项，因为 vLLM 可以通过 `--runner auto` 自动检测要使用的模型运行器。

## 离线推理

[LLM][vllm.LLM] 类提供了多种离线推理方法。
有关初始化模型时的选项列表，请参阅[配置](../api/README.md#configuration)。

### `LLM.generate`

[generate][vllm.LLM.generate] 方法适用于 vLLM 中的所有生成式模型。
它类似于 [HF Transformers 中的对应方法](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate)，
不同之处在于 token 化和反 token 化也会自动执行。

```python
from vllm import LLM

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate("Hello, my name is")

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

您还可以通过传递 [SamplingParams][vllm.SamplingParams] 来控制语言生成。
例如，可以通过设置 `temperature=0` 来使用贪心采样：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
params = SamplingParams(temperature=0)
outputs = llm.generate("Hello, my name is", params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

!!! important
    默认情况下，如果 huggingface 模型仓库中存在 `generation_config.json`，vLLM 将应用其中模型创建者推荐的采样参数。在大多数情况下，如果未指定 [SamplingParams][vllm.SamplingParams]，这能为您提供最佳结果。

    但是，如果您更希望使用 vLLM 的默认采样参数，请在创建 [LLM][vllm.LLM] 实例时传递 `generation_config="vllm"`。
代码示例请参见：[examples/basic/offline_inference/basic.py](../../examples/basic/offline_inference/basic.py)

### `LLM.beam_search`

[beam_search][vllm.LLM.beam_search] 方法在 [generate][vllm.LLM.generate] 之上实现了[束搜索](https://huggingface.co/docs/transformers/en/generation_strategies#beam-search)。
例如，使用 5 个束并最多输出 50 个 token：

```python
from vllm import LLM
from vllm.sampling_params import BeamSearchParams

llm = LLM(model="facebook/opt-125m")
params = BeamSearchParams(beam_width=5, max_tokens=50)
outputs = llm.beam_search([{"prompt": "Hello, my name is "}], params)

for output in outputs:
    generated_text = output.sequences[0].text
    print(f"Generated text: {generated_text!r}")
```

### `LLM.chat`

[chat][vllm.LLM.chat] 方法在 [generate][vllm.LLM.generate] 之上实现了聊天功能。
具体来说，它接受类似于 [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) 的输入，
并自动应用模型的[聊天模板](https://huggingface.co/docs/transformers/en/chat_templating)来格式化提示。

!!! important
    通常，只有指令微调模型才有聊天模板。
    基础模型表现可能不佳，因为它们未经训练来响应聊天对话。

??? code

    ```python
    from vllm import LLM

    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant",
        },
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?",
        },
        {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
        },
    ]
    outputs = llm.chat(conversation)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

代码示例请参见：[examples/basic/offline_inference/chat.py](../../examples/basic/offline_inference/chat.py)

如果模型没有聊天模板，或者您想指定其他模板，
您可以显式传递聊天模板：

```python
from vllm.entrypoints.chat_utils import load_chat_template

# 您可以在 `examples/` 目录下找到现有聊天模板列表
custom_template = load_chat_template(chat_template="<path_to_template>")
print("Loaded chat template:", custom_template)

outputs = llm.chat(conversation, chat_template=custom_template)
```

## 在线服务

我们的[兼容 OpenAI 的服务器](../serving/online_serving/openai_compatible_server.md)提供了与离线 API 对应的端点：

- [Completions API](../serving/online_serving/openai_compatible_server.md#completions-api) 类似于 `LLM.generate`，但只接受文本。
- [Chat API](../serving/online_serving/openai_compatible_server.md#chat-api) 类似于 `LLM.chat`，接受文本和[多模态输入](../features/multimodal_inputs.md)（适用于带有聊天模板的模型）。
