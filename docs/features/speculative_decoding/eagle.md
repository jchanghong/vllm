# EAGLE 草稿模型

以下代码配置 vLLM 使用基于 [EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）](https://arxiv.org/pdf/2401.15077) 草稿模型生成提议的投机解码。更详细的离线模式示例（包括如何提取请求级别的接受率）可在 [examples/features/speculative_decoding/spec_decode_offline.py](../../../examples/features/speculative_decoding/spec_decode_offline.py) 中找到。

## Eagle 起草器示例

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 2,
        "method": "eagle",
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Eagle3 起草器示例

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=2,
    speculative_config={
        "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        "draft_tensor_parallel_size": 2,
        "num_speculative_tokens": 2,
        "method": "eagle3",
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## 预训练的 Eagle 草稿模型

Hugging Face hub 上提供了多种 EAGLE 草稿模型：

* [RedHatAI/speculator-models](https://huggingface.co/collections/RedHatAI/speculator-models)
* [yuhuili/models](https://huggingface.co/yuhuili/models?search=eagle)

!!! warning
    如果您使用的是 `vllm<0.7.0`，请使用[此脚本](https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d)转换推测模型，并在 `speculative_config` 中指定 `"model": "path/to/modified/eagle/model"`。
