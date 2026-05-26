# 后缀解码 (Suffix Decoding)

以下代码配置 vLLM 使用基于后缀解码（[技术报告](https://arxiv.org/abs/2411.04975)）生成提议的投机解码。

与 n-gram 类似，后缀解码可以通过使用最近生成的 `n` 个 token 进行模式匹配来生成草稿 token。与 n-gram 不同，后缀解码：（1）可以同时对提示和先前生成的内容进行模式匹配；（2）使用频率计数来提出最可能的续写内容；（3）在每次迭代中为每个请求推测自适应数量的 token，以获得更好的接受率。

后缀解码对于高重复性任务（如代码编辑、智能体循环（例如自我反思、自我一致性）和 RL 回滚）可以获得更好的性能。

!!! tip "安装 Arctic Inference"
    后缀解码需要 [Arctic Inference](https://github.com/snowflakedb/ArcticInference)。您可以使用 `pip install arctic-inference` 进行安装。

!!! tip "后缀解码推测 Token"
    后缀解码会在每个解码步骤为每个请求推测动态数量的 token，因此 `num_speculative_tokens` 配置指定的是推测 token 的*最大*数量。建议使用较高的数值，例如 `16` 或 `32`（默认值）。

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 32,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
