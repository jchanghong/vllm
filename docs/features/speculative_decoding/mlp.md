# MLP 草稿模型

以下代码配置 vLLM 使用草稿模型进行投机解码，这些草稿模型基于上下文向量和已采样 token 来生成草稿预测。更多信息请参阅 [《 Hitchhiker's Guide to Speculative Decoding》](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/) 和 [IBM 研究院技术报告](https://arxiv.org/abs/2404.19124)。

## MLP 起草器示例

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    speculative_config={
        "model": "ibm-ai-platform/llama3-8b-accelerator",
        "draft_tensor_parallel_size": 1,
        "method": "mlp_speculator",
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

!!! warning "已知问题"
    `ibm-ai-platform/llama3-70b-accelerator` 可能因以下错误而失败：
    `AttributeError: 'MLPSpeculatorConfig' object has no attribute 'num_attention_heads'`。
    请在 [#34106](https://github.com/vllm-project/vllm/issues/34106)
    和 [#34163](https://github.com/vllm-project/vllm/pull/34163) 中跟踪状态。

## 预训练的 MLP 起草器模型

HF hub 上提供了多种此类类型的推测模型：

- [llama-13b-accelerator](https://huggingface.co/ibm-ai-platform/llama-13b-accelerator)
- [llama3-8b-accelerator](https://huggingface.co/ibm-ai-platform/llama3-8b-accelerator)
- [codellama-34b-accelerator](https://huggingface.co/ibm-ai-platform/codellama-34b-accelerator)
- [llama2-70b-accelerator](https://huggingface.co/ibm-ai-platform/llama2-70b-accelerator)
- [llama3-70b-accelerator](https://huggingface.co/ibm-ai-platform/llama3-70b-accelerator)
- [granite-3b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-3b-code-instruct-accelerator)
- [granite-8b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-8b-code-instruct-accelerator)
- [granite-7b-instruct-accelerator](https://huggingface.co/ibm-granite/granite-7b-instruct-accelerator)
- [granite-20b-code-instruct-accelerator](https://huggingface.co/ibm-granite/granite-20b-code-instruct-accelerator)
