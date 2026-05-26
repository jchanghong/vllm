# 模型解析

vLLM 通过检查模型仓库中 `config.json` 文件的 `architectures` 字段，并查找注册到 vLLM 的对应实现，来加载 HuggingFace 兼容模型。然而，我们的模型解析可能因以下原因失败：

- 模型仓库的 `config.json` 缺少 `architectures` 字段。
- 非官方仓库使用替代名称引用模型，而 vLLM 中未记录这些名称。
- 多个模型使用相同的架构名称，导致加载哪个模型存在歧义。

要解决此问题，可以通过 `hf_overrides` 选项显式指定模型架构，传递 `config.json` 覆盖参数。例如：

```python
from vllm import LLM

llm = LLM(
    model="cerebras/Cerebras-GPT-1.3B",
    hf_overrides={"architectures": ["GPT2LMHeadModel"]},  # GPT-2
)
```

我们的[支持的模型列表](../models/supported_models.md)显示了 vLLM 识别的模型架构。
