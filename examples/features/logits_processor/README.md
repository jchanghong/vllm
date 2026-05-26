# 自定义 Logits 处理器

本目录包含演示如何使用自定义 logits 处理器与 vLLM 离线推理 API 的示例。Logits 处理器允许您在采样前修改模型的输出分布，从而实现诸如令牌屏蔽、受约束解码和自定义采样策略等可控生成行为。

## 脚本

### `custom.py` — 引擎级 logits 处理器

演示如何使用在批处理级别运行的自定义 logits 处理器类实例化 vLLM。该示例使用了一个 `DummyLogitsProcessor`，当通过 `SamplingParams.extra_args` 传递时，它会屏蔽除指定 `target_token` 之外的所有令牌。

```bash
python examples/features/logits_processor/custom.py
```

### `custom_req.py` — 请求级 logits 处理器包装器

展示如何包装请求级 logits 处理器（它在单个请求上运行）以兼容 vLLM 的批处理级 logits 处理接口。

```bash
python examples/features/logits_processor/custom_req.py
```

### `custom_req_init.py` — 带引擎配置的请求级处理器

包装请求级 logits 处理器的一个特例，其中处理器在初始化期间需要访问引擎配置或模型元数据（例如，词汇表大小、分词器信息）。

```bash
python examples/features/logits_processor/custom_req_init.py
```

## 关键概念

- **批处理级与请求级**：vLLM 为了效率在批处理级别处理 logits。如果您有每请求的处理器，需要使用 `custom_req.py` 和 `custom_req_init.py` 中展示的模式进行包装。
- **`SamplingParams.extra_args`**：使用此参数在每请求基础上向 logits 处理器传递自定义关键字参数（例如 `target_token`）。
- **`DummyLogitsProcessor`**：在 `vllm/test_utils.py` 中提供的参考实现，可用作自定义处理器的起点。

## 延伸阅读

- [vLLM 采样参数](https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-parameters)
- [vLLM LLM API](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html)
