# 自定义参数

您可以使用 vLLM **自定义参数** 来传递不属于 vLLM `SamplingParams` 和 REST API 规范的参数。添加或删除 vLLM 自定义参数不需要重新编译 vLLM，因为自定义参数是以字典形式传递的。

例如，如果您想使用 [自定义 logits 处理器](./custom_logitsprocs.md) 而无需修改 vLLM 源代码，自定义参数会很有用。

!!! note
    请确保您的自定义 logits 处理器已为自定义参数实现 `validate_params`。否则，无效的自定义参数可能导致意外行为。

## 离线自定义参数

以 `dict` 形式传递给 `SamplingParams.extra_args` 的自定义参数，对于任何有权访问 `SamplingParams` 的代码都是可见的：

``` python
SamplingParams(extra_args={"your_custom_arg_name": 67})
```

这允许将尚未属于 `SamplingParams` 部分的参数作为请求的一部分传递给 `LLM`。

## 在线自定义参数

vLLM REST API 允许通过 `vllm_xargs` 将自定义参数传递给 vLLM 服务器。以下示例展示了如何将自定义参数集成到 vLLM REST API 请求中：

``` bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        ...
        "vllm_xargs": {"your_custom_arg": 67}
    }'
```

此外，OpenAI SDK 用户可以通过 `extra_body` 参数访问 `vllm_xargs`：

``` python
batch = await client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    ...,
    extra_body={
        "vllm_xargs": {
            "your_custom_arg": 67
        }
    }
)
```

!!! note
    `vllm_xargs` 在底层被赋值给 `SamplingParams.extra_args`，因此使用 `SamplingParams.extra_args` 的代码与离线和在线场景都兼容。
