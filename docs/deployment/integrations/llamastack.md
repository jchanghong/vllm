# Llama Stack

vLLM 也可通过 [Llama Stack](https://github.com/llamastack/llama-stack) 使用。

要安装 Llama Stack，运行

```bash
pip install llama-stack -q
```

## 使用 OpenAI 兼容 API 进行推理

然后启动 Llama Stack 服务器，并通过以下设置将其指向您的 vLLM 服务器：

```yaml
inference:
  - provider_id: vllm0
    provider_type: remote::vllm
    config:
      url: http://127.0.0.1:8000
```

请参阅[此指南](https://llama-stack.readthedocs.io/en/latest/providers/inference/remote_vllm.html)以获取有关此远程 vLLM 提供程序的更多详细信息。

## 使用嵌入式 vLLM 进行推理

还提供了一个[内联提供程序](https://github.com/llamastack/llama-stack/tree/main/llama_stack/providers/inline/inference)。以下是使用该方法的配置示例：

```yaml
inference:
  - provider_type: vllm
    config:
      model: Llama3.1-8B-Instruct
      tensor_parallel_size: 4
```
