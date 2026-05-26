# 离线推理

`LLM` 类提供了进行离线推理的主要 Python 接口，即在不使用独立模型推理服务器的情况下与模型进行交互。

## 使用方法

此示例中的第一个脚本展示了 vLLM 最基本的用法。如果你是 Python 和 vLLM 的新手，应该从这里开始。

```bash
python examples/basic/offline_inference/basic.py
```

其余脚本包含一个[参数解析器](https://docs.python.org/3/library/argparse.html)，你可以使用它传递任何与 [`LLM`](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html) 兼容的参数。尝试使用 `--help` 运行脚本以查看所有可用参数列表。

```bash
python examples/basic/offline_inference/classify.py
```

```bash
python examples/basic/offline_inference/embed.py
```

```bash
python examples/basic/offline_inference/score.py
```

对话和生成脚本也接受[采样参数](https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-parameters)：`max_tokens`、`temperature`、`top_p` 和 `top_k`。

```bash
python examples/basic/offline_inference/chat.py
```

```bash
python examples/basic/offline_inference/generate.py
```

## 功能特性

在支持传递参数的脚本中，你可以尝试以下功能。

### 默认生成配置

`--generation-config` 参数指定在调用 `LLM.get_default_sampling_params()` 时从何处加载生成配置。如果设置为 'auto'，则从模型路径加载生成配置。如果设置为文件夹路径，则从指定的文件夹路径加载生成配置。如果未提供，则使用 vLLM 默认值。

> 如果在生成配置中指定了 max_new_tokens，则它会对所有请求设置服务器级别的输出令牌数量限制。

尝试使用以下参数：

```bash
--generation-config auto
```

### 量化

#### GGUF

vLLM 支持使用 GGUF 进行量化的模型。

使用 `repo_id:quant_type` 格式直接从 HuggingFace 加载模型进行尝试：

```bash
--model unsloth/Qwen3-0.6B-GGUF:Q4_K_M --tokenizer Qwen/Qwen3-0.6B
```

### CPU 卸载

`--cpu-offload-gb` 参数可以看作是一种虚拟增加 GPU 内存大小的方法。例如，如果你有一个 24 GB 的 GPU 并将此值设为 10，实际上你可以将其视为 34 GB 的 GPU。然后你可以加载一个使用 BF16 权重的 13B 模型，该模型至少需要 26 GB 的 GPU 内存。请注意，这需要快速的 CPU-GPU 互联，因为模型的每个前向传播过程中会动态地从 CPU 内存加载部分模型到 GPU 内存。

尝试使用以下参数：

```bash
--model meta-llama/Llama-2-13b-chat-hf --cpu-offload-gb 10
```
