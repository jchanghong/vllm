# 与 Hugging Face 的集成

本文档描述了 vLLM 如何与 Hugging Face 库集成。我们将逐步解释当我们运行 `vllm serve` 时底层发生了什么。

假设我们想通过运行 `vllm serve Qwen/Qwen2-7B` 来服务流行的 Qwen 模型。

1. `model` 参数是 `Qwen/Qwen2-7B`。vLLM 通过检查相应的配置文件 `config.json` 来确定该模型是否存在。实现请参见此[代码片段](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L162-L182)。在此过程中：
    - 如果 `model` 参数对应一个已存在的本地路径，vLLM 将直接从该路径加载配置文件。
    - 如果 `model` 参数是由用户名和模型名称组成的 Hugging Face 模型 ID，vLLM 将首先尝试使用 Hugging Face 本地缓存中的配置文件，将 `model` 参数作为模型名称，`--revision` 参数作为修订版本。有关 Hugging Face 缓存工作方式的更多信息，请参见[他们的网站](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome)。
    - 如果 `model` 参数是 Hugging Face 模型 ID 但未在缓存中找到，vLLM 将从 Hugging Face 模型 hub 下载配置文件。实现请参考[此函数](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L91)。输入参数包括 `model` 参数作为模型名称、`--revision` 参数作为修订版本，以及环境变量 `HF_TOKEN` 作为访问模型 hub 的 token。在我们的例子中，vLLM 将下载 [config.json](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json) 文件。

2. 确认模型存在后，vLLM 加载其配置文件并将其转换为字典。实现请参见此[代码片段](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L185-L186)。

3. 接下来，vLLM [检查](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L189)配置字典中的 `model_type` 字段以[生成](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L190-L216)要使用的配置对象。有些 `model_type` 值是 vLLM 直接支持的；请参见[此处](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L48)的列表。如果 `model_type` 不在列表中，vLLM 将使用 [AutoConfig.from_pretrained](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained) 来加载配置类，参数包括 `model`、`--revision` 和 `--trust_remote_code`。请注意：
    - Hugging Face 也有自己的逻辑来确定要使用的配置类。它将再次使用 `model_type` 字段在 transformers 库中搜索类名；请参见[此处](https://github.com/huggingface/transformers/tree/main/src/transformers/models)的支持模型列表。如果未找到 `model_type`，Hugging Face 将使用配置 JSON 文件中的 `auto_map` 字段来确定类名。具体来说，它是 `auto_map` 下的 `AutoConfig` 字段。有关示例，请参见 [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V2.5/blob/main/config.json)。
    - `auto_map` 下的 `AutoConfig` 字段指向模型仓库中的一个模块路径。要创建配置类，Hugging Face 将导入该模块并使用 `from_pretrained` 方法加载配置类。这通常可能导致任意代码执行，因此仅在启用 `--trust_remote_code` 时执行。

4. 随后，vLLM 对配置对象应用一些历史补丁。这些主要与 RoPE 配置有关；实现请参见[此处](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/config.py#L244)。

5. 最后，vLLM 可以找到我们想要初始化的模型类。vLLM 使用配置对象中的 `architectures` 字段来确定要初始化的模型类，因为它维护了从架构名称到模型类的映射，参见[其注册表](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/model_executor/models/registry.py#L80)。如果在注册表中未找到架构名称，则表示 vLLM 不支持此模型架构。对于 `Qwen/Qwen2-7B`，`architectures` 字段是 `["Qwen2ForCausalLM"]`，对应于 [vLLM 代码](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/model_executor/models/qwen2.py#L364)中的 `Qwen2ForCausalLM` 类。该类将根据各种配置自行初始化。

除此之外，vLLM 还有两处依赖于 Hugging Face。

1. **分词器**：vLLM 使用 Hugging Face 的分词器对输入文本进行分词。分词器使用 [AutoTokenizer.from_pretrained](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained) 加载，其中 `model` 参数作为模型名称，`--revision` 参数作为修订版本。也可以通过指定 `vllm serve` 命令中的 `--tokenizer` 参数来使用另一个模型的分词器。其他相关参数包括 `--tokenizer-revision` 和 `--tokenizer-mode`。设置 `VLLM_USE_FASTOKENS=1` 将为 vLLM 加载的任何 HF 快速分词器替换为即插即用的 Rust BPE 后端（参见 [fastokens 后端](../configuration/optimization.md#fastokens-backend)）。请查阅 Hugging Face 的文档了解这些参数的含义。这部分逻辑可以在 [get_tokenizer](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/tokenizer.py#L87) 函数中找到。获取分词器后，值得注意的是，vLLM 会将分词器的一些昂贵属性缓存到 [vllm.tokenizers.hf.get_cached_tokenizer][] 中。

2. **模型权重**：vLLM 使用 `model` 参数作为模型名称，`--revision` 参数作为修订版本，从 Hugging Face 模型 hub 下载模型权重。vLLM 提供了 `--load-format` 参数来控制从模型 hub 下载哪些文件。默认情况下，它将尝试以 safetensors 格式加载权重，如果 safetensors 格式不可用，则回退到 PyTorch bin 格式。我们也可以传递 `--load-format dummy` 来跳过下载权重。
    - 建议使用 safetensors 格式，因为它对于分布式推理中的加载是高效的，并且对任意代码执行也是安全的。有关 safetensors 格式的更多信息，请参见[文档](https://huggingface.co/docs/safetensors/en/index)。这部分逻辑可以在[此处](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/model_executor/model_loader/loader.py#L385)找到。请注意：

以上完成了 vLLM 和 Hugging Face 之间的集成。

总之，vLLM 从 Hugging Face 模型 hub 或本地目录读取配置文件 `config.json`、分词器和模型权重。它使用来自 vLLM、Hugging Face transformers 的配置类，或者从模型仓库加载配置类。
