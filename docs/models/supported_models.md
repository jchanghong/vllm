# 支持的模型

vLLM 支持跨各种任务的[生成式](./generative_models.md)和[池化](./pooling_models/README.md)模型。

对于每个任务，我们列出了已在 vLLM 中实现的模型架构。
在每个架构旁边，我们列出了一些使用该架构的热门模型。

## 模型实现

### vLLM

如果 vLLM 原生支持某个模型，其实现可以在 [vllm/model_executor/models](../../vllm/model_executor/models) 中找到。

这些模型就是我们列在[支持的纯文本模型](#list-of-text-only-language-models)和[支持的多模态模型](#list-of-multimodal-language-models)中的模型。

### Transformers

vLLM 还支持 Transformers 中可用的模型实现。您应该期望在 vLLM 中使用 Transformers 模型实现的性能与专用 vLLM 模型实现的性能相差在 5% 以内。我们将此功能称为"Transformers 建模后端"。

目前，Transformers 建模后端适用于以下情况：

- 模态：嵌入模型、语言模型和视觉语言模型*
- 架构：仅编码器、仅解码器、混合专家
- 注意力类型：完全注意力和/或滑动注意力

_*视觉语言模型目前仅接受图像输入。视频输入的支持将在未来版本中添加。_

如果 Transformers 模型实现遵循[编写自定义模型](#writing-custom-models)中的所有步骤，那么当与 Transformers 建模后端一起使用时，它将兼容 vLLM 的以下功能：

- [兼容性矩阵](../features/README.md#feature-x-feature)中列出的所有功能
- 以下 vLLM 并行化方案的任意组合：
    - 数据并行
    - 张量并行
    - 专家并行
    - 流水线并行

检查建模后端是否为 Transformers 非常简单：

```python
from vllm import LLM
llm = LLM(model=...)  # 您的模型名称或路径
llm.apply_model(lambda model: print(type(model)))
```

如果打印的类型以 `Transformers...` 开头，则说明正在使用 Transformers 模型实现！

如果某个模型有 vLLM 实现，但您更希望通过 Transformers 建模后端使用 Transformers 实现，请在[离线推理](../serving/offline_inference.md)中设置 `model_impl="transformers"`，或在[在线服务](../serving/online_serving/README.md)中设置 `--model-impl transformers`。

!!! note
    对于视觉语言模型，如果使用 `dtype="auto"` 加载，vLLM 会在配置存在的情况下以配置的 `dtype` 加载整个模型。相比之下，原生 Transformers 会尊重模型中每个骨干网络的 `dtype` 属性。这可能会导致性能上的细微差异。

#### 自定义模型

如果某个模型既不受 vLLM 原生支持，也不受 Transformers 支持，它仍然可以在 vLLM 中使用！

要使模型兼容 vLLM 的 Transformers 建模后端，它必须：

- 是一个兼容 Transformers 的自定义模型（请参阅 [Transformers - 自定义模型](https://huggingface.co/docs/transformers/en/custom_models)）：
    - 模型目录必须具有正确的结构（例如存在 `config.json`）。
    - `config.json` 必须包含 `auto_map.AutoModel`。
- 是一个兼容 vLLM Transformers 建模后端的模型（请参阅[编写自定义模型](#writing-custom-models)）：
    - 自定义应在基础模型中进行（例如在 `MyModel` 中，而不是 `MyModelForCausalLM` 中）。

如果兼容模型位于：

- Hugging Face 模型中心，只需在[离线推理](../serving/offline_inference.md)中设置 `trust_remote_code=True`，或在[在线服务](../serving/online_serving/README.md)中设置 `--trust-remote-code`。
- 本地目录，只需将目录路径传递给[离线推理](../serving/offline_inference.md)的 `model=<MODEL_DIR>`，或[在线服务](../serving/online_serving/README.md)的 `vllm serve <MODEL_DIR>`。

这意味着，借助 vLLM 的 Transformers 建模后端，新模型可以在正式受 Transformers 或 vLLM 支持之前使用！

#### 编写自定义模型

本节详细介绍为使 Transformers 兼容的自定义模型与 vLLM 的 Transformers 建模后端兼容而需要进行的必要修改。（我们假设已经创建了 Transformers 兼容的自定义模型，请参阅 [Transformers - 自定义模型](https://huggingface.co/docs/transformers/en/custom_models)）。

要使您的模型与 Transformers 建模后端兼容，它需要：

1. 通过所有模块从 `MyModel` 向下传递 `kwargs` 到 `MyAttention`。
    - 如果您的模型是仅编码器：
        1. 在 `MyAttention` 中添加 `is_causal = False`。
    - 如果您的模型是混合专家 (MoE)：
        1. 您的稀疏 MoE 块必须有一个名为 `experts` 的属性。
        2. `experts` 的类 (`MyExperts`) 必须：
            - 继承自 `nn.ModuleList`（朴素方式）。
            - 或者包含所有 3D `nn.Parameters`（打包方式）。
        3. `MyExperts.forward` 必须接受 `hidden_states`、`top_k_index`、`top_k_weights`。
2. `MyAttention` 必须使用 `ALL_ATTENTION_FUNCTIONS` 来调用注意力。
3. `MyModel` 必须包含 `_supports_attention_backend = True`。

<details class="code">
<summary>modeling_my_model.py</summary>

```python

from transformers import PreTrainedModel
from torch import nn

class MyAttention(nn.Module):
    is_causal = False  # 仅对仅编码器模型执行此操作

    def forward(self, hidden_states, **kwargs):
        ...
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            **kwargs,
        )
        ...

# 仅对混合专家模型执行此操作
class MyExperts(nn.ModuleList):
    def forward(self, hidden_states, top_k_index, top_k_weights):
        ...

# 仅对混合专家模型执行此操作
class MySparseMoEBlock(nn.Module):
    def __init__(self, config):
        ...
        self.experts = MyExperts(config)
        ...

    def forward(self, hidden_states: torch.Tensor):
        ...
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        ...

class MyModel(PreTrainedModel):
    _supports_attention_backend = True
```

</details>

以下是加载此模型时后台发生的情况：

1. 配置被加载。
2. `MyModel` Python 类从配置中的 `auto_map` 加载，我们检查模型 `is_backend_compatible()`。
3. `MyModel` 被加载到 [vllm/model_executor/models/transformers](../../vllm/model_executor/models/transformers) 中的 Transformers 建模后端类之一，该类设置 `self.config._attn_implementation = "vllm"`，以便使用 vLLM 的注意力层。

就是这样！

要使您的模型兼容 vLLM 的张量并行和/或流水线并行功能，您必须将 `base_model_tp_plan` 和/或 `base_model_pp_plan` 添加到模型的配置类中：

<details class="code">
<summary>configuration_my_model.py</summary>

```python

from transformers import PretrainedConfig

class MyConfig(PretrainedConfig):
    base_model_tp_plan = {
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
```

</details>

- `base_model_tp_plan` 是一个 `dict`，将完全限定的层名称模式映射到张量并行样式（目前仅支持 `"colwise"` 和 `"rowwise"`）。
- `base_model_pp_plan` 是一个 `dict`，将直接子层名称映射到 `list` 的 `tuple` 的 `str`：
    - 您只需要对不在所有流水线阶段上都存在的层执行此操作。
    - vLLM 假设只有一个 `nn.ModuleList`，它分布在各个流水线阶段上。
    - `tuple` 的第一个元素中的 `list` 包含输入参数的名称。
    - `tuple` 的最后一个元素中的 `list` 包含该层在建模代码中输出的变量名称。

### 插件

某些模型架构通过 vLLM 插件得到支持。这些插件通过[插件系统](../design/plugin_system.md)扩展了 vLLM 的功能。

| 架构 | 模型 | 插件仓库 |
| ------------ | ------ | ----------------- |
| `BartForConditionalGeneration` | BART | [bart-plugin](https://github.com/vllm-project/bart-plugin) |
| `Florence2ForConditionalGeneration` | Florence-2 | [bart-plugin](https://github.com/vllm-project/bart-plugin) |

对于其他未原生支持的模型架构，特别是编码器-解码器模型，我们建议遵循类似模式，通过插件系统实现支持。

## 加载模型

### Hugging Face Hub

默认情况下，vLLM 从 [Hugging Face (HF) Hub](https://huggingface.co/models) 加载模型。要更改模型的下载路径，您可以设置 `HF_HOME` 环境变量；更多详情请参阅[其官方文档](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhome)。

要确定给定模型是否原生受支持，您可以检查 HF 仓库中的 `config.json` 文件。
如果 `"architectures"` 字段包含下面列出的模型架构，则说明它应该是原生支持的。

模型并不_需要_原生支持才能在 vLLM 中使用。
[Transformers 建模后端](#transformers)使您能够直接使用 Transformers 实现（甚至是 Hugging Face 模型中心上的远程代码！）运行模型。

!!! tip
    在运行时检查您的模型是否真正受支持的最简单方法是运行以下程序：

    ```python
    from vllm import LLM

    # 仅适用于生成式模型 (runner=generate)
    llm = LLM(model=..., runner="generate")  # 您的模型名称或路径
    output = llm.generate("Hello, my name is")
    print(output)

    # 仅适用于池化模型 (runner=pooling)
    llm = LLM(model=..., runner="pooling")  # 您的模型名称或路径
    output = llm.encode("Hello, my name is")
    print(output)
    ```

    如果 vLLM 成功返回文本（对于生成式模型）或隐藏状态（对于池化模型），则表明您的模型受支持。

否则，请参阅[添加新模型](../contributing/model/README.md)了解如何在 vLLM 中实现您的模型的说明。
或者，您可以[在 GitHub 上提交 issue](https://github.com/vllm-project/vllm/issues/new/choose) 来请求 vLLM 支持。

#### 下载模型

如果您愿意，可以使用 Hugging Face CLI [下载模型](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-download)或模型仓库中的特定文件：

```bash
# 下载模型
hf download HuggingFaceH4/zephyr-7b-beta

# 指定自定义缓存目录
hf download HuggingFaceH4/zephyr-7b-beta --cache-dir ./path/to/cache

# 从模型仓库下载特定文件
hf download HuggingFaceH4/zephyr-7b-beta eval_results.json
```

#### 列出已下载的模型

使用 Hugging Face CLI [管理](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#scan-your-cache)存储在本地缓存中的模型：

```bash
# 列出缓存的模型
hf scan-cache

# 显示详细输出
hf scan-cache -v

# 指定自定义缓存目录
hf scan-cache --dir ~/.cache/huggingface/hub
```

#### 删除缓存的模型

使用 Hugging Face CLI 以交互方式从缓存中[删除已下载的模型](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#clean-your-cache)：

<details>
<summary>命令</summary>

```console
# `delete-cache` 命令需要额外依赖才能与 TUI 配合使用。
# 请运行 `pip install huggingface_hub[cli]` 来安装它们。

# 启动交互式 TUI 选择要删除的模型
$ hf delete-cache
? 选择要删除的修订版本：已选择 1 个修订版本，共 438.9M。
  ○ 以下都不选（如果选择，则不会删除任何内容）。
模型 BAAI/bge-base-en-v1.5 (438.9M, 1 周前使用)
❯ ◉ a5beb1e3: main # 1 周前修改

模型 BAAI/bge-large-en-v1.5 (1.3G, 1 周前使用)
  ○ d4aa6901: main # 1 周前修改

模型 BAAI/bge-reranker-base (1.1G, 4 周前使用)
  ○ 2cfc18c9: main # 4 周前修改

按 <space> 选择，按 <enter> 确认，按 <ctrl+c> 退出而不修改。

# 选择后需要确认
? 选择要删除的修订版本：已选择 1 个修订版本。
? 已选择 1 个修订版本，共 438.9M。确认删除？是
开始删除。
完成。共删除 1 个仓库和 0 个修订版本，总计 438.9M。
```

</details>

#### 使用代理

以下是通过代理从 Hugging Face 加载/下载模型的一些提示：

- 为您的会话全局设置代理（或将其设置在配置文件中）：

```shell
export http_proxy=http://your.proxy.server:port
export https_proxy=http://your.proxy.server:port
```

- 仅为当前命令设置代理：

```shell
https_proxy=http://your.proxy.server:port hf download <model_name>

# 或直接使用 vllm 命令
https_proxy=http://your.proxy.server:port  vllm serve <model_name>
```

- 在 Python 解释器中设置代理：

```python
import os

os.environ["http_proxy"] = "http://your.proxy.server:port"
os.environ["https_proxy"] = "http://your.proxy.server:port"
```

### ModelScope

要使用 [ModelScope](https://www.modelscope.cn) 的模型而不是 Hugging Face Hub，请设置一个环境变量：

```shell
export VLLM_USE_MODELSCOPE=True
```

并与 `trust_remote_code=True` 一起使用。

```python
from vllm import LLM

llm = LLM(model=..., revision=..., runner=..., trust_remote_code=True)

# 仅适用于生成式模型 (runner=generate)
output = llm.generate("Hello, my name is")
print(output)

# 仅适用于池化模型 (runner=pooling)
output = llm.encode("Hello, my name is")
print(output)
```

## 功能状态图例

- ✅︎ 表示该模型支持此功能。

- 🚧 表示该功能已计划但尚未对该模型支持。

- ⚠️ 表示该功能可用但可能存在已知问题或限制。

## 纯文本语言模型列表

### 生成式模型

有关如何使用生成式模型的更多信息，请参见[此页面](generative_models.md)。

#### 文本生成

这些模型主要接受 [`LLM.generate`](./generative_models.md#llmgenerate) API。聊天/指令模型还额外支持 [`LLM.chat`](./generative_models.md#llmchat) API。

<style>
th {
  white-space: nowrap;
  min-width: 0 !important;
}
</style>

| 架构 | 模型 | 示例 HF 模型 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `AfmoeForCausalLM` | Afmoe | 待定 | ✅︎ | ✅︎ |
| `ApertusForCausalLM` | Apertus | `swiss-ai/Apertus-8B-2509`, `swiss-ai/Apertus-70B-Instruct-2509` 等 | ✅︎ | ✅︎ |
| `AquilaForCausalLM` | Aquila, Aquila2 | `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B` 等 | ✅︎ | ✅︎ |
| `ArceeForCausalLM` | Arcee (AFM) | `arcee-ai/AFM-4.5B-Base` 等 | ✅︎ | ✅︎ |
| `ArcticForCausalLM` | Arctic | `Snowflake/snowflake-arctic-base`, `Snowflake/snowflake-arctic-instruct` 等 | | ✅︎ |
| `AXK1ForCausalLM` | A.X-K1 | `skt/A.X-K1` 等 | | ✅︎ |
| `BaiChuanForCausalLM` | Baichuan2, Baichuan | `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B` 等 | ✅︎ | ✅︎ |
| `BailingMoeForCausalLM` | Ling | `inclusionAI/Ling-lite-1.5`, `inclusionAI/Ling-plus` 等 | ✅︎ | ✅︎ |
| `BailingMoeV2ForCausalLM` | Ling | `inclusionAI/Ling-mini-2.0` 等 | ✅︎ | ✅︎ |
| `BailingMoeV2_5ForCausalLM` | Ling | `inclusionAI/Ling-2.5-1T`, `inclusionAI/Ring-2.5-1T` | | ✅︎ |
| `BambaForCausalLM` | Bamba | `ibm-ai-platform/Bamba-9B-fp8`, `ibm-ai-platform/Bamba-9B` | ✅︎ | ✅︎ |
| `BloomForCausalLM` | BLOOM, BLOOMZ, BLOOMChat | `bigscience/bloom`, `bigscience/bloomz` 等 | | ✅︎ |
| `ChatGLMModel`, `ChatGLMForConditionalGeneration` | ChatGLM | `zai-org/chatglm2-6b`, `zai-org/chatglm3-6b`, `thu-coai/ShieldLM-6B-chatglm3` 等 | ✅︎ | ✅︎ |
| `CohereForCausalLM`, `Cohere2ForCausalLM` | Command-R, Command-A | `CohereLabs/c4ai-command-r-v01`, `CohereLabs/c4ai-command-r7b-12-2024`, `CohereLabs/c4ai-command-a-03-2025`, `CohereLabs/command-a-reasoning-08-2025` 等 | ✅︎ | ✅︎ |
| `Cohere2MoeForCausalLM` | Command-A+ | `CohereLabs/command-a-plus-05-2026` 等 | ✅︎ | ✅︎ |
| `CwmForCausalLM` | CWM | `facebook/cwm` 等 | ✅︎ | ✅︎ |
| `DbrxForCausalLM` | DBRX | `databricks/dbrx-base`, `databricks/dbrx-instruct` 等 | | ✅︎ |
| `DeciLMForCausalLM` | DeciLM | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` 等 | ✅︎ | ✅︎ |
| `DeepseekForCausalLM` | DeepSeek | `deepseek-ai/deepseek-llm-67b-base`, `deepseek-ai/deepseek-llm-7b-chat` 等 | ✅︎ | ✅︎ |
| `DeepseekV2ForCausalLM` | DeepSeek-V2 | `deepseek-ai/DeepSeek-V2`, `deepseek-ai/DeepSeek-V2-Chat` 等 | ✅︎ | ✅︎ |
| `DeepseekV3ForCausalLM` | DeepSeek-V3 | `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-V3.1` 等 | ✅︎ | ✅︎ |
| `DeepseekV4ForCausalLM` | DeepSeek-V4 | `deepseek-ai/DeepSeek-V4-Flash`, `deepseek-ai/DeepSeek-V4-Pro` 等 | | ✅︎ |
| `Dots1ForCausalLM` | dots.llm1 | `rednote-hilab/dots.llm1.base`, `rednote-hilab/dots.llm1.inst` 等 | | ✅︎ |
| `DotsOCRForCausalLM` | dots_ocr | `rednote-hilab/dots.ocr` | ✅︎ | ✅︎ |
| `Ernie4_5ForCausalLM` | Ernie4.5 | `baidu/ERNIE-4.5-0.3B-PT` 等 | ✅︎ | ✅︎ |
| `Ernie4_5_MoeForCausalLM` | Ernie4.5MoE | `baidu/ERNIE-4.5-21B-A3B-PT`, `baidu/ERNIE-4.5-300B-A47B-PT` 等 | ✅︎ | ✅︎ |
| `ExaoneForCausalLM` | EXAONE-3 | `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct` 等 | ✅︎ | ✅︎ |
| `ExaoneMoEForCausalLM` | K-EXAONE | `LGAI-EXAONE/K-EXAONE-236B-A23B` 等 | | |
| `Exaone4ForCausalLM` | EXAONE-4 | `LGAI-EXAONE/EXAONE-4.0-32B` 等 | ✅︎ | ✅︎ |
| `Fairseq2LlamaForCausalLM` | Llama (fairseq2 格式) | `mgleize/fairseq2-dummy-Llama-3.2-1B` 等 | ✅︎ | ✅︎ |
| `FalconForCausalLM` | Falcon | `tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b` 等 | | ✅︎ |
| `FalconMambaForCausalLM` | FalconMamba | `tiiuae/falcon-mamba-7b`, `tiiuae/falcon-mamba-7b-instruct` 等 | | ✅︎ |
| `FalconH1ForCausalLM` | Falcon-H1 | `tiiuae/Falcon-H1-34B-Base`, `tiiuae/Falcon-H1-34B-Instruct` 等 | ✅︎ | ✅︎ |
| `FlexOlmoForCausalLM` | FlexOlmo | `allenai/FlexOlmo-7x7B-1T`, `allenai/FlexOlmo-7x7B-1T-RT` 等 | | ✅︎ |
| `GemmaForCausalLM` | Gemma | `google/gemma-2b`, `google/gemma-1.1-2b-it` 等 | ✅︎ | ✅︎ |
| `Gemma2ForCausalLM` | Gemma 2 | `google/gemma-2-9b`, `google/gemma-2-27b` 等 | ✅︎ | ✅︎ |
| `Gemma3ForCausalLM` | Gemma 3 | `google/gemma-3-1b-it` 等 | ✅︎ | ✅︎ |
| `Gemma3nForCausalLM` | Gemma 3n | `google/gemma-3n-E2B-it`, `google/gemma-3n-E4B-it` 等 | | |
| `Gemma4ForCausalLM` | Gemma 4 | `google/gemma-4-E2B-it` 等 | ✅︎ | ✅︎ |
| `GlmForCausalLM` | GLM-4 | `zai-org/glm-4-9b-chat-hf` 等 | ✅︎ | ✅︎ |
| `Glm4ForCausalLM` | GLM-4-0414 | `zai-org/GLM-4-32B-0414` 等 | ✅︎ | ✅︎ |
| `Glm4MoeForCausalLM` | GLM-4.5, GLM-4.6, GLM-4.7 | `zai-org/GLM-4.5` 等 | ✅︎ | ✅︎ |
| `Glm4MoeLiteForCausalLM` | GLM-4.7-Flash | `zai-org/GLM-4.7-Flash` 等 | ✅︎ | ✅︎ |
| `GPT2LMHeadModel` | GPT-2 | `openai-community/gpt2`, `openai-community/gpt2-xl` 等 | | ✅︎ |
| `GPTBigCodeForCausalLM` | StarCoder, SantaCoder, WizardCoder | `bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, `WizardLM/WizardCoder-15B-V1.0` 等 | ✅︎ | ✅︎ |
| `GPTJForCausalLM` | GPT-J | `EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j` 等 | | ✅︎ |
| `GPTNeoXForCausalLM` | GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM | `EleutherAI/gpt-neox-20b`, `EleutherAI/pythia-12b`, `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b` 等 | | ✅︎ |
| `GptOssForCausalLM` | GPT-OSS | `openai/gpt-oss-120b`, `openai/gpt-oss-20b` | ✅︎ | ✅︎ |
| `GraniteForCausalLM` | Granite 3.0, Granite 3.1, PowerLM | `ibm-granite/granite-3.0-2b-base`, `ibm-granite/granite-3.1-8b-instruct`, `ibm/PowerLM-3b` 等 | ✅︎ | ✅︎ |
| `GraniteMoeForCausalLM` | Granite 3.0 MoE, PowerMoE | `ibm-granite/granite-3.0-1b-a400m-base`, `ibm-granite/granite-3.0-3b-a800m-instruct`, `ibm/PowerMoE-3b` 等 | ✅︎ | ✅︎ |
| `GraniteMoeHybridForCausalLM` | Granite 4.0 MoE 混合 | `ibm-granite/granite-4.0-tiny-preview` 等 | ✅︎ | ✅︎ |
| `GraniteMoeSharedForCausalLM` | Granite MoE 共享 | `ibm-research/moe-7b-1b-active-shared-experts` (测试模型) | ✅︎ | ✅︎ |
| `GritLM` | GritLM | `parasail-ai/GritLM-7B-vllm` | ✅︎ | ✅︎ |
| `Grok1ModelForCausalLM` | Grok1 | `hpcai-tech/grok-1` | ✅︎ | ✅︎ |
| `Grok1ForCausalLM` | Grok2 | `xai-org/grok-2` | ✅︎ | ✅︎ |
| `HunYuanDenseV1ForCausalLM` | Hunyuan Dense | `tencent/Hunyuan-7B-Instruct` | ✅︎ | ✅︎ |
| `HunYuanMoEV1ForCausalLM` | Hunyuan-A13B | `tencent/Hunyuan-A13B-Instruct`, `tencent/Hunyuan-A13B-Pretrain`, `tencent/Hunyuan-A13B-Instruct-FP8` 等 | ✅︎ | ✅︎ |
| `HYV3ForCausalLM` | HY3 | `tencent/Hy3-preview-Base`, `tencent/Hy3-preview` | ✅︎ | ✅︎ |
| `HyperCLOVAXForCausalLM` | HyperCLOVAX-SEED-Think-14B | `naver-hyperclovax/HyperCLOVAX-SEED-Think-14B` | ✅︎ | ✅︎ |
| `InternLMForCausalLM` | InternLM | `internlm/internlm-7b`, `internlm/internlm-chat-7b` 等 | ✅︎ | ✅︎ |
| `InternLM2ForCausalLM` | InternLM2 | `internlm/internlm2-7b`, `internlm/internlm2-chat-7b` 等 | ✅︎ | ✅︎ |
| `InternLM3ForCausalLM` | InternLM3 | `internlm/internlm3-8b-instruct` 等 | ✅︎ | ✅︎ |
| `IQuestCoderForCausalLM` | IQuestCoderV1 | `IQuestLab/IQuest-Coder-V1-40B-Instruct` 等 | | |
| `IQuestLoopCoderForCausalLM` | IQuestLoopCoderV1 | `IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct` 等 | | |
| `JAISLMHeadModel` | Jais | `inceptionai/jais-13b`, `inceptionai/jais-13b-chat`, `inceptionai/jais-30b-v3`, `inceptionai/jais-30b-chat-v3` 等 | | ✅︎ |
| `Jais2ForCausalLM` | Jais2 | `inceptionai/Jais-2-8B-Chat`, `inceptionai/Jais-2-70B-Chat` 等 | | ✅︎ |
| `JambaForCausalLM` | Jamba | `ai21labs/AI21-Jamba-1.5-Large`, `ai21labs/AI21-Jamba-1.5-Mini`, `ai21labs/Jamba-v0.1` 等 | ✅︎ | ✅︎ |
| `KimiLinearForCausalLM` | Kimi-Linear-48B-A3B-Base, Kimi-Linear-48B-A3B-Instruct | `moonshotai/Kimi-Linear-48B-A3B-Base`, `moonshotai/Kimi-Linear-48B-A3B-Instruct` | | ✅︎ |
| `Lfm2ForCausalLM` | LFM2 | `LiquidAI/LFM2-1.2B`, `LiquidAI/LFM2-700M`, `LiquidAI/LFM2-350M` 等 | ✅︎ | ✅︎ |
| `Lfm2MoeForCausalLM` | LFM2MoE | `LiquidAI/LFM2-8B-A1B-preview` 等 | ✅︎ | ✅︎ |
| `LlamaForCausalLM` | Llama 3.1, Llama 3, Llama 2, LLaMA, Yi | `meta-llama/Meta-Llama-3.1-405B-Instruct`, `meta-llama/Meta-Llama-3.1-70B`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, `01-ai/Yi-34B` 等 | ✅︎ | ✅︎ |
| `LongcatFlashForCausalLM` | LongCat-Flash | `meituan-longcat/LongCat-Flash-Chat`, `meituan-longcat/LongCat-Flash-Chat-FP8` | ✅︎ | ✅︎ |
| `MambaForCausalLM` | Mamba | `state-spaces/mamba-130m-hf`, `state-spaces/mamba-790m-hf`, `state-spaces/mamba-2.8b-hf` 等 | | ✅︎ |
| `Mamba2ForCausalLM` | Mamba2 | `mistralai/Mamba-Codestral-7B-v0.1` 等 | | ✅︎ |
| `MiMoForCausalLM` | MiMo | `XiaomiMiMo/MiMo-7B-RL` 等 | ✅︎ | ✅︎ |
| `MiMoV2FlashForCausalLM` | MiMoV2Flash | `XiaomiMiMo/MiMo-V2-Flash` 等 | | ✅︎ |
| `MiMoV2ForCausalLM` | MiMoV2Pro | `XiaomiMiMo/MiMo-V2.5-Pro` 等 | | ✅︎ |
| `MiniCPMForCausalLM` | MiniCPM | `openbmb/MiniCPM-2B-sft-bf16`, `openbmb/MiniCPM-2B-dpo-bf16`, `openbmb/MiniCPM-S-1B-sft` 等 | ✅︎ | ✅︎ |
| `MiniCPM3ForCausalLM` | MiniCPM3 | `openbmb/MiniCPM3-4B` 等 | ✅︎ | ✅︎ |
| `MiniMaxForCausalLM` | MiniMax-Text | `MiniMaxAI/MiniMax-Text-01-hf` 等 | | |
| `MiniMaxM2ForCausalLM` | MiniMax-M2, MiniMax-M2.1 | `MiniMaxAI/MiniMax-M2` 等 | ✅︎ | ✅︎ |
| `MistralForCausalLM` | Ministral-3, Mistral, Mistral-Instruct | `mistralai/Ministral-3-3B-Instruct-2512`, `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1` 等 | ✅︎ | ✅︎ |
| `MistralLarge3ForCausalLM` | Mistral-Large-3-675B-Base-2512, Mistral-Large-3-675B-Instruct-2512 | `mistralai/Mistral-Large-3-675B-Base-2512`, `mistralai/Mistral-Large-3-675B-Instruct-2512` 等 | ✅︎ | ✅︎ |
| `MixtralForCausalLM` | Mixtral-8x7B, Mixtral-8x7B-Instruct | `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistral-community/Mixtral-8x22B-v0.1` 等 | ✅︎ | ✅︎ |
| `MPTForCausalLM` | MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter | `mosaicml/mpt-7b`, `mosaicml/mpt-7b-storywriter`, `mosaicml/mpt-30b` 等 | | ✅︎ |
| `NemotronForCausalLM` | Nemotron-3, Nemotron-4, Minitron | `nvidia/Minitron-8B-Base`, `mgoin/Nemotron-4-340B-Base-hf-FP8` 等 | ✅︎ | ✅︎ |
| `NemotronHForCausalLM` | Nemotron-H | `nvidia/Nemotron-H-8B-Base-8K`, `nvidia/Nemotron-H-47B-Base-8K`, `nvidia/Nemotron-H-56B-Base-8K` 等 | ✅︎ | ✅︎ |
| `OlmoForCausalLM` | OLMo | `allenai/OLMo-1B-hf`, `allenai/OLMo-7B-hf` 等 | ✅︎ | ✅︎ |
| `Olmo2ForCausalLM` | OLMo2 | `allenai/OLMo-2-0425-1B` 等 | ✅︎ | ✅︎ |
| `Olmo3ForCausalLM` | OLMo3 | `allenai/Olmo-3-7B-Instruct`, `allenai/Olmo-3-32B-Think` 等 | ✅︎ | ✅︎ |
| `OlmoHybridForCausalLM` | OLMo 混合 | `allenai/Olmo-Hybrid-7B` | ✅︎ | ✅︎ |
| `OlmoeForCausalLM` | OLMoE | `allenai/OLMoE-1B-7B-0924`, `allenai/OLMoE-1B-7B-0924-Instruct` 等 | | ✅︎ |
| `OPTForCausalLM` | OPT, OPT-IML | `facebook/opt-66b`, `facebook/opt-iml-max-30b` 等 | ✅︎ | ✅︎ |
| `OrionForCausalLM` | Orion | `OrionStarAI/Orion-14B-Base`, `OrionStarAI/Orion-14B-Chat` 等 | | ✅︎ |
| `OuroForCausalLM` | ouro | `ByteDance/Ouro-1.4B`, `ByteDance/Ouro-2.6B` 等 | ✅︎ | |
| `PanguEmbeddedForCausalLM` | openPangu-Embedded-7B | `FreedomIntelligence/openPangu-Embedded-7B-V1.1` | ✅︎ | ✅︎ |
| `PanguProMoEV2ForCausalLM` | openpangu-pro-moe-v2 | | ✅︎ | ✅︎ |
| `PanguUltraMoEForCausalLM` | openpangu-ultra-moe-718b-model | `FreedomIntelligence/openPangu-Ultra-MoE-718B-V1.1` | ✅︎ | ✅︎ |
| `Param2MoEForCausalLM` | param2moe | `bharatgenai/Param2-17B-A2.4B-Thinking` 等 | ✅︎ | ✅︎ |
| `PhiForCausalLM` | Phi | `microsoft/phi-1_5`, `microsoft/phi-2` 等 | ✅︎ | ✅︎ |
| `Phi3ForCausalLM` | Phi-4, Phi-3 | `microsoft/Phi-4-mini-instruct`, `microsoft/Phi-4`, `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-3-mini-128k-instruct`, `microsoft/Phi-3-medium-128k-instruct` 等 | ✅︎ | ✅︎ |
| `PhiMoEForCausalLM` | Phi-3.5-MoE | `microsoft/Phi-3.5-MoE-instruct` 等 | ✅︎ | ✅︎ |
| `PersimmonForCausalLM` | Persimmon | `adept/persimmon-8b-base`, `adept/persimmon-8b-chat` 等 | | ✅︎ |
| `Plamo2ForCausalLM` | PLaMo2 | `pfnet/plamo-2-1b`, `pfnet/plamo-2-8b` 等 | ✅ | ✅︎ |
| `Plamo3ForCausalLM` | PLaMo3 | `pfnet/plamo-3-nict-2b-base`, `pfnet/plamo-3-nict-8b-base` 等 | ✅ | ✅︎ |
| `QWenLMHeadModel` | Qwen | `Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat` 等 | ✅︎ | ✅︎ |
| `Qwen2ForCausalLM` | QwQ, Qwen2 | `Qwen/QwQ-32B-Preview`, `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen2-7B` 等 | ✅︎ | ✅︎ |
| `Qwen2MoeForCausalLM` | Qwen2MoE | `Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat` 等 | ✅︎ | ✅︎ |
| `Qwen3ForCausalLM` | Qwen3 | `Qwen/Qwen3-8B` 等 | ✅︎ | ✅︎ |
| `Qwen3MoeForCausalLM` | Qwen3MoE | `Qwen/Qwen3-30B-A3B` 等 | ✅︎ | ✅︎ |
| `Qwen3NextForCausalLM` | Qwen3NextMoE | `Qwen/Qwen3-Next-80B-A3B-Instruct` 等 | ✅︎ | ✅︎ |
| `RWForCausalLM` | Falcon RW | `tiiuae/falcon-40b` 等 | | ✅︎ |
| `Rnj1ForCausalLM` | Rnj1 | `EssentialAI/rnj-1-instruct` 等 | | |
| `SarvamMoEForCausalLM` | Sarvam 2 | `sarvamai/sarvam2-30b-a3b` 等 | ✅︎ | ✅︎ |
| `SarvamMLAForCausalLM` | Sarvam 2 | `sarvamai/sarvam2-105b-a9b` 等 | | ✅︎ |
| `SeedOssForCausalLM` | SeedOss | `ByteDance-Seed/Seed-OSS-36B-Instruct` 等 | ✅︎ | ✅︎ |
| `SolarForCausalLM` | Solar Pro | `upstage/solar-pro-preview-instruct` 等 | ✅︎ | ✅︎ |
| `StableLmForCausalLM` | StableLM | `stabilityai/stablelm-3b-4e1t`, `stabilityai/stablelm-base-alpha-7b-v2` 等 | | |
| `StableLMEpochForCausalLM` | StableLM Epoch | `stabilityai/stablelm-zephyr-3b` 等 | | ✅︎ |
| `Starcoder2ForCausalLM` | Starcoder2 | `bigcode/starcoder2-3b`, `bigcode/starcoder2-7b`, `bigcode/starcoder2-15b` 等 | | ✅︎ |
| `Step1ForCausalLM` | Step-Audio | `stepfun-ai/Step-Audio-EditX` 等 | ✅︎ | ✅︎ |
| `Step3p5ForCausalLM` | Step-3.5-flash | `stepfun-ai/Step-3.5-Flash` 等 | | ✅︎ |
| `TeleChatForCausalLM` | TeleChat | `chuhac/TeleChat2-35B` 等 | ✅︎ | ✅︎ |
| `TeleChat2ForCausalLM` | TeleChat2 | `Tele-AI/TeleChat2-3B`, `Tele-AI/TeleChat2-7B`, `Tele-AI/TeleChat2-35B` 等 | ✅︎ | ✅︎ |
| `TeleChat3ForCausalLM` | TeleChat3 | `Tele-AI/TeleChat3-36B-Thinking`, `Tele-AI/TeleChat3-Coder-36B-Thinking` 等 | ✅︎ | ✅︎ |
| `TeleFLMForCausalLM` | TeleFLM | `CofeAI/FLM-2-52B-Instruct-2407`, `CofeAI/Tele-FLM` 等 | ✅︎ | ✅︎ |
| `XverseForCausalLM` | XVERSE | `xverse/XVERSE-7B-Chat`, `xverse/XVERSE-13B-Chat`, `xverse/XVERSE-65B-Chat` 等 | ✅︎ | ✅︎ |
| `MiniMaxM1ForCausalLM` | MiniMax-Text | `MiniMaxAI/MiniMax-M1-40k`, `MiniMaxAI/MiniMax-M1-80k` 等 | | |
| `MiniMaxText01ForCausalLM` | MiniMax-Text | `MiniMaxAI/MiniMax-Text-01` 等 | | |
| `Zamba2ForCausalLM` | Zamba2 | `Zyphra/Zamba2-7B-instruct`, `Zyphra/Zamba2-2.7B-instruct`, `Zyphra/Zamba2-1.2B-instruct` 等 | | |

!!! note
    Grok2 需要安装 `tokenizer.tok.json` 与 `tiktoken`。您可以选择使用 `moe_router_renormalize` 覆盖 MoE 路由器的重新归一化。

某些模型仅通过 [Transformers 建模后端](#transformers) 支持。下表的目的在于确认我们以此方式正式支持的模型。日志会显示正在使用 Transformers 建模后端，并且您不会看到任何表示这是回退行为的警告。这意味着，如果您在使用下面列出的任何模型时遇到问题，请[提交 issue](https://github.com/vllm-project/vllm/issues/new/choose)，我们会尽力修复！

| 架构 | 模型 | 示例 HF 模型 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `SmolLM3ForCausalLM` | SmolLM3 | `HuggingFaceTB/SmolLM3-3B` | ✅︎ | ✅︎ |

!!! note
    目前，ROCm 版本的 vLLM 仅支持上下文长度不超过 4096 的 Mistral 和 Mixtral。

## 多模态语言模型列表

以下模态根据模型的不同而支持：

- **T**ext (文本)
- **I**mage (图像)
- **V**ideo (视频)
- **A**udio (音频)

由 `+` 连接的模态组合是支持的。

- 例如：`T + I` 表示模型支持纯文本、纯图像以及文本+图像输入。

另一方面，由 `/` 分隔的模态是互斥的。

- 例如：`T / I` 表示模型支持纯文本和纯图像输入，但不支持文本+图像输入。

请参见[此页面](../features/multimodal_inputs.md)了解如何向模型传递多模态输入。

!!! tip
    对于仅混合模型，例如 Llama-4、Step3、Mistral-3 和 Qwen-3.5，可以通过将所有支持的多模态模态设置为 0 (`--language-model-only`) 来启用纯文本模式，这样它们的多模态模块将不会被加载，从而为 KV 缓存释放更多 GPU 内存。

!!! note
    vLLM 目前支持对大多数多模态模型的语言骨干网络添加 LoRA 适配器。此外，vLLM 现在实验性地支持对某些多模态模型的视觉塔和连接器模块添加 LoRA。请参见[此页面](../features/lora.md)。

### 生成式模型

有关如何使用生成式模型的更多信息，请参见[此页面](generative_models.md)。

#### 文本生成

这些模型主要接受 [`LLM.generate`](./generative_models.md#llmgenerate) API。聊天/指令模型还额外支持 [`LLM.chat`](./generative_models.md#llmchat) API。

| 架构 | 模型 | 输入 | 示例 HF 模型 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | -------------------- | ------------------------- |
| `AriaForConditionalGeneration` | Aria | T + I<sup>+</sup> | `rhymes-ai/Aria` | | |
| `AudioFlamingo3ForConditionalGeneration` | AudioFlamingo3 | T + A | `nvidia/audio-flamingo-3-hf`, `nvidia/music-flamingo-hf` | ✅︎ | ✅︎ |
| `AyaVisionForConditionalGeneration` | Aya Vision | T + I<sup>+</sup> | `CohereLabs/aya-vision-8b`, `CohereLabs/aya-vision-32b` 等 | | ✅︎ |
| `BagelForConditionalGeneration` | BAGEL | T + I<sup>+</sup> | `ByteDance-Seed/BAGEL-7B-MoT` | ✅︎ | ✅︎ |
| `BeeForConditionalGeneration` | Bee-8B | T + I<sup>E+</sup> | `Open-Bee/Bee-8B-RL`, `Open-Bee/Bee-8B-SFT` | | ✅︎ |
| `Blip2ForConditionalGeneration` | BLIP-2 | T + I<sup>E</sup> | `Salesforce/blip2-opt-2.7b`, `Salesforce/blip2-opt-6.7b` 等 | ✅︎ | ✅︎ |
| `ChameleonForConditionalGeneration` | Chameleon | T + I | `facebook/chameleon-7b` 等 | | ✅︎ |
| `CheersForConditionalGeneration` | Cheers | T + I | `ai9stars/Cheers` | | ✅︎ |
| `Cohere2VisionForConditionalGeneration` | Command A Vision | T + I<sup>+</sup> | `CohereLabs/command-a-vision-07-2025` 等 | | ✅︎ |
| `DeepseekVLV2ForCausalLM` | DeepSeek-VL2 | T + I<sup>+</sup> | `deepseek-ai/deepseek-vl2-tiny`, `deepseek-ai/deepseek-vl2-small`, `deepseek-ai/deepseek-vl2` 等 | | ✅︎ |
| `DeepseekOCRForCausalLM` | DeepSeek-OCR | T + I<sup>+</sup> | `deepseek-ai/DeepSeek-OCR` 等 | ✅︎ | ✅︎ |
| `DeepseekOCR2ForCausalLM` | DeepSeek-OCR-2 | T + I<sup>+</sup> | `deepseek-ai/DeepSeek-OCR-2` 等 | ✅︎ | ✅︎ |
| `Eagle2_5_VLForConditionalGeneration` | Eagle2.5-VL | T + I<sup>E+</sup> | `nvidia/Eagle2.5-8B` 等 | ✅︎ | ✅︎ |
| `Ernie4_5_VLMoeForConditionalGeneration` | Ernie4.5-VL | T + I<sup>+</sup>/ V<sup>+</sup> | `baidu/ERNIE-4.5-VL-28B-A3B-PT`, `baidu/ERNIE-4.5-VL-424B-A47B-PT` | | ✅︎ |
| `Exaone4_5_ForConditionalGeneration` | EXAONE-4.5 | T + I<sup>E+</sup> | `LGAI-EXAONE/EXAONE-4.5-33B` 等 | ✅︎ | ✅︎ |
| `FuyuForCausalLM` | Fuyu | T + I | `adept/fuyu-8b` 等 | | ✅︎ |
| `Gemma3ForConditionalGeneration` | Gemma 3 | T + I<sup>E+</sup> | `google/gemma-3-4b-it`, `google/gemma-3-27b-it` 等 | ✅︎ | ✅︎ |
| `Gemma3nForConditionalGeneration` | Gemma 3n | T + I + A | `google/gemma-3n-E2B-it`, `google/gemma-3n-E4B-it` 等 | | |
| `Gemma4ForConditionalGeneration` | Gemma 4 | T + I<sup>+</sup> + V + A<sup>*</sup> | `google/gemma-4-E2B-it` 等 | | ✅︎ |
| `GLM4VForCausalLM`<sup>^</sup> | GLM-4V | T + I | `zai-org/glm-4v-9b`, `zai-org/cogagent-9b-20241220` 等 | ✅︎ | ✅︎ |
| `Glm4vForConditionalGeneration` | GLM-4.1V-Thinking | T + I<sup>E+</sup> + V<sup>E+</sup> | `zai-org/GLM-4.1V-9B-Thinking` 等 | ✅︎ | ✅︎ |
| `Glm4vMoeForConditionalGeneration` | GLM-4.5V | T + I<sup>E+</sup> + V<sup>E+</sup> | `zai-org/GLM-4.5V` 等 | ✅︎ | ✅︎ |
| `GlmOcrForConditionalGeneration` | GLM-OCR | T + I<sup>E+</sup> | `zai-org/GLM-OCR` 等 | ✅︎ | ✅︎ |
| `Granite4VisionForConditionalGeneration` | Granite 4 Vision | T + I<sup>E+</sup> | `ibm-granite/granite-4.1-3b-vision` 等 | ✅︎ | ✅︎ |
| `GraniteSpeechForConditionalGeneration` | Granite Speech | T + A | `ibm-granite/granite-speech-3.3-8b` | ✅︎ | ✅︎ |
| `HCXVisionForCausalLM` | HyperCLOVAX-SEED-Vision-Instruct-3B | T + I<sup>+</sup> + V<sup>+</sup> | `naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B` | | |
| `HCXVisionV2ForCausalLM` | HyperCLOVAX-SEED-Think-32B | T + I<sup>+</sup> + V<sup>+</sup> | `naver-hyperclovax/HyperCLOVAX-SEED-Think-32B` | | |
| `H2OVLChatModel` | H2OVL | T + I<sup>E+</sup> | `h2oai/h2ovl-mississippi-800m`, `h2oai/h2ovl-mississippi-2b` 等 | ✅︎ | ✅︎ |
| `HunYuanVLForConditionalGeneration` | HunyuanOCR | T + I<sup>E+</sup> | `tencent/HunyuanOCR` 等 | ✅︎ | ✅︎ |
| `Idefics3ForConditionalGeneration` | Idefics3 | T + I | `HuggingFaceM4/Idefics3-8B-Llama3` 等 | ✅︎ | |
| `IsaacForConditionalGeneration` | Isaac | T + I<sup>+</sup> | `PerceptronAI/Isaac-0.1` | ✅︎ | ✅︎ |
| `InternS1ForConditionalGeneration` | Intern-S1 | T + I<sup>E+</sup> + V<sup>E+</sup> | `internlm/Intern-S1`, `internlm/Intern-S1-mini` 等 | ✅︎ | ✅︎ |
| `InternS1ProForConditionalGeneration` | Intern-S1-Pro | T + I<sup>E+</sup> + V<sup>E+</sup> | `internlm/Intern-S1-Pro` 等 | ✅︎ | ✅︎ |
| `InternS2PreviewForConditionalGeneration` | Intern-S2-Preview | T + I<sup>E+</sup> + V<sup>E+</sup> | `internlm/Intern-S2-Preview` 等 | ✅︎ | ✅︎ |
| `InternVLChatModel` | InternVL 3.5, InternVL 3.0, InternVideo 2.5, InternVL 2.5, Mono-InternVL, InternVL 2.0 | T + I<sup>E+</sup> + (V<sup>E+</sup>) | `OpenGVLab/InternVL3_5-14B`, `OpenGVLab/InternVL3-9B`, `OpenGVLab/InternVideo2_5_Chat_8B`, `OpenGVLab/InternVL2_5-4B`, `OpenGVLab/Mono-InternVL-2B`, `OpenGVLab/InternVL2-4B` 等 | ✅︎ | ✅︎ |
| `InternVLForConditionalGeneration` | InternVL 3.0 (HF 格式) | T + I<sup>E+</sup> + V<sup>E+</sup> | `OpenGVLab/InternVL3-1B-hf` 等 | ✅︎ | ✅︎ |
| `KananaVForConditionalGeneration` | Kanana-V | T + I<sup>+</sup> | `kakaocorp/kanana-1.5-v-3b-instruct` 等 | | ✅︎ |
| `KeyeForConditionalGeneration` | Keye-VL-8B-Preview | T + I<sup>E+</sup> + V<sup>E+</sup> | `Kwai-Keye/Keye-VL-8B-Preview` | ✅︎ | ✅︎ |
| `KeyeVL1_5ForConditionalGeneration` | Keye-VL-1_5-8B | T + I<sup>E+</sup> + V<sup>E+</sup> | `Kwai-Keye/Keye-VL-1_5-8B` | ✅︎ | ✅︎ |
| `KimiAudioForConditionalGeneration` | Kimi-Audio | T + A<sup>+</sup> | `moonshotai/Kimi-Audio-7B-Instruct` | | ✅︎ |
| `KimiK25ForConditionalGeneration` | Kimi-K2.5 | T + I<sup>+</sup> | `moonshotai/Kimi-K2.5` | | ✅︎ |
| `KimiVLForConditionalGeneration` | Kimi-VL-A3B-Instruct, Kimi-VL-A3B-Thinking | T + I<sup>+</sup> | `moonshotai/Kimi-VL-A3B-Instruct`, `moonshotai/Kimi-VL-A3B-Thinking` | | ✅︎ |
| `LightOnOCRForConditionalGeneration` | LightOnOCR-1B | T + I<sup>+</sup> | `lightonai/LightOnOCR-1B` 等 | ✅︎ | ✅︎ |
| `Lfm2VlForConditionalGeneration` | LFM2-VL | T + I<sup>+</sup> | `LiquidAI/LFM2-VL-450M`, `LiquidAI/LFM2-VL-3B`, `LiquidAI/LFM2-VL-8B-A1B` 等 | ✅︎ | ✅︎ |
| `Llama4ForConditionalGeneration` | Llama 4 | T + I<sup>+</sup> | `meta-llama/Llama-4-Scout-17B-16E-Instruct`, `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`, `meta-llama/Llama-4-Maverick-17B-128E-Instruct` 等 | ✅︎ | ✅︎ |
| `Llama_Nemotron_Nano_VL` | Llama Nemotron Nano VL | T + I<sup>E+</sup> | `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` | ✅︎ | ✅︎ |
| `LlavaForConditionalGeneration` | LLaVA-1.5, Pixtral (HF Transformers) | T + I<sup>E+</sup> | `llava-hf/llava-1.5-7b-hf`, `TIGER-Lab/Mantis-8B-siglip-llama3` (见附注), `mistral-community/pixtral-12b` 等 | ✅︎ | ✅︎ |
| `LlavaNextForConditionalGeneration` | LLaVA-NeXT, Granite Vision | T + I<sup>E+</sup> | `llava-hf/llava-v1.6-mistral-7b-hf`, `llava-hf/llava-v1.6-vicuna-7b-hf`, `ibm-granite/granite-vision-3.3-2b` 等 | | ✅︎ |
| `LlavaNextVideoForConditionalGeneration` | LLaVA-NeXT-Video | T + V | `llava-hf/LLaVA-NeXT-Video-7B-hf` 等 | | ✅︎ |
| `LlavaOnevisionForConditionalGeneration` | LLaVA-Onevision | T + I<sup>+</sup> + V<sup>+</sup> | `llava-hf/llava-onevision-qwen2-7b-ov-hf`, `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` 等 | | ✅︎ |
| `MiDashengLMModel` | MiDashengLM | T + A<sup>+</sup> | `mispeech/midashenglm-7b` | | ✅︎ |
| `MiMoV2OmniForCausalLM` | MiMo-V2.5-Omni | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>+</sup> | `XiaomiMiMo/MiMo-V2.5-Omni` | | ✅︎ |
| `MiniCPMO` | MiniCPM-O | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>E+</sup> | `openbmb/MiniCPM-o-2_6` 等 | ✅︎ | ✅︎ |
| `MiniCPMV` | MiniCPM-V | T + I<sup>E+</sup> + V<sup>E+</sup> | `openbmb/MiniCPM-V-2` (见附注), `openbmb/MiniCPM-Llama3-V-2_5`, `openbmb/MiniCPM-V-2_6`, `openbmb/MiniCPM-V-4`, `openbmb/MiniCPM-V-4_5` 等 | ✅︎ | |
| `MiniMaxVL01ForConditionalGeneration` | MiniMax-VL | T + I<sup>E+</sup> | `MiniMaxAI/MiniMax-VL-01` 等 | | ✅︎ |
| `Mistral3ForConditionalGeneration` | Mistral3 (HF Transformers) | T + I<sup>+</sup> | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` 等 | ✅︎ | ✅︎ |
| `MolmoForCausalLM` | Molmo | T + I<sup>+</sup> | `allenai/Molmo-7B-D-0924`, `allenai/Molmo-7B-O-0924` 等 | ✅︎ | ✅︎ |
| `Molmo2ForConditionalGeneration` | Molmo2 | T + I<sup>+</sup> / V | `allenai/Molmo2-4B`, `allenai/Molmo2-8B`, `allenai/Molmo2-O-7B`, `allenai/MolmoWeb-4B`<sup>^</sup>, `allenai/MolmoWeb-8B`<sup>^</sup> | ✅︎ | ✅︎ |
| `Moondream3ForCausalLM` | Moondream3 | T + I | `moondream/moondream3-preview` | | ✅︎ |
| `MusicFlamingoForConditionalGeneration` | MusicFlamingo | T + A | `nvidia/music-flamingo-2601-hf`, `nvidia/music-flamingo-think-2601-hf` | ✅︎ | ✅︎ |
| `NVLM_D_Model` | NVLM-D 1.0 | T + I<sup>+</sup> | `nvidia/NVLM-D-72B` 等 | | ✅︎ |
| `OpenCUAForConditionalGeneration` | OpenCUA-7B | T + I<sup>E+</sup> | `xlangai/OpenCUA-7B` | ✅︎ | ✅︎ |
| `OpenPanguVLForConditionalGeneration` | openpangu-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `FreedomIntelligence/openPangu-VL-7B` | ✅︎ | ✅︎ |
| `OpenVLAForActionPrediction` | OpenVLA | T + I | `openvla/openvla-7b` | | ✅︎ |
| `Ovis` | Ovis2, Ovis1.6 | T + I<sup>+</sup> | `AIDC-AI/Ovis2-1B`, `AIDC-AI/Ovis1.6-Llama3.2-3B` 等 | | ✅︎ |
| `Ovis2_5` | Ovis2.5 | T + I<sup>+</sup> + V | `AIDC-AI/Ovis2.5-9B` 等 | | |
| `Ovis2_6ForCausalLM` | Ovis2.6 | T + I<sup>+</sup> + V | `AIDC-AI/Ovis2.6-2B` 等 | | |
| `Ovis2_6_MoeForCausalLM` | Ovis2.6 | T + I<sup>+</sup> + V | `AIDC-AI/Ovis2.6-30B-A3B` 等 | | |
| `PaddleOCRVLForConditionalGeneration` | Paddle-OCR | T + I<sup>+</sup> | `PaddlePaddle/PaddleOCR-VL` 等 | | |
| `PaliGemmaForConditionalGeneration` | PaliGemma, PaliGemma 2 | T + I<sup>E</sup> | `google/paligemma-3b-pt-224`, `google/paligemma-3b-mix-224`, `google/paligemma2-3b-ft-docci-448` 等 | ✅︎ | ✅︎ |
| `Phi3VForCausalLM` | Phi-3-Vision, Phi-3.5-Vision | T + I<sup>E+</sup> | `microsoft/Phi-3-vision-128k-instruct`, `microsoft/Phi-3.5-vision-instruct` 等 | | ✅︎ |
| `Phi4MMForCausalLM` | Phi-4-multimodal | T + I<sup>+</sup> / T + A<sup>+</sup> / I<sup>+</sup> + A<sup>+</sup> | `microsoft/Phi-4-multimodal-instruct` 等 | ✅︎ | ✅︎ |
| `Phi4ForCausalLMV` | Phi-4-reasoning-vision | T + I<sup>+</sup> | `microsoft/Phi-4-reasoning-vision-15B` 等 | | ✅︎ |
| `PixtralForConditionalGeneration` | Ministral 3 (Mistral 格式), Mistral 3 (Mistral 格式), Mistral Large 3 (Mistral 格式), Pixtral (Mistral 格式) | T + I<sup>+</sup> | `mistralai/Ministral-3-3B-Instruct-2512`, `mistralai/Mistral-Small-3.1-24B-Instruct-2503`, `mistralai/Mistral-Large-3-675B-Instruct-2512` `mistralai/Pixtral-12B-2409` 等 | ✅︎ | ✅︎ |
| `QianfanOCRForConditionalGeneration` | QianfanOCR | T + I<sup>E+</sup> | `baidu/Qianfan-OCR` 等 | ✅︎ | ✅︎ |
| `QwenVLForConditionalGeneration`<sup>^</sup> | Qwen-VL | T + I<sup>E+</sup> | `Qwen/Qwen-VL`, `Qwen/Qwen-VL-Chat` 等 | ✅︎ | ✅︎ |
| `Qwen2AudioForConditionalGeneration` | Qwen2-Audio | T + A<sup>+</sup> | `Qwen/Qwen2-Audio-7B-Instruct` | | ✅︎ |
| `Qwen2VLForConditionalGeneration` <sup>Q</sup> | QVQ, Qwen2-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/QVQ-72B-Preview`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2-VL-72B-Instruct` 等 | ✅︎ | ✅︎ |
| `Qwen2_5_VLForConditionalGeneration` <sup>Q</sup> | Qwen2.5-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct` 等 | ✅︎ | ✅︎ |
| `Qwen2_5OmniThinkerForConditionalGeneration` | Qwen2.5-Omni | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>+</sup> | `Qwen/Qwen2.5-Omni-3B`, `Qwen/Qwen2.5-Omni-7B` | ✅︎ | ✅︎ |
| `Qwen3_5ForConditionalGeneration` | Qwen3.5 | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3.5-9B-Instruct` 等 | ✅︎ | ✅︎ |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.5-MOE | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3.5-35B-A3B-Instruct` 等 | ✅︎ | ✅︎ |
| `Qwen3VLForConditionalGeneration` <sup>Q</sup> | Qwen3-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-4B-Instruct` 等 | ✅︎ | ✅︎ |
| `Qwen3VLMoeForConditionalGeneration` <sup>Q</sup> | Qwen3-VL-MOE | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-30B-A3B-Instruct` 等 | ✅︎ | ✅︎ |
| `Qwen3OmniMoeThinkerForConditionalGeneration` | Qwen3-Omni | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>+</sup> | `Qwen/Qwen3-Omni-30B-A3B-Instruct`, `Qwen/Qwen3-Omni-30B-A3B-Thinking` | ✅︎ | ✅︎ |
| `Qwen3ASRForConditionalGeneration` | Qwen3-ASR | T + A<sup>+</sup> | `Qwen/Qwen3-ASR-1.7B` | ✅︎ | ✅︎ |
| `RForConditionalGeneration` | R-VL-4B | T + I<sup>E+</sup> | `YannQi/R-4B` | | ✅︎ |
| `SkyworkR1VChatModel` | Skywork-R1V-38B | T + I | `Skywork/Skywork-R1V-38B` | | ✅︎ |
| `SmolVLMForConditionalGeneration` | SmolVLM2 | T + I | `SmolVLM2-2.2B-Instruct` | ✅︎ | |
| `Step3VLForConditionalGeneration` | Step3-VL | T + I<sup>+</sup> | `stepfun-ai/step3` | | ✅︎ |
| `StepVLForConditionalGeneration` | Step3-VL-10B | T + I<sup>+</sup> | `stepfun-ai/Step3-VL-10B` | | ✅︎ |
| `TarsierForConditionalGeneration` | Tarsier | T + I<sup>E+</sup> | `omni-search/Tarsier-7b`, `omni-search/Tarsier-34b` | | ✅︎ |
| `Tarsier2ForConditionalGeneration`<sup>^</sup> | Tarsier2 | T + I<sup>E+</sup> + V<sup>E+</sup> | `omni-research/Tarsier2-Recap-7b`, `omni-research/Tarsier2-7b-0115` | | ✅︎ |
| `UltravoxModel` | Ultravox | T + A<sup>E+</sup> | `fixie-ai/ultravox-v0_5-llama-3_2-1b` | ✅︎ | ✅︎ |

某些模型仅通过 [Transformers 建模后端](#transformers) 支持。下表的目的在于确认我们以此方式正式支持的模型。日志会显示正在使用 Transformers 建模后端，并且您不会看到任何表示这是回退行为的警告。这意味着，如果您在使用下面列出的任何模型时遇到问题，请[提交 issue](https://github.com/vllm-project/vllm/issues/new/choose)，我们会尽力修复！

| 架构 | 模型 | 输入 | 示例 HF 模型 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | --------------------------- | --------------------------------------- |
| `Emu3ForConditionalGeneration` | Emu3 | T + I | `BAAI/Emu3-Chat-hf` | ✅︎ | ✅︎ |

<sup>^</sup> 您需要通过 `--hf-overrides` 将架构名称设置为与 vLLM 中的架构名称匹配。</br>
<sup>E</sup> 对于此模态，可以输入预计算的嵌入。</br>
<sup>+</sup> 对于此模态，每个文本提示可以输入多个项目。
<sup>*</sup> 只有特定变体的模型支持此模态（请参见下面的附注）。</br>
<sup>Q</sup> `Qwen*-VL` 官方使用 `qwen_vl_utils` 进行图像预处理，而 vLLM 使用 Transformers 的 `video_processing_qwen*`，这会导致与官方 Hugging Face 仓库示例略有不同的结果。

!!! note
    `Gemma3nForConditionalGeneration` 仅在 V1 上受支持，因为它使用了共享 KV 缓存，并且依赖于 `timm>=1.0.17` 来使用其 MobileNet-v5 视觉骨干网络。

    性能尚未完全优化，主要原因是：

    - 音频和视觉 MM 编码器都使用 `transformers.AutoModel` 实现。
    - 没有 PLE 缓存或内存不足交换支持，如 [Google 的博客](https://developers.googleblog.com/en/introducing-gemma-3n/)所述。这些功能可能对 vLLM 来说过于模型特定，特别是交换功能可能更适合受约束的设置。

!!! note
    对于 `Gemma4ForConditionalGeneration`：
    - 音频输入仅由 `gemma-4-E2B` 和 `gemma-4-E4B` 变体支持。
    - 该模型不直接接收视频。但是，vLLM 的 Gemma 4 实现通过内部处理视频来支持视频输入。用户可以直接在消息结构中向 vLLM 发送视频，视频将被转换为文本和图像帧后再传递给模型。
    - Gemma 4 辅助检查点用于推测解码，使用的是 vLLM 的 Gemma 4 MTP 路径，而不是通用的草稿模型推测解码。请参见 [Gemma 4 辅助模型 MTP 示例](../features/speculative_decoding/mtp.md#gemma-4-assistant-models)。

!!! note
    对于 `InternVLChatModel`，目前只有使用 Qwen2.5 文本骨干网络的 InternVL2.5（`OpenGVLab/InternVL2.5-1B` 等）、InternVL3 和 InternVL3.5 支持视频输入。

!!! note
    要使用 `allenai/MolmoWeb-4B` 或 `allenai/MolmoWeb-8B`，请使用 Molmo2 架构提供检查点并禁用多模态前缀注意力：
    `--hf-overrides '{"architectures": ["Molmo2ForConditionalGeneration"], "is_mm_prefix_lm": false}'`。

!!! note
    `Moondream3ForCausalLM` 使用特定于任务的提示模板进行 `query` 和 `caption`。原生的 `detect` 和 `point` 技能需要自定义坐标解码，并且不在此 vLLM 实现中暴露。请参见 [Moondream3 提示配方](../features/multimodal_inputs.md#moondream3-prompt-recipes)。

!!! note
    要使用 `TIGER-Lab/Mantis-8B-siglip-llama3`，您必须在运行 vLLM 时传递 `--hf_overrides '{"architectures": ["MantisForConditionalGeneration"]}'`。

!!! note
    官方的 `openbmb/MiniCPM-V-2` 目前无法正常工作，因此我们需要暂时使用一个 fork (`HwwwH/MiniCPM-V-2`)。
    更多详情请参见：<https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630>

#### 转录

专门为自动语音识别训练的语音转文本模型。

| 架构 | 模型 | 示例 HF 模型 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `CohereAsrForConditionalGeneration` | Cohere-Transcribe | `CohereLabs/cohere-transcribe-03-2026` | | |
| `FireRedASR2ForConditionalGeneration` | FireRedASR2 | `allendou/FireRedASR2-LLM-vllm` 等 | | |
| `FireRedLIDForConditionalGeneration` | FireRedLID | `PatchyTisa/FireRedLID-vllm` 等 | | |
| `FunASRForConditionalGeneration` | FunASR | `allendou/Fun-ASR-Nano-2512-vllm` 等 | | |
| `Gemma3nForConditionalGeneration` | Gemma3n | `google/gemma-3n-E2B-it`, `google/gemma-3n-E4B-it` 等 | | |
| `GlmAsrForConditionalGeneration` | GLM-ASR | `zai-org/GLM-ASR-Nano-2512` | ✅︎ | ✅︎ |
| `GraniteSpeechForConditionalGeneration` | Granite Speech | `ibm-granite/granite-4.0-1b-speech`, `ibm-granite/granite-speech-3.3-2b` 等 | ✅︎ | ✅︎ |
| `Qwen3ASRForConditionalGeneration` | Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B` 等 | ✅︎ | ✅︎ |
| `Qwen3OmniMoeThinkerForConditionalGeneration` | Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct` 等 | | ✅︎ |
| `VoxtralForConditionalGeneration` | Voxtral (Mistral 格式) | `mistralai/Voxtral-Mini-3B-2507`, `mistralai/Voxtral-Small-24B-2507` 等 | ✅︎ | ✅︎ |
| `WhisperForConditionalGeneration` | Whisper | `openai/whisper-small`, `openai/whisper-large-v3-turbo` 等 | | |

!!! note
    `VoxtralForConditionalGeneration` 需要安装 `mistral-common[audio]`。

#### 实时转录

通过 [`/v1/realtime`](../serving/online_serving/speech_to_text.md#realtime-api) WebSocket 端点支持流式转录的语音模型。

| 架构 | 模型 | 示例 HF 模型 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `VoxtralRealtimeGeneration` | Voxtral Realtime | `mistralai/Voxtral-Mini-4B-Realtime-2602` | | |
| `Qwen3ASRRealtimeGeneration` | Qwen3-ASR Realtime | `Qwen/Qwen3-ASR-0.6B` | | |

!!! note
    `VoxtralRealtimeGeneration` 需要安装 `mistral-common[audio]`，并且必须使用 `--tokenizer-mode mistral` 提供服务。

    `Qwen3ASRRealtimeGeneration` 不会从 `config.json` 自动检测。
    您必须在提供服务时传递 `--hf-overrides '{"architectures":["Qwen3ASRRealtimeGeneration"]}'`。

## 池化模型

有关如何使用池化模型的更多信息，请参见[此页面](pooling_models/README.md)。

!!! important
    由于某些模型架构同时支持生成式和池化任务，
    您应该显式指定 `--runner pooling` 以确保模型在池化模式下运行，而不是生成模式。

有关特定池化任务支持的模型的更多信息，请参见下面的链接。

- [分类用法](pooling_models/classify.md)
- [嵌入用法](pooling_models/embed.md)
- [奖励用法](pooling_models/reward.md)
- [Token 分类用法](pooling_models/token_classify.md)
- [Token 嵌入用法](pooling_models/token_embed.md)
- [评分用法](pooling_models/scoring.md)
- [特定模型示例](pooling_models/specific_models.md)

## 模型支持策略

在 vLLM，我们致力于促进第三方模型在我们生态系统中的集成和支持。我们的方法旨在平衡对鲁棒性的需求与支持广泛模型的实际限制。以下是我们管理第三方模型支持的方式：

1. **社区驱动的支持**：我们鼓励社区贡献以添加新模型。当用户请求支持新模型时，我们欢迎来自社区的拉取请求 (PR)。这些贡献主要根据其生成输出的合理性进行评估，而不是与 transformers 等现有实现的严格一致性。**征集贡献：** 来自模型供应商的直接 PR 受到高度赞赏！

2. **尽力而为的一致性**：虽然我们旨在保持 vLLM 中实现的模型与 transformers 等其他框架之间的一致性，但完全对齐并不总是可行的。加速技术和低精度计算的使用等因素可能会导致差异。我们的承诺是确保实现的模型功能正常并产生合理的结果。

    !!! tip
        在比较 Hugging Face Transformers 的 `model.generate` 输出与 vLLM 的 `llm.generate` 输出时，请注意前者会读取模型的生成配置文件（即 [generation_config.json](https://github.com/huggingface/transformers/blob/19dabe96362803fb0a9ae7073d03533966598b17/src/transformers/generation/utils.py#L1945)）并应用默认的生成参数，而后者仅使用传递给函数的参数。在比较输出时，请确保所有采样参数相同。

3. **问题解决和模型更新**：鼓励用户报告他们在第三方模型中遇到的任何错误或问题。建议的修复应通过 PR 提交，并附有对问题的清晰解释以及所提出解决方案的理由。如果一个模型的修复影响了另一个模型，我们依靠社区来指出并解决这些跨模型依赖关系。注意：对于错误修复 PR，告知原始作者以征求其反馈是良好的礼仪。

4. **监控和更新**：对特定模型感兴趣的用户应监控这些模型的提交历史（例如，通过跟踪 main/vllm/model_executor/models 目录中的更改）。这种主动方法有助于用户及时了解可能影响他们所用模型的更新和更改。

5. **选择性关注**：我们的资源主要投向具有显著用户兴趣和影响力的模型。使用较少的模型可能获得较少的关注，我们依靠社区在其维护和改进方面发挥更积极的作用。

通过这种方式，vLLM 营造了一个协作环境，核心开发团队和更广泛的社区共同为我们生态系统中支持的第三方模型的鲁棒性和多样性做出贡献。

请注意，作为一个推理引擎，vLLM 不会引入新模型。因此，vLLM 支持的所有模型在这方面都是第三方模型。

我们对模型有以下级别的测试：

1. **严格一致性**：我们在贪心解码下将模型的输出与 HuggingFace Transformers 库中模型的输出进行比较。这是最严格的测试。请参考通过此测试的模型的[模型测试](https://github.com/vllm-project/vllm/blob/main/tests/models)。
2. **输出合理性**：我们通过测量输出的困惑度和检查是否存在明显错误，来检查模型的输出是否合理且连贯。这是一个不那么严格的测试。
3. **运行时功能性**：我们检查模型是否可以无错误地加载和运行。这是最不严格的测试。请参考通过此测试的模型的[功能性测试](../../tests)和[示例](../../examples)。
4. **社区反馈**：我们依靠社区提供关于模型的反馈。如果某个模型损坏或无法按预期工作，我们鼓励用户提交 issue 来报告，或提交拉取请求来修复。其余模型属于此类别。
