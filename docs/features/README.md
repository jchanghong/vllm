# 功能特性

## 兼容性矩阵

下表展示了互斥的功能特性以及在某些硬件上的支持情况。

使用的符号含义如下：

- ✅ = 完全兼容
- 🟠 = 部分兼容
- ❌ = 不兼容
- ❔ = 未知或待定

!!! note
    点击带有链接的 ❌ 或 🟠 符号，可查看不支持的功能/硬件组合的跟踪问题。

### 功能 x 功能

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| 功能 | [CP](../configuration/optimization.md#chunked-prefill) | [APC](automatic_prefix_caching.md) | [LoRA](lora.md) | [SD](speculative_decoding/README.md) | CUDA graph | [pooling](../models/pooling_models/README.md) | <abbr title="编码器-解码器模型">enc-dec</abbr> | <abbr title="Logprobs">logP</abbr> | <abbr title="Prompt Logprobs">prmpt logP</abbr> | <abbr title="异步输出处理">async output</abbr> | multi-step | <abbr title="多模态输入">mm</abbr> | best-of | beam-search | [prompt-embeds](prompt_embeds.md) |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| [CP](../configuration/optimization.md#chunked-prefill) | ✅ | | | | | | | | | | | | | | |
| [APC](automatic_prefix_caching.md) | ✅ | ✅ | | | | | | | | | | | | | |
| [LoRA](lora.md) | ✅ | ✅ | ✅ | | | | | | | | | | | | |
| [SD](speculative_decoding/README.md) | ✅ | ✅ | ❌ | ✅ | | | | | | | | | | | |
| CUDA graph | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | |
| [pooling](../models/pooling_models/README.md) | 🟠\* | 🟠\* | ✅ | ❌ | ✅ | ✅ | | | | | | | | | |
| <abbr title="编码器-解码器模型">enc-dec</abbr> | ❌ | [❌](https://github.com/vllm-project/vllm/issues/7366) | ❌ | [❌](https://github.com/vllm-project/vllm/issues/7366) | ✅ | ✅ | ✅ | | | | | | | | |
| <abbr title="Logprobs">logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | | | | | | | |
| <abbr title="Prompt Logprobs">prmpt logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | | | | | | |
| <abbr title="异步输出处理">async output</abbr> | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | | | | | |
| multi-step | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | | | | |
| [mm](multimodal_inputs.md) | ✅ | ✅ | [🟠](https://github.com/vllm-project/vllm/pull/4194)<sup>^</sup> | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | | | |
| best-of | ✅ | ✅ | ✅ | [❌](https://github.com/vllm-project/vllm/issues/6137) | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | [❌](https://github.com/vllm-project/vllm/issues/7968) | ✅ | ✅ | | |
| beam-search | ✅ | ✅ | ✅ | [❌](https://github.com/vllm-project/vllm/issues/6137) | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | [❌](https://github.com/vllm-project/vllm/issues/7968) | ❔ | ✅ | ✅ | |
| [prompt-embeds](prompt_embeds.md) | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❔ | ❔ | ✅ | ❔ | ❔ | ✅ |

\* 分块预填充和前缀缓存仅适用于具有因果注意力的最后 token 或全部池化。
<sup>^</sup> LoRA 仅适用于多模态模型的语言主干部分。

### 功能 x 硬件

| 功能 | Volta | Turing | Ampere | Ada | Hopper | CPU | AMD | Intel GPU |
| ------- | ----- | ------ | ------ | --- | ------ | --- | --- | --------- |
| [CP](../configuration/optimization.md#chunked-prefill) | [❌](https://github.com/vllm-project/vllm/issues/2729) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [APC](automatic_prefix_caching.md) | [❌](https://github.com/vllm-project/vllm/issues/3687) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [LoRA](lora.md) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [SD](speculative_decoding/README.md) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| CUDA graph | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | [❌](https://github.com/vllm-project/vllm/issues/26970) |
| [pooling](../models/pooling_models/README.md) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| <abbr title="编码器-解码器模型">enc-dec</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| [mm](multimodal_inputs.md) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [prompt-embeds](prompt_embeds.md) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ |
| <abbr title="Logprobs">logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| <abbr title="Prompt Logprobs">prmpt logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| <abbr title="异步输出处理">async output</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| multi-step | ✅ | ✅ | ✅ | ✅ | ✅ | [❌](https://github.com/vllm-project/vllm/issues/8477) | ✅ | ✅ |
| best-of | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| beam-search | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

!!! note
    有关 Google TPU 上功能支持的信息，请参阅 [TPU-推理推荐模型和功能](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/) 文档。
