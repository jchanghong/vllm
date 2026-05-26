# 投机解码 (Speculative Decoding)

本文档介绍如何使用 vLLM 的 [投机解码](https://arxiv.org/pdf/2302.01318) 功能，在中低 QPS（每秒查询数）、内存受限的工作负载下降低 token 间延迟。

要训练自定义草稿模型以优化投机解码，请参阅 [vllm-project/speculators](speculators.md)，了解如何无缝训练并与 vLLM 集成。

## vLLM 投机方法

vLLM 支持多种投机解码方法。基于模型的方法（如 EAGLE、MTP、草稿模型、PARD 和 MLP）提供最佳的延迟降低效果，而更简单的方法（如 n-gram 和后缀解码）则可在不增加高峰期工作负载的情况下提供适度的加速。

- [EAGLE](eagle.md)
- [多 token 预测 (MTP)](mtp.md)
- [草稿模型 (Draft Model)](draft_model.md)
- [并行草稿模型 (PARD)](parallel_draft_model.md)
- [多层感知机 (MLP)](mlp.md)
- [N-Gram](n_gram.md)
- [后缀解码 (Suffix Decoding)](suffix.md)
- [隐藏状态提取](extract_hidden_states.md)
- [自定义提议后端（实验性）](#custom-proposer-backend-experimental)

## 方法速览

使用以下定性表格作为方法选择的起点。实际收益取决于您的模型家族、流量模式、硬件和采样设置。

| 方法 | 低 QPS（面向延迟） | 高 QPS（面向吞吐量） | 说明 |
| --- | --- | --- | --- |
| EAGLE | 高收益 | 中到高收益 | 强大的通用型基于模型的方法。 |
| MTP | 高收益 | 中到高收益 | 当目标模型原生支持 MTP 时效果最佳。 |
| 草稿模型 | 高收益 | 中等收益 | 需要单独的草稿模型。 |
| 并行草稿模型 | 高收益 | 中到高收益 | 低草稿模型延迟。 |
| MLP 推测器 | 中到高收益 | 中等收益 | 当兼容的 MLP 推测器可用时效果良好。 |
| N-gram | 低到中等收益 | 中等收益 | 轻量级，易于启用。 |
| 后缀解码 | 低到中等收益 | 中等收益 | 无需额外草稿模型；动态推测深度。 |
| 自定义提议器 | 视情况而定 | 视情况而定 | 自带自定义提议器类（实验性）。 |

要在您自己的环境中获得可重现的测量结果，请使用
[`examples/features/speculative_decoding/spec_decode_offline.py`](../../../examples/features/speculative_decoding/spec_decode_offline.py)
或 [基准测试 CLI 指南](../../benchmarking/cli.md)。

## 自定义提议后端（实验性）

您可以通过将方法设置为 `custom_class` 并提供类的完整模块路径，来插入自己的自定义提议器类以进行投机解码。
您的自定义类在实例化时必须接受一个 `VllmConfig` 并实现一个 `propose` 方法。

**配置示例：**

- `speculative_config.method = "custom_class"`
- `speculative_config.model = "your_module.YourCustomProposerClass"`

## `--speculative-config` 模式

使用 `--speculative-config` 将投机解码设置作为 JSON 对象传递给 CLI：

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "draft_model",
    "model": "<draft-model>",
    "num_speculative_tokens": 5
  }'
```

相同的键也可通过 Python 的 `LLM(..., speculative_config={...})` 接受。
下表列出了该 JSON 对象中常用的用户面向键；它们并非详尽的模式参考。
更多详细信息，请参阅生成的 [引擎参数参考](../../configuration/engine_args.md)
以及 [vllm.config.SpeculativeConfig][] 的 API 文档。

### 通用键

以下键在投机解码配置中常用，尽管某些键仅适用于基于模型的方法，如 `draft_model`、`mtp`、`eagle3` 和 `dflash`。

| 键 | 类型 | 默认值 | 允许值/含义 |
| --- | --- | --- | --- |
| `method` | `string` | `None` | 推测方法。常用值包括 `draft_model`、`ngram`、`suffix`、`mtp`、`eagle3` 和 `dflash`。如果省略，vLLM 会在可能的情况下从提供的配置中推断方法。 |
| `model` | `string` | `None` | 草稿模型、EAGLE 头或辅助模型标识符。对于 `ngram`、`ngram_gpu`、`suffix` 和 `mtp`，通常可以省略此项。 |
| `num_speculative_tokens` | `integer > 0` | `None` | 每步提议的推测 token 数量。对于无法从模型元数据推断此值的方法是必需的。 |
| `draft_tensor_parallel_size` | `integer >= 1` | `None` | 草稿模型的 tensor parallel 大小。 |
| `max_model_len` | `integer >= 1` | `None` | 草稿模型的最大上下文长度。 |
| `parallel_drafting` | `boolean` | `false` | 启用并行草稿 token 生成。仅与 EAGLE 和草稿模型方法兼容。 |
| `rejection_sample_method` | `string` | `strict` | `strict`、`probabilistic` 或 `synthetic`。 |
| `synthetic_acceptance_rate` | `float` | `None` | 当 `rejection_sample_method` 为 `synthetic` 时目标平均接受率。有效范围为 `[0, 1]`。 |

!!! note
    Gemma 4 辅助检查点作为 Gemma 4 MTP 推测器处理，而非通用草稿模型。如 [MTP 指南](mtp.md#gemma-4-assistant-models) 所示，在 `model` 中使用辅助检查点时请使用 `"method": "mtp"`。

    如果启动日志中对 Gemma 4 辅助检查点显示 `SpeculativeConfig(method='draft_model', ...)`，则说明已安装的 vLLM 版本不包含对该路径的 Gemma 4 MTP 支持。请升级到包含 Gemma 4 MTP 支持的版本，而不是强制将辅助检查点通过通用草稿模型投机解码运行。

### 方法特定键

#### N-gram

| 键 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `prompt_lookup_max` | `integer >= 1` | 如果两个查找边界都省略则为 `5`；否则省略时镜像 `prompt_lookup_min` | 最大 n-gram 窗口大小。 |
| `prompt_lookup_min` | `integer >= 1` | 如果两个查找边界都省略则为 `5`；否则省略时镜像 `prompt_lookup_max` | 最小 n-gram 窗口大小。 |

示例：

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "ngram",
    "num_speculative_tokens": 4,
    "prompt_lookup_min": 2,
    "prompt_lookup_max": 5
  }'
```

#### 后缀解码

| 键 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `suffix_decoding_max_tree_depth` | `integer` | `24` | 最大前缀匹配和推测树组合深度。 |
| `suffix_decoding_max_cached_requests` | `integer` | `10000` | 全局后缀树中缓存的最大请求数。设为 `0` 可禁用全局缓存。 |
| `suffix_decoding_max_spec_factor` | `float` | `1.0` | 将推测长度限制为前缀匹配长度的倍数。 |
| `suffix_decoding_min_token_prob` | `float` | `0.1` | 推测 token 所需的最小估计 token 概率。 |

示例：

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "suffix",
    "num_speculative_tokens": 8,
    "suffix_decoding_max_tree_depth": 24,
    "suffix_decoding_max_cached_requests": 10000,
    "suffix_decoding_max_spec_factor": 1.0,
    "suffix_decoding_min_token_prob": 0.1
  }'
```

### 说明

- `--speculative-config` 在 CLI 上期望一个 JSON 对象。在 YAML 配置文件中，使用嵌套映射而不是转义的 JSON 字符串。
- `tensor_parallel_size` 不是 `speculative_config` 中的有效键。请改用 `draft_tensor_parallel_size`。
- `temperature` 和 `top_p` 等键是采样参数，而非 `--speculative-config` 字段。
- 内部字段如 `target_model_config`、`draft_model_config`、`target_parallel_config`、`draft_parallel_config` 和 `draft_load_config` 由 vLLM 填充，不旨在由用户设置。

## 投机解码的无损保证

在 vLLM 中，投机解码旨在提高推理效率的同时保持准确性。本节讨论投机解码的无损保证，将保证分为三个关键领域：

1. **理论无损**
   \- 投机解码采样在硬件数值精度限制内理论上是无损的。浮点误差可能导致输出分布的细微变化，如
   [使用投机采样加速大语言模型解码](https://arxiv.org/pdf/2302.01318) 中所述。

2. **算法无损**
   \- vLLM 的投机解码实现在算法上经过验证是无损的。关键的验证测试包括：

    > - **拒绝采样器收敛性**：确保 vLLM 拒绝采样器的样本与目标分布一致。[查看测试代码](https://github.com/vllm-project/vllm/blob/47b65a550866c7ffbd076ecb74106714838ce7da/tests/samplers/test_rejection_sampler.py#L252)
    > - **贪婪采样等价性**：确认使用投机解码的贪婪采样与不使用投机解码的贪婪采样相匹配。这验证了 vLLM 的投机解码框架在与 vLLM 前向传播和 vLLM 拒绝采样器集成时提供无损保证。[tests/spec_decode/e2e](/tests/v1/spec_decode) 中的几乎所有测试都使用[此断言实现](https://github.com/vllm-project/vllm/blob/b67ae00cdbbe1a58ffc8ff170f0c8d79044a684a/tests/spec_decode/e2e/conftest.py#L291)验证此属性。

3. **vLLM Logprob 稳定性**
   \- vLLM 目前不保证 token log 概率（logprobs）的稳定性。这可能导致同一请求在不同运行中产生不同输出。更多详情，请参阅 [常见问题](../../usage/faq.md) 中的 *vLLM 中同一提示的输出是否会因运行而异？* 一节。

虽然 vLLM 努力确保投机解码的无损性，但使用和不使用投机解码时生成输出的差异可能由以下因素引起：

- **浮点精度**：硬件数值精度的差异可能导致输出分布的细微差异。
- **批量大小和数值稳定性**：批量大小的变化可能导致 logprobs 和输出概率的变化，这可能是由于批处理操作中的非确定性行为或数值不稳定性所致。

有关缓解策略，请参阅 [常见问题](../../usage/faq.md) 中的 *vLLM 中同一提示的输出是否会因运行而异？* 条目。

## 已知功能不兼容

1. 在 `vllm<=0.15.0` 中，流水线并行与投机解码不兼容。
2. 在 `vllm<=0.10.0` 中，不支持使用草稿模型进行投机解码。

## vLLM 贡献者资源

- [[vLLM Office Hours #40] 推测器简介](https://www.youtube.com/watch?v=2ISAr_JVGLs)
- [vLLM 投机解码黑客指南](https://www.youtube.com/watch?v=9wNAgpX6z_4)
- [vLLM 中的超前调度是什么？](https://docs.google.com/document/d/1Z9TvqzzBPnh5WHcRwjvK2UEeFeq5zMZb5mFE8jR0HCs/edit#heading=h.1fjfb0donq5a)
- [批量扩展信息](https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit#heading=h.kk7dq05lc6q8)
- [动态投机解码](https://github.com/vllm-project/vllm/issues/4565)
