# 离线推理

离线推理可以在你自己的代码中使用 vLLM 的 [`LLM`][vllm.LLM] 类实现。

## 模型类型

vLLM 模型可分为两类：

- **[生成式模型](../models/supported_models.md)**——生成文本补全或聊天响应的模型（例如 LLaMA、Qwen、DeepSeek）。对此类模型使用 `LLM.generate()` 和 `LLM.chat()`。

- **[池化模型](../models/pooling_models/README.md)**——这些模型不生成内容。它们主要用于分类和检索任务，例如 bge-m3 和 Qwen3 Reranker。

## 生成式 API

有关生成式模型的更多详细信息，请参考[此页面](../models/supported_models.md)。

- `LLM.generate`——为给定的输入提示生成补全。
- `LLM.chat`——为聊天对话生成响应。

## 异步队列 API

- `LLM.enqueue`——将提示加入生成队列，无需等待完成。
- `LLM.enqueue_chat`——将聊天对话加入生成队列，无需等待。
- `LLM.wait_for_completion`——等待所有已入队的请求完成并返回结果。

## 池化 API

有关池化模型的更多详细信息，请参考[此页面](../models/pooling_models/README.md)。

- `LLM.classify`——仅适用于[分类模型](../models/pooling_models/classify.md)。
- `LLM.embed`——仅适用于[嵌入模型](../models/pooling_models/embed.md)。
- `LLM.score`——适用于[评分模型](../models/pooling_models/scoring.md)（交叉编码器、双编码器、后期交互）。
- `LLM.encode`——适用于所有[池化模型](../models/pooling_models/README.md)。

## 性能分析 API

有关性能分析的更多详细信息，请参考[此页面](../contributing/profiling.md)。

- `LLM.start_profile`——使用可选的自定义跟踪前缀开始性能分析。
- `LLM.stop_profile`——停止正在进行的性能分析会话。

## 休眠模式 API

有关休眠模式的更多详细信息，请参考[此页面](../features/sleep_mode.md)。

- `LLM.sleep`——将引擎置于休眠模式。
- `LLM.wake_up`——将引擎从休眠模式唤醒。

## 缓存管理 API

- `LLM.reset_mm_cache`——重置多模态缓存。
- `LLM.reset_prefix_cache`——重置前缀缓存。

## 指标 API

有关指标的更多详细信息，请参考[此页面](../design/metrics.md)。

- `LLM.get_metrics`——返回来自 Prometheus 的聚合指标快照。

## 权重传输 API（RL 训练）

有关权重传输的更多详细信息，请参考[此页面](../training/weight_transfer/README.md)。

- `LLM.init_weight_transfer_engine`——初始化用于 RL 训练的权重传输引擎。
- `LLM.start_weight_update`——开始新的权重更新周期。
- `LLM.update_weights`——更新模型权重。
- `LLM.finish_weight_update`——完成当前权重更新周期。

## 其他 API

- `LLM.collective_rpc`——在所有工作节点上集体执行方法或可调用对象。
- `LLM.apply_model`——在每个工作节点内部直接对模型应用函数。

## API 参考

[离线推理](../api/README.md#offline-inference)

## Ray Data LLM API

Ray Data LLM 是另一种离线推理 API，使用 vLLM 作为底层引擎。
该 API 增加了多项内置功能，简化了大规模、GPU 高效的推理：

- 流式执行处理超出聚合集群内存的数据集。
- 自动分片、负载均衡和自动扩展可在 Ray 集群上分配工作，并具有内置的容错能力。
- 持续批处理使 vLLM 副本保持饱和状态，最大化 GPU 利用率。
- 透明支持张量和流水线并行，实现高效的多 GPU 推理。
- 支持读写大多数流行文件格式和云对象存储。
- 无需更改代码即可扩展工作负载。

??? code

    ```python
    import ray  # 需要 ray>=2.44.1
    from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

    config = vLLMEngineProcessorConfig(model_source="unsloth/Llama-3.2-1B-Instruct")
    processor = build_llm_processor(
        config,
        preprocess=lambda row: {
            "messages": [
                {"role": "system", "content": "You are a bot that completes unfinished haikus."},
                {"role": "user", "content": row["item"]},
            ],
            "sampling_params": {"temperature": 0.3, "max_tokens": 250},
        },
        postprocess=lambda row: {"answer": row["generated_text"]},
    )

    ds = ray.data.from_items(["An old silent pond..."])
    ds = processor(ds)
    ds.write_parquet("local:///tmp/data/")
    ```

有关 Ray Data LLM API 的更多信息，请参见 [Ray Data LLM 文档](https://docs.ray.io/en/latest/data/working-with-llms.html)。
