# 异步强化学习

## 概述

在标准的 RL 训练循环中，生成和训练是按顺序进行的：策略生成 rollout，然后训练基于这些 rollout 运行，如此循环往复。在生成期间，训练加速器处于空闲状态，反之亦然。

**一次性流水线**方法将生成和训练阶段分离为两个并行的协程，使得模型能够同时生成新样本并在之前生成的数据上进行训练。这可以带来更好的 GPU 利用率和更高的训练吞吐量。

然而，这种重叠引入了一个复杂问题：权重必须在推理引擎运行过程中进行更新，而此时可能仍有请求正在进行中。

## 暂停和恢复 API

为了在推理引擎运行时安全地更新权重，vLLM 提供了 `pause_generation` 和 `resume_generation` 方法。这些方法让训练器能够协调出一个干净的权重同步窗口，而不会丢失进行中的工作。

### pause_generation

```python
await engine.pause_generation(mode="keep", clear_cache=True)
```

`mode` 参数控制如何处理进行中的请求：

| 模式 | 行为 |
| ---- | -------- |
| `"abort"` | 立即中止所有进行中的请求并返回部分结果（默认） |
| `"wait"` | 等待所有进行中的请求完成后暂停 |
| `"keep"` | 将请求冻结在队列中；调用 `resume_generation` 时恢复 |

`clear_cache` 参数控制暂停后是否清除 KV 缓存和前缀缓存。

### resume_generation

```python
await engine.resume_generation()
```

暂停后恢复调度器。任何使用 `mode="keep"` 冻结的请求将继续生成。

### HTTP 端点

使用 vLLM HTTP 服务器时，相同功能可通过以下方式使用：

- `POST /pause?mode=keep` - 暂停生成
- `POST /resume` - 恢复生成

!!! note "数据并行"
    使用 vLLM **内部负载均衡器**（即 `data_parallel_backend="ray"`）的数据并行时，暂停和恢复会在所有 DP  ranks 上自动处理——调用一次即可。使用**外部负载均衡器**（即代理后方的多个独立 vLLM 实例）时，您必须在权重更新前后**逐个**向每个引擎实例发送暂停和恢复请求。

## 典型异步 RL 流程

一个带权重同步的典型异步 RL 循环如下所示：

1. 开始从当前策略生成 rollout
2. 一旦训练器有新权重需要更新，使用 `mode="keep"` 暂停生成
3. 将更新后的权重从训练器同步到推理引擎（参见[权重传输](weight_transfer/README.md)）
4. 恢复生成——进行中的请求将继续使用新权重
5. 重复

关键点在于，使用 `mode="keep"` 暂停的请求将在暂停前使用**旧**权重生成 token，恢复后使用**新**权重生成 token。`clear_cache` 参数控制暂停期间 KV 缓存是否失效。当 `clear_cache=True` 时，先前缓存的键值条目将被丢弃，因此恢复后生成的所有 token 将完全使用新权重计算。当 `clear_cache=False` 时，现有的 KV 缓存条目将被保留，这意味着上下文中的某些 token 可能仍然反映旧权重（过期的 KV 缓存）。

## 示例

[异步 RLHF 示例](../../examples/rl/rlhf_async_new_apis.py)演示了这种模式，使用了 `vllm.AsyncLLMEngine`、NCCL 权重传输以及带验证的飞行中暂停/恢复。
