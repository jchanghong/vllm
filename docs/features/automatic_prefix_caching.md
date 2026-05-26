# 自动前缀缓存

## 简介

自动前缀缓存（简称 APC）会缓存已有查询的 KV 缓存，这样如果新查询与某个已有查询共享相同的前缀，就可以直接复用该 KV 缓存，从而跳过共享部分的计算。

!!! note
    关于 vLLM 如何实现 APC 的技术细节，请参见[此处](../design/prefix_caching.md)。

## 在 vLLM 中启用 APC

在 vLLM 引擎中设置 `enable_prefix_caching=True` 即可启用 APC。示例如下：

[examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py](../../examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py)

## 示例工作负载

我们描述两种示例工作负载，APC 可以在其中提供巨大的性能优势：

- **长文档查询**，用户反复查询同一份长文档（例如软件手册或年度报告），但每次查询不同的问题。在这种情况下，APC 允许 vLLM **只处理一次** 这份长文档，所有后续请求可以通过复用其 KV 缓存来避免重新计算。这使得 vLLM 能够以更高的吞吐量和更低的延迟来服务后续请求。
- **多轮对话**，用户在同一个聊天会话中多次与应用程序聊天。在这种情况下，APC 允许 vLLM 在所有后续对话轮次中复用聊天历史的处理结果，而不是一遍又一遍地处理整个聊天历史，从而使 vLLM 能够以更高的吞吐量和更低的延迟来服务后续请求。

## 限制

APC 通常不会降低 vLLM 的性能。话虽如此，APC 仅减少处理查询的时间（预填充阶段），并不会减少生成新 token 的时间（解码阶段）。因此，当 vLLM 大部分时间用于生成查询答案时（例如答案长度较长时），或者新查询不与任何已有查询共享相同前缀时（使得计算无法复用），APC 不会带来性能提升。
