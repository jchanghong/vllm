# 示例

vLLM 的示例分为以下几类：

- **[`basic/`](../../examples/basic)** – 离线和在线推理的最小示例。
- **[`generate/`](../../examples/generate)** – 文本生成示例，包括多模态模型。
- **[`pooling/`](../../examples/pooling)** – 嵌入、分类、评分、奖励等示例。
- **[`speech_to_text/`](../../examples/speech_to_text)** – 语音转录、翻译和实时音频示例。
- **[`features/`](../../examples/features)** – 单个 vLLM 特性的演示：自动前缀缓存、推测解码、LoRA、结构化输出、提示嵌入、暂停/恢复、批处理不变性、KV 事件、数据并行等。
- **[`reasoning/`](../../examples/reasoning)** – 使用 vLLM 进行推理的示例。
- **[`tool_calling/`](../../examples/tool_calling)** – 使用 vLLM 进行函数/工具调用的示例。
- **[`applications/`](../../examples/applications)** – 应用示例，如聊天机器人和 RAG（检索增强生成）。
- **[`rl/`](../../examples/rl)** – 强化学习示例。
- **[`deployment/`](../../examples/deployment)** – 在生产环境中部署 vLLM 的示例。
- **[`ray_serving/`](../../examples/ray_serving)** – 使用 Ray 进行可扩展服务。
- **[`disaggregated/`](../../examples/disaggregated)** – 分离式服务示例（分离预填充和解码），包括各种 KV 缓存连接器（LMCache、Mooncake、FlexKV、P2P NCCL）和故障恢复。
- **[`observability/`](../../examples/observability)** – 指标、日志、追踪（OpenTelemetry）和仪表盘（Grafana、Perses）。
