# Anyscale

[Anyscale](https://www.anyscale.com) 是一个由 Ray 的创建者开发的多云托管平台。

Anyscale 在您的 AWS、GCP 或 Azure 账户中自动化 Ray 集群的整个生命周期，提供开源 Ray 的灵活性，
而无需承担维护 Kubernetes 控制平面、配置自动缩放器、管理可观测性堆栈，或使用辅助脚本（如 [examples/ray_serving/run_cluster.sh](../../../examples/ray_serving/run_cluster.sh)）手动管理头节点和工作节点的运维负担。

当使用 vLLM 服务大语言模型时，Anyscale 可以快速配置[生产就绪的 HTTPS 端点](https://docs.anyscale.com/examples/deploy-ray-serve-llms)或[容错的批量推理任务](https://docs.anyscale.com/examples/ray-data-llm)。

## 在 Anyscale 上运行生产就绪 vLLM 的快速入门

- [离线批量推理](https://console.anyscale.com/template-preview/llm_batch_inference?utm_source=vllm_docs)
- [部署 vLLM 服务](https://console.anyscale.com/template-preview/llm_serving?utm_source=vllm_docs)
- [整理数据集](https://console.anyscale.com/template-preview/audio-dataset-curation-llm-judge?utm_source=vllm_docs)
- [微调 LLM](https://console.anyscale.com/template-preview/entity-recognition-with-llms?utm_source=vllm_docs)
