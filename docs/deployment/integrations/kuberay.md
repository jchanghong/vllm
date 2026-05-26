# KubeRay

[KubeRay](https://github.com/ray-project/kuberay) 提供了一种 Kubernetes 原生的方式在 Ray 集群上运行 vLLM 工作负载。
Ray 集群可以用 YAML 声明，然后算子处理 Pod 调度、网络配置、重启和蓝绿部署——同时保留熟悉的 Kubernetes 体验。

## 为什么选择 KubeRay 而不是手动脚本？

| 特性 | 手动脚本 | KubeRay |
| ------- | --------------------------------------------------------- | ------- |
| 集群引导 | 手动 SSH 到每个节点并运行脚本 | 一条命令即可创建或更新整个集群：`kubectl apply -f cluster.yaml` |
| 自动扩缩容 | 手动 | 自动修补 CRD 以调整集群大小 |
| 升级 | 手动拆除并重建 | 支持蓝绿部署更新 |
| 声明式配置 | Bash 标志和环境变量 | GitOps 友好的 YAML CRD（RayCluster/RayService） |

使用 KubeRay 可降低运维负担，并简化 Ray + vLLM 与现有 Kubernetes 工作流（CI/CD、密钥、存储类等）的集成。

## 了解更多

* ["在 Kubernetes 上使用 Ray Serve LLM 提供大语言模型服务"](https://docs.ray.io/en/master/cluster/kubernetes/examples/rayserve-llm-example.html) - 一个端到端示例，演示如何使用 vLLM、KubeRay 和 Ray Serve 提供模型服务。
* [KubeRay 文档](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
