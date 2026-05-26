# 分布式部署故障排除

有关一般故障排除，请参见[故障排除](../usage/troubleshooting.md)。

## 验证节点间 GPU 通信

启动 Ray 集群后，验证节点间的 GPU 到 GPU 通信。正确的配置可能并不简单。有关更多信息，请参见[故障排除脚本](../usage/troubleshooting.md#incorrect-hardwaredriver)。如果你需要额外的环境变量用于通信配置，请将它们附加到 [examples/ray_serving/run_cluster.sh](../../examples/ray_serving/run_cluster.sh) 中，例如 `-e NCCL_SOCKET_IFNAME=eth0`。建议在集群创建期间设置环境变量，因为变量会传播到所有节点。相比之下，在 shell 中设置环境变量仅影响本地节点。有关更多信息，请参见 <https://github.com/vllm-project/vllm/issues/6803>。

## 没有可用的节点类型能够满足资源请求

即使集群有足够的 GPU，也可能出现错误消息 `Error: No available node types can fulfill resource request`。该问题通常发生在节点有多个 IP 地址且 vLLM 无法选择正确地址时。通过在 [examples/ray_serving/run_cluster.sh](../../examples/ray_serving/run_cluster.sh) 中设置 `VLLM_HOST_IP`（在每个节点上使用不同的值），确保 vLLM 和 Ray 使用相同的 IP 地址。使用 `ray status` 和 `ray list nodes` 来验证所选的 IP 地址。有关更多信息，请参见 <https://github.com/vllm-project/vllm/issues/7815>。

## Ray 可观测性

由于规模庞大和复杂性，调试分布式系统可能具有挑战性。Ray 提供了一套工具来帮助监控、调试和优化 Ray 应用和集群。有关 Ray 可观测性的更多信息，请访问[官方 Ray 可观测性文档](https://docs.ray.io/en/latest/ray-observability/index.html)。有关调试 Ray 应用的更多信息，请访问 [Ray 调试指南](https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/index.html)。有关 Kubernetes 集群故障排除的信息，请参阅[官方 KubeRay 故障排除指南](https://docs.ray.io/en/latest/serve/advanced-guides/multi-node-gpu-troubleshooting.html)。
