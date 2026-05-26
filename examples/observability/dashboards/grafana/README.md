# 用于 vLLM 监控的 Grafana 仪表板

此目录包含旨在监控 vLLM 性能和指标的 Grafana 仪表板配置（以 JSON 格式）。

## 要求

- Grafana 8.0+
- 在 Grafana 中配置了 Prometheus 数据源
- 已启用 Prometheus 指标的 vLLM 部署

## 仪表板描述

- **performance_statistics.json**：跟踪性能指标，包括您的 vLLM 服务的延迟和吞吐量。
- **query_statistics.json**：跟踪查询性能、请求量以及您的 vLLM 服务的关键性能指标。

## 部署选项

### 手动导入（推荐）

使用这些仪表板最简单的方式是将 JSON 配置直接手动导入到您的 Grafana 实例中：

1. 导航到您的 Grafana 实例
2. 点击侧边栏中的 '+' 图标
3. 选择 'Import'
4. 复制并粘贴仪表板文件中的 JSON 内容，或直接上传 JSON 文件

### Grafana Operator

如果您在 Kubernetes 中使用 [Grafana Operator](https://github.com/grafana-operator/grafana-operator)，可以将这些 JSON 配置包装在 `GrafanaDashboard` 自定义资源中：

```yaml
# 注意：调整 instanceSelector 以匹配您 Grafana 实例的标签
# 您可以通过以下命令检查：kubectl get grafana -o yaml
apiVersion: grafana.integreatly.org/v1beta1
kind: GrafanaDashboard
metadata:
  name: vllm-performance-dashboard
spec:
  instanceSelector:
    matchLabels:
      dashboards: grafana  # 调整以匹配您的 Grafana 实例标签
  folder: "vLLM Monitoring"
  json: |
    # 将此注释替换为来自
    # performance_statistics.json 的完整 JSON 内容 - JSON 应以 { 开头并以 } 结尾
```

然后应用到您的集群：

```bash
kubectl apply -f your-dashboard.yaml -n <namespace>
```
