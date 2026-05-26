# 监控仪表板

此目录包含用于 vLLM 的监控仪表板配置，为您的 vLLM 部署提供
全面的可观测性。

## 仪表板平台

我们为两个流行的可观测性平台提供了仪表板：

- **[Grafana](https://grafana.com)**
- **[Perses](https://perses.dev)**

## 仪表板格式方案

所有仪表板均以**原生格式**提供，适用于不同的
部署方式：

### Grafana (JSON)

- ✅ 适用于任何 Grafana 实例（云端、自托管、Docker）
- ✅ 通过 Grafana UI 或 API 直接导入
- ✅ 必要时可包装在 Kubernetes Operator 中
- ✅ 无供应商锁定或部署依赖

### Perses (YAML)

- ✅ 适用于独立 Perses 实例
- ✅ 兼容 Perses API 和 CLI
- ✅ 支持仪表板即代码（Dashboard-as-Code）工作流
- ✅ 必要时可包装在 Kubernetes Operator 中

## 仪表板内容

两个平台提供同等的监控能力：

| 仪表板 | 描述 |
| --------- | ----------- |
| **性能统计** | 跟踪延迟、吞吐量和性能指标 |
| **查询统计** | 监控请求量、查询性能和关键指标 |

## 快速开始

首先，导航到此示例的目录：

```bash
cd examples/observability/dashboards
```

### Grafana

将 JSON 直接导入 Grafana UI，或使用 API：

```bash
curl -X POST http://grafana/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana/performance_statistics.json
```

### Perses

通过 Perses CLI 导入：

```bash
percli apply -f perses/performance_statistics.yaml
```

## 要求

- 来自 vLLM 部署的 **Prometheus** 指标
- 在您的监控平台中配置了**数据源**
- **vLLM 指标**已启用并可访问

## 平台特定文档

有关详细的部署说明和平台特定选项，请参阅：

- **[Grafana 文档](grafana)** - JSON 仪表板、Operator 用法、手动导入
- **[Perses 文档](perses)** - YAML 规范、CLI 用法、Operator 封装

## 贡献

在添加新的仪表板时，请：

1. 提供原生格式（Grafana 使用 JSON，Perses 使用 YAML 规范）
2. 更新特定平台的 README 文件
3. 确保仪表板在各种部署方法下都能工作
4. 使用最新的平台版本进行测试
