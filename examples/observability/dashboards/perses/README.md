# 用于 vLLM 监控的 Perses 仪表板

此目录包含旨在监控 vLLM 性能和指标的 Perses 仪表板配置。

## 要求

- Perses 实例（独立或通过 Operator）
- 在 Perses 中配置了 Prometheus 数据源
- 已启用 Prometheus 指标的 vLLM 部署

## 仪表板格式

我们以**原生 Perses YAML 格式**提供仪表板，适用于所有部署方法：

- **文件**：`*.yaml`（原生 Perses 仪表板规范）
- **格式**：纯仪表板规范，随处可用
- **用法**：适用于独立 Perses、API 导入、CLI 和文件配置
- **Kubernetes**：与 Perses Operator 直接兼容

## 仪表板描述

- **performance_statistics.yaml**：带有聚合延迟统计的性能指标
- **query_statistics.yaml**：查询性能和部署指标

## 部署选项

### 直接导入到 Perses

通过 Perses API 或 CLI 导入仪表板规范：

```bash
percli apply -f performance_statistics.yaml
```

### Perses Operator (Kubernetes)

原生 YAML 格式可直接与 Perses Operator 配合使用：

```bash
kubectl apply -f performance_statistics.yaml -n <namespace>
```

### 文件配置

将 YAML 文件放置在 Perses 配置文件夹中，以便自动加载。
