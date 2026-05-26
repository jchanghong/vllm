# Helm Charts

此目录包含用于部署 vllm 应用的 Helm chart。该 chart 包含部署、自动扩缩容、资源管理等配置。

## 文件

- Chart.yaml: 定义 chart 元数据，包括名称、版本和维护者。
- ct.yaml: 用于 chart 测试的配置。
- lintconf.yaml: YAML 文件的 lint 规则。
- values.schema.json: 用于验证 values.yaml 的 JSON schema。
- values.yaml: Helm chart 的默认值。
- templates/_helpers.tpl: 用于定义通用配置的辅助模板。
- templates/configmap.yaml: 用于创建 ConfigMap 的模板。
- templates/custom-objects.yaml: 用于自定义 Kubernetes 对象的模板。
- templates/deployment.yaml: 用于创建 Deployment 的模板。
- templates/hpa.yaml: 用于创建 Horizontal Pod Autoscaler 的模板。
- templates/job.yaml: 用于创建 Kubernetes Job 的模板。
- templates/poddisruptionbudget.yaml: 用于创建 Pod Disruption Budget 的模板。
- templates/pvc.yaml: 用于创建 Persistent Volume Claim 的模板。
- templates/secrets.yaml: 用于创建 Kubernetes Secret 的模板。
- templates/service.yaml: 用于创建 Service 的模板。

## 运行测试

此 chart 包含使用 [helm-unittest](https://github.com/helm-unittest/helm-unittest) 的单元测试。安装插件并运行测试：

```bash
# 安装插件
helm plugin install https://github.com/helm-unittest/helm-unittest

# 运行测试
helm unittest .
```
