# Helm

用于在 Kubernetes 上部署 vLLM 的 Helm Chart

Helm 是 Kubernetes 的包管理器。它帮助自动化在 Kubernetes 上部署 vLLM 应用程序。通过 Helm，您可以使用相同的框架架构，通过覆盖变量值将不同的配置部署到多个命名空间。

本指南将引导您完成使用 Helm 部署 vLLM 的过程，包括必要的先决条件、Helm 安装步骤以及架构和 values 文件的文档说明。

## 前提条件

在开始之前，请确保您具备以下条件：

- 一个正在运行的 Kubernetes 集群
- NVIDIA Kubernetes 设备插件（`k8s-device-plugin`）：可在 [https://github.com/NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) 找到
- 集群中可用的 GPU 资源
- （可选）包含模型权重的 S3 存储桶或其他存储（如果使用自动模型下载）

## 安装 Chart

本指南使用位于 [examples/deployment/chart-helm](../../../examples/deployment/chart-helm) 的 Helm chart。

要安装 release 名称为 `test-vllm` 的 chart：

```bash
helm upgrade --install --create-namespace \
  --namespace=ns-vllm test-vllm . \
  -f values.yaml \
  --set secrets.s3endpoint=$ACCESS_POINT \
  --set secrets.s3bucketname=$BUCKET \
  --set secrets.s3accesskeyid=$ACCESS_KEY \
  --set secrets.s3accesskey=$SECRET_KEY
```

## 卸载 Chart

要卸载 `test-vllm` 部署：

```bash
helm uninstall test-vllm --namespace=ns-vllm
```

该命令会移除所有与 chart 关联的 Kubernetes 组件（**包括持久卷**）并删除该 release。

## 架构

![helm deployment architecture](../../assets/deployment/architecture_helm_deployment.png)

## 配置值

下表描述了 `values.yaml` 中 chart 的可配置参数：

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| autoscaling | object | {"enabled":false,"maxReplicas":100,"minReplicas":1,"targetCPUUtilizationPercentage":80} | 自动缩放配置 |
| autoscaling.enabled | bool | false | 启用自动缩放 |
| autoscaling.maxReplicas | int | 100 | 最大副本数 |
| autoscaling.minReplicas | int | 1 | 最小副本数 |
| autoscaling.targetCPUUtilizationPercentage | int | 80 | 自动缩放的目标 CPU 利用率 |
| configs | object | {} | Configmap |
| containerPort | int | 8000 | 容器端口 |
| customObjects | list | [] | 自定义对象配置 |
| deploymentStrategy | object | {} | 部署策略配置 |
| externalConfigs | list | [] | 外部配置 |
| extraContainers | list | [] | 额外容器配置 |
| extraInit | object | {"modelDownload":{"enabled":true},"initContainers":[],"pvcStorage":"1Gi"} | 初始化容器的额外配置 |
| extraInit.modelDownload | object | {"enabled":true} | 模型下载功能配置 |
| extraInit.modelDownload.enabled | bool | true | 启用自动模型下载作业和等待容器 |
| extraInit.modelDownload.image | object | {"repository":"amazon/aws-cli","tag":"2.6.4","pullPolicy":"IfNotPresent"} | 模型下载操作的镜像 |
| extraInit.modelDownload.waitContainer | object | {} | 等待容器配置（command、args、env） |
| extraInit.modelDownload.downloadJob | object | {} | 下载作业配置（command、args、env） |
| extraInit.initContainers | list | [] | 自定义初始化容器（如果启用了模型下载，则追加在其后） |
| extraInit.pvcStorage | string | "1Gi" | PVC 的存储大小 |
| extraInit.s3modelpath | string | "relative_s3_model_path/opt-125m" | （可选）S3 上模型的路径 |
| extraInit.awsEc2MetadataDisabled | bool | true | （可选）禁用 AWS EC2 元数据服务 |
| extraPorts | list | [] | 额外端口配置 |
| gpuModels | list | ["TYPE_GPU_USED"] | 使用的 GPU 类型 |
| image | object | {"command":["vllm","serve","/data/","--served-model-name","opt-125m","--host","0.0.0.0","--port","8000"],"repository":"vllm/vllm-openai","tag":"latest"} | 镜像配置 |
| image.command | list | ["vllm","serve","/data/","--served-model-name","opt-125m","--host","0.0.0.0","--port","8000"] | 容器启动命令 |
| image.repository | string | "vllm/vllm-openai" | 镜像仓库 |
| image.tag | string | "latest" | 镜像标签 |
| livenessProbe | object | {"failureThreshold":3,"httpGet":{"path":"/health","port":8000},"initialDelaySeconds":15,"periodSeconds":10} | 存活探针配置 |
| livenessProbe.failureThreshold | int | 3 | 连续探测失败次数，超过此次数后 Kubernetes 认为整体检查失败：容器不存活 |
| livenessProbe.httpGet | object | {"path":"/health","port":8000} | kubelet 在服务器上执行 HTTP 请求的配置 |
| livenessProbe.httpGet.path | string | "/health" | HTTP 服务器上访问的路径 |
| livenessProbe.httpGet.port | int | 8000 | 容器上要访问的端口号或名称，服务器在其上监听 |
| livenessProbe.initialDelaySeconds | int | 15 | 容器启动后等待多少秒才开始执行存活探针 |
| livenessProbe.periodSeconds | int | 10 | 执行存活探针的间隔（秒） |
| maxUnavailablePodDisruptionBudget | string | "" | 干扰预算配置 |
| readinessProbe | object | {"failureThreshold":3,"httpGet":{"path":"/health","port":8000},"initialDelaySeconds":5,"periodSeconds":5} | 就绪探针配置 |
| readinessProbe.failureThreshold | int | 3 | 连续探测失败次数，超过此次数后 Kubernetes 认为整体检查失败：容器未就绪 |
| readinessProbe.httpGet | object | {"path":"/health","port":8000} | kubelet 在服务器上执行 HTTP 请求的配置 |
| readinessProbe.httpGet.path | string | "/health" | HTTP 服务器上访问的路径 |
| readinessProbe.httpGet.port | int | 8000 | 容器上要访问的端口号或名称，服务器在其上监听 |
| readinessProbe.initialDelaySeconds | int | 5 | 容器启动后等待多少秒才开始执行就绪探针 |
| readinessProbe.periodSeconds | int | 5 | 执行就绪探针的间隔（秒） |
| replicaCount | int | 1 | 副本数 |
| resources | object | {"limits":{"cpu":4,"memory":"16Gi","nvidia.com/gpu":1},"requests":{"cpu":4,"memory":"16Gi","nvidia.com/gpu":1}} | 资源配置 |
| resources.limits."nvidia.com/gpu" | int | 1 | 使用的 GPU 数量 |
| resources.limits.cpu | int | 4 | CPU 数量 |
| resources.limits.memory | string | "16Gi" | CPU 内存配置 |
| resources.requests."nvidia.com/gpu" | int | 1 | 使用的 GPU 数量 |
| resources.requests.cpu | int | 4 | CPU 数量 |
| resources.requests.memory | string | "16Gi" | CPU 内存配置 |
| secrets | object | {} | 密钥配置 |
| serviceName | string | "" | 服务名称 |
| servicePort | int | 80 | 服务端口 |
| labels.environment | string | test | 环境名称 |

## 配置示例

### 使用 S3 模型下载（默认）

```yaml
extraInit:
  modelDownload:
    enabled: true
  pvcStorage: "10Gi"
  s3modelpath: "models/llama-7b"
```

### 仅使用自定义初始化容器

适用于像 llm-d 这样需要自定义 sidecar 而不需要模型下载的用例：

```yaml
extraInit:
  modelDownload:
    enabled: false
  initContainers:
    - name: llm-d-routing-proxy
      image: ghcr.io/llm-d/llm-d-routing-sidecar:v0.2.0
      imagePullPolicy: IfNotPresent
      ports:
        - containerPort: 8080
          name: proxy
      securityContext:
        runAsUser: 1000
      restartPolicy: Always
  pvcStorage: "10Gi"
```
