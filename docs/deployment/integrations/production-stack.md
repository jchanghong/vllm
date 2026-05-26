# 生产栈

在 Kubernetes 上部署 vLLM 是一种可扩展且高效的方式来提供机器学习模型服务。本指南将引导您使用 [vLLM 生产栈](https://github.com/vllm-project/production-stack) 部署 vLLM。源于伯克利-芝加哥大学的合作项目，[vLLM 生产栈](https://github.com/vllm-project/production-stack) 是 [vLLM 项目](https://github.com/vllm-project) 下官方发布的、面向生产优化的代码库，专为 LLM 部署而设计，具有以下特点：

* **上游 vLLM 兼容性** – 它封装了上游 vLLM，不修改其代码。
* **易于使用** – 通过 Helm Chart 简化部署，通过 Grafana 仪表盘提供可观测性。
* **高性能** – 针对 LLM 工作负载进行了优化，具有多模型支持、模型感知和前缀感知路由、快速 vLLM 引导启动以及使用 [LMCache](https://github.com/LMCache/LMCache) 的 KV 缓存卸载等功能。

如果您是 Kubernetes 新手，别担心：在 vLLM 生产栈 [仓库](https://github.com/vllm-project/production-stack) 中，我们提供了逐步的 [指南](https://github.com/vllm-project/production-stack/blob/main/tutorials/00-install-kubernetes-env.md) 和一个 [短视频](https://www.youtube.com/watch?v=EsTJbQtzj0g)，帮助您在 **4 分钟** 内完成设置并开始使用！

## 前提条件

确保您有一个运行中的 Kubernetes 环境并配备 GPU（您可以按照 [此教程](https://github.com/vllm-project/production-stack/blob/main/tutorials/00-install-kubernetes-env.md) 在裸机 GPU 机器上安装 Kubernetes 环境）。

## 使用 vLLM 生产栈进行部署

标准 vLLM 生产栈使用 Helm Chart 安装。您可以运行此 [bash 脚本](https://github.com/vllm-project/production-stack/blob/main/utils/install-helm.sh) 在 GPU 服务器上安装 Helm。

要在您的桌面计算机上安装 vLLM 生产栈，运行以下命令：

```bash
sudo helm repo add vllm https://vllm-project.github.io/production-stack
sudo helm install vllm vllm/vllm-stack -f tutorials/assets/values-01-minimal-example.yaml
```

这将实例化一个名为 `vllm` 的基于 vLLM 生产栈的部署，运行一个小型 LLM（Facebook opt-125M 模型）。

### 验证安装

使用以下命令监视部署状态：

```bash
sudo kubectl get pods
```

您将看到 `vllm` 部署的 Pod 转换为 `Running` 状态。

```text
NAME                                           READY   STATUS    RESTARTS   AGE
vllm-deployment-router-859d8fb668-2x2b7        1/1     Running   0          2m38s
vllm-opt125m-deployment-vllm-84dfc9bd7-vb9bs   1/1     Running   0          2m38s
```

!!! note
    容器下载 Docker 镜像和 LLM 权重可能需要一些时间。

### 向栈发送查询

将 `vllm-router-service` 端口转发到宿主机：

```bash
sudo kubectl port-forward svc/vllm-router-service 30080:80
```

然后您就可以向 OpenAI 兼容的 API 发送查询以检查可用的模型：

```bash
curl -o- http://localhost:30080/v1/models
```

??? console "输出"

    ```json
    {
      "object": "list",
      "data": [
        {
          "id": "facebook/opt-125m",
          "object": "model",
          "created": 1737428424,
          "owned_by": "vllm",
          "root": null
        }
      ]
    }
    ```

要发送实际的聊天请求，您可以向 OpenAI 的 `/completion` 端点发出 curl 请求：

```bash
curl -X POST http://localhost:30080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Once upon a time,",
    "max_tokens": 10
  }'
```

??? console "输出"

    ```json
    {
      "id": "completion-id",
      "object": "text_completion",
      "created": 1737428424,
      "model": "facebook/opt-125m",
      "choices": [
        {
          "text": " there was a brave knight who...",
          "index": 0,
          "finish_reason": "length"
        }
      ]
    }
    ```

### 卸载

要移除部署，运行：

```bash
sudo helm uninstall vllm
```

---

### （高级）配置 vLLM 生产栈

核心 vLLM 生产栈配置使用 YAML 管理。以下是上述安装中使用的示例配置：

??? code "Yaml"

    ```yaml
    servingEngineSpec:
      runtimeClassName: ""
      modelSpec:
      - name: "opt125m"
        repository: "vllm/vllm-openai"
        tag: "latest"
        modelURL: "facebook/opt-125m"

        replicaCount: 1

        requestCPU: 6
        requestMemory: "16Gi"
        requestGPU: 1

        pvcStorage: "10Gi"
    ```

在此 YAML 配置中：

* **`modelSpec`** 包含：
    * `name`：您对该模型的偏好昵称。
    * `repository`：vLLM 的 Docker 仓库。
    * `tag`：Docker 镜像标签。
    * `modelURL`：您想要使用的 LLM 模型。
* **`replicaCount`**：副本数量。
* **`requestCPU` 和 `requestMemory`**：指定 Pod 的 CPU 和内存资源请求。
* **`requestGPU`**：指定所需的 GPU 数量。
* **`pvcStorage`**：为模型分配持久化存储。

!!! note
    如果您打算设置两个 Pod，请参考此 [YAML 文件](https://github.com/vllm-project/production-stack/blob/main/tutorials/assets/values-01-2pods-minimal-example.yaml)。

!!! tip
    vLLM 生产栈提供更多特性（*例如* CPU 卸载和广泛的负载均衡算法）。请查看这些 [示例和教程](https://github.com/vllm-project/production-stack/tree/main/tutorials) 以及我们的 [仓库](https://github.com/vllm-project/production-stack) 以了解更多详情！
