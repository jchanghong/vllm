# Kthena

[**Kthena**](https://github.com/volcano-sh/kthena) 是一个 Kubernetes 原生的 LLM 推理平台，改变了组织在生产环境中部署和管理大语言模型的方式。它基于声明式模型生命周期管理和智能请求路由构建，为 LLM 推理工作负载提供高性能和企业级可扩展性。

本指南展示了如何在 Kubernetes 上部署一个生产级的、**多节点 vLLM** 服务。

我们将：

- 安装所需的组件（Kthena + Volcano）。
- 通过 Kthena 的 `ModelServing` CR 部署一个多节点 vLLM 模型。
- 验证部署。

---

## 1. 前提条件

您需要：

- 一个带有 **GPU 节点**的 Kubernetes 集群。
- 具有 cluster-admin 或同等权限的 `kubectl` 访问权限。
- 安装 **Volcano** 用于组调度。
- 安装 **Kthena**，且 `ModelServing` CRD 可用。
- 如果从 Hugging Face Hub 加载模型，需要有效的 **Hugging Face 令牌**。

### 1.1 安装 Volcano

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
```

这提供了 Kthena 使用的组调度和网络拓扑功能。

### 1.2 安装 Kthena

```bash
helm install kthena oci://ghcr.io/volcano-sh/charts/kthena --version v0.1.0 --namespace kthena-system --create-namespace
```

- 将创建 `kthena-system` 命名空间。
- Kthena 控制器和 CRD（包括 `ModelServing`）将被安装并运行正常。

验证：

```bash
kubectl get crd | grep modelserving
```

您应该看到：

```text
modelservings.workload.serving.volcano.sh   ...
```

---

## 2. 多节点 vLLM `ModelServing` 示例

Kthena 提供了一个示例清单，用于部署一个**运行 Llama 的多节点 vLLM 集群**。从概念上讲，这相当于 vLLM 生产栈的 Helm 部署，但使用 `ModelServing` 来表达。

该示例（`llama-multinode`）的简化版本如下所示：

- `spec.replicas: 1` – 一个 `ServingGroup`（一个逻辑模型部署）。
- `roles`：
    - `entryTemplate` – 定义 **leader** Pod，运行：
        - vLLM 的 **多节点集群引导脚本**（Ray 集群）。
        - vLLM **OpenAI 兼容的 API 服务器**。
    - `workerTemplate` – 定义 **worker** Pod，加入 leader 的 Ray 集群。

示例 YAML 的关键点：

- **镜像**：`vllm/vllm-openai:latest`（与上游 vLLM 镜像一致）。
- **命令**（leader）：

  ```yaml
  command:
    - sh
    - -c
    - >
      bash /vllm-workspace/examples/ray_serving/multi-node-serving.sh leader --ray_cluster_size=2;
      vllm serve meta-llama/Llama-3.1-405B-Instruct
        --port 8080
        --tensor-parallel-size 8
        --pipeline-parallel-size 2
  ```

- **命令**（worker）：

  ```yaml
  command:
    - sh
    - -c
    - >
      bash /vllm-workspace/examples/ray_serving/multi-node-serving.sh worker --ray_address=$(ENTRY_ADDRESS)
  ```

---

## 3. 通过 Kthena 部署多节点 Llama vLLM

### 3.1 准备清单

**推荐**：使用 Secret 而不是原始环境变量：

```bash
kubectl create secret generic hf-token \
  -n default \
  --from-literal=HUGGING_FACE_HUB_TOKEN='<your-token>'
```

### 3.2 应用 `ModelServing`

```bash
cat  <<EOF | kubectl apply -f -
apiVersion: workload.serving.volcano.sh/v1alpha1
kind: ModelServing
metadata:
  name: llama-multinode
  namespace: default
spec:
  schedulerName: volcano
  replicas: 1  # group replicas
  template:
    restartGracePeriodSeconds: 60
    gangPolicy:
      minRoleReplicas:
        405b: 1
    roles:
      - name: 405b
        replicas: 2
        entryTemplate:
          spec:
            containers:
              - name: leader
                image: vllm/vllm-openai:latest
                env:
                  - name: HUGGING_FACE_HUB_TOKEN
                    valueFrom:
                      secretKeyRef:
                        name: hf-token
                        key: HUGGING_FACE_HUB_TOKEN
                command:
                  - sh
                  - -c
                  - "bash /vllm-workspace/examples/ray_serving/multi-node-serving.sh leader --ray_cluster_size=2; 
                    vllm serve meta-llama/Llama-3.1-405B-Instruct --port 8080 --tensor-parallel-size 8 --pipeline-parallel-size 2"
                resources:
                  limits:
                    nvidia.com/gpu: "8"
                    memory: 1124Gi
                    ephemeral-storage: 800Gi
                  requests:
                    ephemeral-storage: 800Gi
                    cpu: 125
                ports:
                  - containerPort: 8080
                readinessProbe:
                  tcpSocket:
                    port: 8080
                  initialDelaySeconds: 15
                  periodSeconds: 10
                volumeMounts:
                  - mountPath: /dev/shm
                    name: dshm
            volumes:
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 15Gi
        workerReplicas: 1
        workerTemplate:
          spec:
            containers:
              - name: worker
                image: vllm/vllm-openai:latest
                command:
                  - sh
                  - -c
                  - "bash /vllm-workspace/examples/ray_serving/multi-node-serving.sh worker --ray_address=$(ENTRY_ADDRESS)"
                resources:
                  limits:
                    nvidia.com/gpu: "8"
                    memory: 1124Gi
                    ephemeral-storage: 800Gi
                  requests:
                    ephemeral-storage: 800Gi
                    cpu: 125
                env:
                  - name: HUGGING_FACE_HUB_TOKEN
                    valueFrom:
                      secretKeyRef:
                        name: hf-token
                        key: HUGGING_FACE_HUB_TOKEN
                volumeMounts:
                  - mountPath: /dev/shm
                    name: dshm   
            volumes:
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: 15Gi
EOF
```

Kthena 将：

- 创建一个 `ModelServing` 对象。
- 派生一个用于 Volcano 组调度的 `PodGroup`。
- 为每个 `ServingGroup` 和 `Role` 创建 leader 和 worker Pod。

---

## 4. 验证部署

### 4.1 检查 ModelServing 状态

使用来自 Kthena 文档的代码片段：

```bash
kubectl get modelserving -oyaml | grep status -A 10
```

您应该看到类似以下内容：

```yaml
status:
  availableReplicas: 1
  conditions:
    - type: Available
      status: "True"
      reason: AllGroupsReady
      message: All Serving groups are ready
    - type: Progressing
      status: "False"
      ...
  replicas: 1
  updatedReplicas: 1
```

### 4.2 检查 Pod

列出您部署的 Pod：

```bash
kubectl get pod -owide -l modelserving.volcano.sh/name=llama-multinode
```

示例输出（来自文档）：

```text
NAMESPACE   NAME                          READY   STATUS    RESTARTS   AGE   IP            NODE           ...
default     llama-multinode-0-405b-0-0    1/1     Running   0          15m   10.244.0.56   192.168.5.12   ...
default     llama-multinode-0-405b-0-1    1/1     Running   0          15m   10.244.0.58   192.168.5.43   ...
default     llama-multinode-0-405b-1-0    1/1     Running   0          15m   10.244.0.57   192.168.5.58   ...
default     llama-multinode-0-405b-1-1    1/1     Running   0          15m   10.244.0.53   192.168.5.36   ...
```

Pod 名称模式：

- `llama-multinode-<group-idx>-<role-name>-<replica-idx>-<ordinal>`。

第一个数字表示 `ServingGroup`。第二个（`405b`）是 `Role`。其余索引标识角色内的 Pod。

---

## 6. 访问 vLLM OpenAI 兼容 API

通过 Service 暴露入口：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: llama-multinode-openai
  namespace: default
spec:
  selector:
    modelserving.volcano.sh/name: llama-multinode
    modelserving.volcano.sh/entry: "true"
    # 如果需要，可进一步缩小到 leader 角色范围
  ports:
    - name: http
      port: 80
      targetPort: 8080
  type: ClusterIP
```

从本地机器进行端口转发：

```bash
kubectl port-forward svc/llama-multinode-openai 30080:80 -n default
```

然后：

- 列出模型：

  ```bash
  curl -s http://localhost:30080/v1/models
  ```

- 发送补全请求（与 vLLM 生产栈文档相同）：

  ```bash
  curl -X POST http://localhost:30080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "meta-llama/Llama-3.1-405B-Instruct",
      "prompt": "Once upon a time,",
      "max_tokens": 10
    }'
  ```

您应该会看到来自 vLLM 的 OpenAI 风格响应。

---

## 7. 清理

要移除部署及其资源：

```bash
kubectl delete modelserving llama-multinode -n default
```

如果您已完成整个栈的操作：

```bash
helm uninstall kthena -n kthena-system   # 或您的 Kthena 发布名称
helm uninstall volcano -n volcano-system
```
