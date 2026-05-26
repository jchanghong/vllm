# 使用 Kubernetes

在 Kubernetes 上部署 vLLM 是一种可扩展且高效的方式来提供机器学习模型服务。本指南将引导您使用原生 Kubernetes 部署 vLLM。

- [使用 CPU 部署](#deployment-with-cpus)
- [使用 GPU 部署](#deployment-with-gpus)
- [使用 gRPC 提供服务](#serving-with-grpc)
- [故障排除](#troubleshooting)
    - [启动探针或就绪探针失败，容器日志包含 "KeyboardInterrupt: terminated"](#startup-probe-or-readiness-probe-failure-container-log-contains-keyboardinterrupt-terminated)
- [总结](#conclusion)

或者，您也可以使用以下任一方式将 vLLM 部署到 Kubernetes：

- [Helm](frameworks/helm.md)
- [NVIDIA Dynamo](integrations/dynamo.md)
- [InftyAI/llmaz](integrations/llmaz.md)
- [llm-d](integrations/llm-d.md)
- [KAITO](integrations/kaito.md)
- [KServe](integrations/kserve.md)
- [Kthena](integrations/kthena.md)
- [KubeRay](integrations/kuberay.md)
- [kubernetes-sigs/lws](frameworks/lws.md)
- [meta-llama/llama-stack](integrations/llamastack.md)
- [substratusai/kubeai](integrations/kubeai.md)
- [vllm-project/AIBrix](integrations/aibrix.md)
- [vllm-project/production-stack](integrations/production-stack.md)

## 使用 CPU 部署

!!! note
    此处使用 CPU 仅用于演示和测试目的，其性能无法与 GPU 相提并论。

首先，创建一个 Kubernetes PVC 和 Secret，用于下载和存储 Hugging Face 模型：

??? console "配置"

    ```bash
    cat <<EOF |kubectl apply -f -
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: vllm-models
    spec:
      accessModes:
        - ReadWriteOnce
      volumeMode: Filesystem
      resources:
        requests:
          storage: 50Gi
    ---
    apiVersion: v1
    kind: Secret
    metadata:
      name: hf-token-secret
    type: Opaque
    stringData:
      token: "REPLACE_WITH_TOKEN"
    EOF
    ```

此处，`token` 字段存储您的 **Hugging Face 访问令牌**。有关如何生成令牌的详细信息，请参阅 [Hugging Face 文档](https://huggingface.co/docs/hub/en/security-tokens)。

接下来，以 Kubernetes Deployment 和 Service 的形式启动 vLLM 服务器。

注意，您需要根据处理器架构配置 vLLM 镜像：

??? console "配置"

    ```bash
    VLLM_IMAGE=public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:latest       # x86_64 使用此镜像
    VLLM_IMAGE=public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:latest # arm64 使用此镜像
    cat <<EOF |kubectl apply -f -
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-server
    spec:
      replicas: 1
      selector:
        matchLabels:
          app.kubernetes.io/name: vllm
      template:
        metadata:
          labels:
            app.kubernetes.io/name: vllm
        spec:
          containers:
          - name: vllm
            image: $VLLM_IMAGE
            command: ["/bin/sh", "-c"]
            args: [
              "vllm serve meta-llama/Llama-3.2-1B-Instruct"
            ]
            env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: token
            ports:
              - containerPort: 8000
            volumeMounts:
              - name: llama-storage
                mountPath: /root/.cache/huggingface
          volumes:
          - name: llama-storage
            persistentVolumeClaim:
              claimName: vllm-models
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: vllm-server
    spec:
      selector:
        app.kubernetes.io/name: vllm
      ports:
      - protocol: TCP
        port: 8000
        targetPort: 8000
      type: ClusterIP
    EOF
    ```

我们可以通过日志验证 vLLM 服务器是否已成功启动（下载模型可能需要几分钟）：

```bash
kubectl logs -l app.kubernetes.io/name=vllm
...
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 使用 GPU 部署

**前置条件**：确保您有一个运行中的 [带有 GPU 的 Kubernetes 集群](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)。

1. 为 vLLM 创建 PVC、Secret 和 Deployment

      PVC 用于存储模型缓存，是可选的；您也可以使用 hostPath 或其他存储选项。

      <details>
      <summary>Yaml</summary>

      ```yaml
      apiVersion: v1
      kind: PersistentVolumeClaim
      metadata:
        name: mistral-7b
        namespace: default
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 50Gi
        storageClassName: default
        volumeMode: Filesystem
      ```

      </details>

      Secret 是可选的，仅在访问受限模型时需要；如果您不使用受限模型，可以跳过此步骤。

      ```yaml
      apiVersion: v1
      kind: Secret
      metadata:
        name: hf-token-secret
        namespace: default
      type: Opaque
      stringData:
        token: "REPLACE_WITH_TOKEN"
      ```
  
      接下来，创建 Deployment 文件以运行 vLLM 模型服务器。以下示例部署了 `Mistral-7B-Instruct-v0.3` 模型。

      以下是使用 NVIDIA GPU 和 AMD GPU 的两个示例。

      NVIDIA GPU：

      <details>
      <summary>Yaml</summary>

      ```yaml
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: mistral-7b
        namespace: default
        labels:
          app: mistral-7b
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: mistral-7b
        template:
          metadata:
            labels:
              app: mistral-7b
          spec:
            volumes:
            - name: cache-volume
              persistentVolumeClaim:
                claimName: mistral-7b
            # vLLM 需要访问主机的共享内存以进行张量并行推理。
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "2Gi"
            containers:
            - name: mistral-7b
              image: vllm/vllm-openai:latest
              command: ["/bin/sh", "-c"]
              args: [
                "vllm serve mistralai/Mistral-7B-Instruct-v0.3 --trust-remote-code --enable-chunked-prefill --max_num_batched_tokens 1024"
              ]
              env:
              - name: HF_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token-secret
                    key: token
              ports:
              - containerPort: 8000
              resources:
                limits:
                  cpu: "10"
                  memory: 20G
                  nvidia.com/gpu: "1"
                requests:
                  cpu: "2"
                  memory: 6G
                  nvidia.com/gpu: "1"
              volumeMounts:
              - mountPath: /root/.cache/huggingface
                name: cache-volume
              - name: shm
                mountPath: /dev/shm
              livenessProbe:
                httpGet:
                  path: /health
                  port: 8000
                initialDelaySeconds: 60
                periodSeconds: 10
              readinessProbe:
                httpGet:
                  path: /health
                  port: 8000
                initialDelaySeconds: 60
                periodSeconds: 5
      ```

      </details>

      AMD GPU：

      如果您使用的是 AMD ROCm GPU（如 MI300X），可以参考下面的 `deployment.yaml`。

      <details>
      <summary>Yaml</summary>

      ```yaml
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: mistral-7b
        namespace: default
        labels:
          app: mistral-7b
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: mistral-7b
        template:
          metadata:
            labels:
              app: mistral-7b
          spec:
            volumes:
            # PVC
            - name: cache-volume
              persistentVolumeClaim:
                claimName: mistral-7b
            # vLLM 需要访问主机的共享内存以进行张量并行推理。
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "8Gi"
            hostNetwork: true
            hostIPC: true
            containers:
            - name: mistral-7b
              image: rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
              securityContext:
                seccompProfile:
                  type: Unconfined
                runAsGroup: 44
                capabilities:
                  add:
                  - SYS_PTRACE
              command: ["/bin/sh", "-c"]
              args: [
                "vllm serve mistralai/Mistral-7B-v0.3 --port 8000 --trust-remote-code --enable-chunked-prefill --max_num_batched_tokens 1024"
              ]
              env:
              - name: HF_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token-secret
                    key: token
              ports:
              - containerPort: 8000
              resources:
                limits:
                  cpu: "10"
                  memory: 20G
                  amd.com/gpu: "1"
                requests:
                  cpu: "6"
                  memory: 6G
                  amd.com/gpu: "1"
              volumeMounts:
              - name: cache-volume
                mountPath: /root/.cache/huggingface
              - name: shm
                mountPath: /dev/shm
      ```

      </details>

      您可以从 <https://github.com/ROCm/k8s-device-plugin/tree/master/example/vllm-serve> 获取包含步骤和示例 yaml 文件的完整示例。

2. 为 vLLM 创建 Kubernetes Service

      接下来，创建 Kubernetes Service 文件以暴露 `mistral-7b` 部署：

      <details>
      <summary>Yaml</summary>

      ```yaml
      apiVersion: v1
      kind: Service
      metadata:
        name: mistral-7b
        namespace: default
      spec:
        ports:
        - name: http-mistral-7b
          port: 80
          protocol: TCP
          targetPort: 8000
        # 标签选择器应与部署标签匹配，对前缀缓存功能有用
        selector:
          app: mistral-7b
        sessionAffinity: None
        type: ClusterIP
      ```

      </details>

3. 部署和测试

      使用 `kubectl apply -f <filename>` 应用部署和服务配置：

      ```bash
      kubectl apply -f deployment.yaml
      kubectl apply -f service.yaml
      ```

      要测试部署，运行以下 `curl` 命令：

      ```bash
      curl http://mistral-7b.default.svc.cluster.local/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
              "model": "mistralai/Mistral-7B-Instruct-v0.3",
              "prompt": "San Francisco is a",
              "max_tokens": 7,
              "temperature": 0
            }'
      ```

      如果服务正确部署，您将收到来自 vLLM 模型的响应。

## 使用 gRPC 提供服务

vLLM 可以通过传递 `--grpc` 标志来通过 gRPC 而非 HTTP 提供模型服务。这需要可选的 gRPC 依赖：

```bash
pip install vllm[grpc]
```

使用 `--grpc` 时，服务器会暴露标准的 [gRPC 健康检查协议](https://github.com/grpc/grpc/blob/master/doc/health-checking.md)（`grpc.health.v1.Health`），该协议可与 Kubernetes [原生 gRPC 探针](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/#define-a-grpc-liveness-probe)（自 Kubernetes 1.24 起可用）集成。

要使用 gRPC 部署，更改 `vllm serve` 命令以包含 `--grpc`，并将 `httpGet` 探针替换为 `grpc` 探针：

```yaml
containers:
- name: mistral-7b
  image: vllm/vllm-openai:latest
  command: ["/bin/sh", "-c"]
  args: [
    "pip install vllm[grpc] && vllm serve mistralai/Mistral-7B-Instruct-v0.3 --grpc --port 50051 --trust-remote-code"
  ]
  ports:
  - containerPort: 50051
  livenessProbe:
    grpc:
      port: 50051
    initialDelaySeconds: 120
    periodSeconds: 10
  readinessProbe:
    grpc:
      port: 50051
    initialDelaySeconds: 120
    periodSeconds: 5
```

!!! note
    gRPC 健康服务会在每次探测时检查引擎状态。如果引擎不健康或服务器正在关闭，探针将返回 `NOT_SERVING`。

您也可以使用 `grpcurl` 手动验证健康服务：

```bash
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check
```

## 故障排除

### 启动探针或就绪探针失败，容器日志包含 "KeyboardInterrupt: terminated"

如果启动探针或就绪探针的 failureThreshold 对于服务器启动所需时间来说太低，Kubernetes 调度程序将杀死容器。以下迹象表明发生了这种情况：

1. 容器日志包含 "KeyboardInterrupt: terminated"
2. `kubectl get events` 显示消息 `Container $NAME failed startup probe, will be restarted`

要缓解此问题，请增加 failureThreshold 以允许模型服务器有更多时间启动服务。您可以通过从清单中移除探针，然后测量模型服务器显示就绪所需的时间来确定理想的 failureThreshold。

## 总结

使用 Kubernetes 部署 vLLM 能够有效扩缩容和管理利用 GPU 资源的 ML 模型。通过遵循上述步骤，您应该能够在 Kubernetes 集群中设置和测试 vLLM 部署。如果您遇到任何问题或有任何建议，请随时为文档做出贡献。
