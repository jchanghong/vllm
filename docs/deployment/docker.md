---
toc_depth: 2
---

# 使用 Docker

## 预构建镜像

--8<-- "docs/getting_started/installation/gpu.md:pre-built-images"

## 以非 root 用户运行

为了向后兼容，CUDA `vllm/vllm-openai` 镜像默认以 root 用户运行。它也准备好以内置的 `vllm` 用户（UID 2000，GID 0）运行：

```bash
docker run --rm --gpus all \
    --user 2000:0 \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    meta-llama/Llama-3.1-8B-Instruct
```

当为非 root 容器挂载模型或缓存卷时，请将可写路径挂载到 `/home/vllm` 下，而不是 `/root`。例如，将 Hugging Face 缓存挂载到 `/home/vllm/.cache/huggingface`，并确保挂载的目录对组 0 可写。

```bash
docker run --rm --gpus all \
    --user 2000:0 \
    -v ~/.cache/huggingface:/home/vllm/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    meta-llama/Llama-3.1-8B-Instruct
```

要构建默认以非 root `vllm` 用户运行的镜像，请使用 opt-in 的 `vllm-openai-nonroot` 目标：

```bash
docker build --target vllm-openai-nonroot \
    -t vllm-openai-nonroot:local \
    -f docker/Dockerfile .

docker run --rm --gpus all \
    -p 8000:8000 \
    vllm-openai-nonroot:local \
    meta-llama/Llama-3.1-8B-Instruct
```

`vllm-openai-nonroot` 目标还支持 OpenShift 风格的任意 UID，只要运行时 UID 是组 0 的成员即可。在 Kubernetes 清单中，相应地设置容器安全上下文，并保持挂载的缓存/模型路径对组 0 可写：

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000540000
  runAsGroup: 0
  fsGroup: 0
```

不在组 0 内的运行时 UID 不在文档化的支持矩阵之内，因为它们可能无法写入 `/home/vllm` 或 `/opt/uv/cache`。

## 从源码构建镜像

--8<-- "docs/getting_started/installation/gpu.md:build-image-from-source"
