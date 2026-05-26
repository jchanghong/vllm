# 使用 Nginx

本文档展示了如何启动多个 vLLM 服务容器，并使用 Nginx 作为这些服务器之间的负载均衡器。

## 构建 Nginx 容器

本指南假设您刚刚克隆了 vLLM 项目，并且当前位于 vllm 根目录。

```bash
export vllm_root=`pwd`
```

创建一个名为 `Dockerfile.nginx` 的文件：

```dockerfile
FROM nginx:latest
RUN rm /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

构建容器：

```bash
docker build . -f Dockerfile.nginx --tag nginx-lb
```

## 创建简单的 Nginx 配置文件

创建一个名为 `nginx_conf/nginx.conf` 的文件。注意，您可以添加任意数量的服务器。在下面的示例中，我们从两个服务器开始。要添加更多服务器，请在 `upstream backend` 中添加另一个 `server vllmN:8000 max_fails=3 fail_timeout=10000s;` 条目。

??? console "配置"

    ```console
    upstream backend {
        least_conn;
        server vllm0:8000 max_fails=3 fail_timeout=10000s;
        server vllm1:8000 max_fails=3 fail_timeout=10000s;
    }
    server {
        listen 80;
        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
    ```

## 构建 vLLM 容器

```bash
cd $vllm_root
docker build -f docker/Dockerfile . --tag vllm
```

如果您在代理后面，可以将代理设置传递给 docker build 命令，如下所示：

```bash
cd $vllm_root
docker build \
    -f docker/Dockerfile . \
    --tag vllm \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy
```

## 创建 Docker 网络

```bash
docker network create vllm_nginx
```

## 启动 vLLM 容器

注意事项：

- 如果您的 HuggingFace 模型缓存在其他位置，请更新下面的 `hf_cache_dir`。
- 如果您没有现有的 HuggingFace 缓存，建议先启动 `vllm0` 并等待模型完成下载和服务器就绪。这样可以确保 `vllm1` 能够利用您刚刚下载的模型，而无需再次下载。
- 下面的示例假设使用 GPU 后端。如果您使用 CPU 后端，请移除 `--gpus device=ID`，并在 docker run 命令中添加 `VLLM_CPU_KVCACHE_SPACE` 和 `VLLM_CPU_OMP_THREADS_BIND` 环境变量。
- 如果您不想使用 `Llama-2-7b-chat-hf`，请调整您想在 vLLM 服务器中使用的模型名称。

??? console "命令"

    ```console
    mkdir -p ~/.cache/huggingface/hub/
    hf_cache_dir=~/.cache/huggingface/
    docker run \
        -itd \
        --ipc host \
        --network vllm_nginx \
        --gpus device=0 \
        --shm-size=10.24gb \
        -v $hf_cache_dir:/root/.cache/huggingface/ \
        -p 8081:8000 \
        --name vllm0 vllm \
        --model meta-llama/Llama-2-7b-chat-hf
    docker run \
        -itd \
        --ipc host \
        --network vllm_nginx \
        --gpus device=1 \
        --shm-size=10.24gb \
        -v $hf_cache_dir:/root/.cache/huggingface/ \
        -p 8082:8000 \
        --name vllm1 vllm \
        --model meta-llama/Llama-2-7b-chat-hf
    ```

!!! note
    如果您在代理后面，可以通过 `-e http_proxy=$http_proxy -e https_proxy=$https_proxy` 将代理设置传递给 docker run 命令。

## 启动 Nginx

```bash
docker run \
    -itd \
    -p 8000:80 \
    --network vllm_nginx \
    -v ./nginx_conf/:/etc/nginx/conf.d/ \
    --name nginx-lb nginx-lb:latest
```

## 验证 vLLM 服务器已就绪

```bash
docker logs vllm0 | grep Uvicorn
docker logs vllm1 | grep Uvicorn
```

两个输出都应该如下所示：

```console
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
