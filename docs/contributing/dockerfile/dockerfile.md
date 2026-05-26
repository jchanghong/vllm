# Dockerfile

我们提供了一个 [docker/Dockerfile](../../../docker/Dockerfile) 来构建用于运行 vLLM OpenAI 兼容服务器的镜像。
有关使用 Docker 部署的更多信息，请参见[此处](../../deployment/docker.md)。

以下是多阶段 Dockerfile 的可视化表示。构建图包含以下节点：

- 所有构建阶段
- 默认构建目标（以灰色高亮）
- 外部镜像（虚线边框）

构建图的边表示：

- `FROM ...` 依赖关系（实线，实心箭头）

- `COPY --from=...` 依赖关系（虚线，空心箭头）

- `RUN --mount=(.\*)from=...` 依赖关系（点线，空心菱形箭头）

  > <figure markdown="span">
  >   ![](../../assets/contributing/dockerfile-stages-dependency.png){ align="center" alt="query" width="100%" }
  > </figure>
  >
  > 使用工具：<https://github.com/patrickhoefler/dockerfilegraph>
  >
  > 重新生成构建图的命令（确保**从 vLLM 仓库的 \`root\` 目录**运行，其中 dockerfile 所在）：
  >
  > ```bash
  > dockerfilegraph \
  >   -o png \
  >   --legend \
  >   --dpi 200 \
  >   --max-label-length 50 \
  >   --filename docker/Dockerfile
  > ```
  >
  > 或者如果您想直接使用 Docker 镜像运行：
  >
  > ```bash
  > docker run \
  >    --rm \
  >    --user "$(id -u):$(id -g)" \
  >    --workdir /workspace \
  >    --volume "$(pwd)":/workspace \
  >    ghcr.io/patrickhoefler/dockerfilegraph:alpine \
  >    --output png \
  >    --dpi 200 \
  >    --max-label-length 50 \
  >    --filename docker/Dockerfile \
  >    --legend
  > ```
  >
  > （要针对不同文件运行，可以向 `--filename` 标志传递不同的参数。）
