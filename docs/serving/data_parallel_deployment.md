# 数据并行部署

vLLM 支持数据并行部署，其中模型权重在单独的实例/GPU 之间复制，以处理独立的请求批次。

这适用于密集模型和 MoE 模型。

对于 MoE 模型，特别是像 DeepSeek 这样采用 MLA（多头潜在注意力）的模型，对注意力层使用数据并行，对专家层使用专家并行或张量并行（EP 或 TP）可能是有利的。

在这些情况下，数据并行秩不是完全独立的。前向传递必须对齐，并且所有秩上的专家层需要在每次前向传递期间同步，即使待处理的请求数量少于 DP 秩的数量。

默认情况下，专家层形成一个大小为 `DP × TP` 的张量并行组。要改用专家并行，请包含 `--enable-expert-parallel` CLI 参数（在多节点情况下，在所有节点上）。有关启用 EP 后注意力和专家层行为的详细信息，请参见[专家并行部署](expert_parallel_deployment.md)。

在 vLLM 中，每个 DP 秩作为一个独立的"核心引擎"进程部署，通过 ZMQ 套接字与前端进程通信。数据并行注意力可以与张量并行注意力结合使用，在这种情况下，每个 DP 引擎拥有多个等于配置的 TP 大小的按 GPU 工作进程。

对于 MoE 模型，当任何秩中有请求正在处理时，我们必须确保在所有当前没有调度请求的秩中执行空的"虚拟"前向传递。这是通过一个单独的 DP 协调器进程与所有秩通信，以及每 N 步执行一次集合操作来确定所有秩何时变为空闲并可以暂停来处理的。当 TP 与 DP 结合使用时，专家层形成一个大小为 `DP × TP` 的组（默认使用张量并行，如果设置了 `--enable-expert-parallel`，则使用专家并行）。

在所有情况下，在 DP 秩之间负载均衡请求都是有益的。对于在线部署，可以通过考虑每个 DP 引擎的状态（特别是其当前调度和等待（排队）的请求以及 KV 缓存状态）来优化这种平衡。每个 DP 引擎具有独立的 KV 缓存，并且可以通过智能地引导提示来最大化前缀缓存的好处。

本文档侧重于在线部署（使用 API 服务器）。DP + EP 也支持离线使用（通过 LLM 类），示例请参见 [examples/features/data_parallel/data_parallel_offline.py](../../examples/features/data_parallel/data_parallel_offline.py)。

在线部署支持两种不同的模式——带有内部负载均衡的自包含模式，或外部按秩进程部署和负载均衡模式。

## 内部负载均衡

vLLM 支持"自包含"的数据并行部署，暴露单个 API 端点。

可以通过在 `vllm serve` 命令行参数中简单地包含例如 `--data-parallel-size=4` 来配置。这将需要 4 个 GPU。它可以与张量并行结合使用，例如 `--data-parallel-size=4 --tensor-parallel-size=2`，这将需要 8 个 GPU。在确定 DP 部署的大小时，请记住 `--max-num-seqs` 适用于每个 DP 秩。

跨多个节点运行单个数据并行部署需要在每个节点上运行不同的 `vllm serve`，指定该节点上应运行的 DP 秩。在这种情况下，仍将有一个单一的 HTTP 入口点——API 服务器将只在一个节点上运行，但不一定需要与 DP 秩位于同一位置。

这将在单个 8-GPU 节点上运行 DP=4、TP=2：

```bash
vllm serve $MODEL --data-parallel-size 4 --tensor-parallel-size 2
```

这将运行 DP=4，DP 秩 0 和 1 在头节点上，秩 2 和 3 在第二个节点上：

```bash
# 节点 0（IP 地址 10.99.48.128）
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# 节点 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-start-rank 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

这将运行 DP=4，仅 API 服务器在第一个节点上，所有引擎在第二个节点上：

```bash
# 节点 0（IP 地址 10.99.48.128）
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 0 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# 节点 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 4 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

此 DP 模式也可以通过指定 `--data-parallel-backend=ray` 与 Ray 一起使用：

```bash
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-backend=ray
```

使用 Ray 时有几个显著的区别：

- 只需要一个启动命令（在任何节点上）即可启动所有本地和远程 DP 秩，因此比在每个节点上启动更方便
- 无需指定 `--data-parallel-address`，运行命令的节点被用作 `--data-parallel-address`
- 无需指定 `--data-parallel-rpc-port`
- 当单个 DP 组需要多个节点时，*例如*，单个模型副本需要在至少两个节点上运行时，请确保设置 `VLLM_RAY_DP_PACK_STRATEGY="span"`，此时 `--data-parallel-size-local` 将被忽略并自动确定
- 远程 DP 秩将根据 Ray 集群的节点资源进行分配

目前，内部 DP 负载均衡在 API 服务器进程中完成，基于每个引擎中的运行和等待队列。未来可以通过引入 KV 缓存感知逻辑使其更加复杂。

使用此方法部署大型 DP 时，API 服务器进程可能成为瓶颈。在这种情况下，可以使用正交的 `--api-server-count` 命令行选项来扩展它（例如 `--api-server-count=4`）。这对用户是透明的——仍然暴露单个 HTTP 端点/端口。请注意，此 API 服务器扩展是"内部的"，仍然局限于"头"节点。

<figure markdown="1">
![DP 内部负载均衡图](../assets/deployment/dp_internal_lb.png)
</figure>

## 混合负载均衡

混合负载均衡介于内部和外部方法之间。每个节点运行自己的 API 服务器，仅将请求排队到同一节点上的数据并行引擎。上游负载均衡器（例如，入口控制器或流量路由器）将用户请求分发到这些按节点端点。

使用 `--data-parallel-hybrid-lb` 启用此模式，同时仍然使用全局数据并行大小启动每个节点。与内部负载均衡的主要区别在于：

- 你必须提供 `--data-parallel-size-local` 和 `--data-parallel-start-rank`，以便每个节点知道它拥有哪些秩。
- 与 `--headless` 不兼容，因为每个节点都暴露一个 API 端点。
- 根据本地秩的数量，在每个节点上扩展 `--api-server-count`

在此配置中，每个节点将调度决策保持本地，从而减少跨节点流量并避免在更大 DP 规模下出现单节点瓶颈。

## 外部负载均衡

特别是对于更大规模的部署，外部处理数据并行秩的编排和负载均衡是有意义的。

在这种情况下，将每个 DP 秩视为一个单独的 vLLM 部署（具有自己的端点）更方便，并由外部路由器在它们之间均衡 HTTP 请求，利用每个服务器的适当实时遥测数据进行路由决策。

对于非 MoE 模型，这已经可以轻松实现，因为每个部署的服务器是完全独立的。在这种情况下，启动独立的 vLLM 实例，无需任何 `--data-parallel-*` 参数；仅在 MoE 部署中支持外部 DP CLI 选项。

我们支持 MoE DP+EP 的等效拓扑结构，可以通过以下 CLI 参数进行配置。

如果 DP 秩位于同一位置（同一节点/IP 地址），则使用默认的 RPC 端口，但必须为每个秩指定不同的 HTTP 服务器端口：

```bash
# 秩 0
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                                         --port 8000
# 秩 1
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                                         --port 8001
```

对于多节点情况，还必须指定秩 0 的地址/端口：

```bash
# 秩 0（IP 地址 10.99.48.128）
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# 秩 1
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

协调器进程也在此场景中运行，与 DP 秩 0 引擎位于同一位置。

<figure markdown="1">
![DP 外部负载均衡图](../assets/deployment/dp_external_lb.png)
</figure>

在上图中，每个虚线框对应一个单独的 `vllm serve` 启动——例如，这些可以是单独的 Kubernetes Pod。
