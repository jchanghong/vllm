# P2P NCCL 连接器

一种基于点对点通信、具有动态扩展能力的 xPyD 实现，部分灵感来自 Dynamo。

## 详细设计

### 整体流程

如图 1 所示，此 **PD 分离** 解决方案的整体流程通过一个请求流程来描述：

1. 客户端向代理/路由器的 `/v1/completions` 接口发送 HTTP 请求。
2. 代理/路由器通过轮询或随机选择的方式选择一个 **1P1D（1 个 Prefill 实例 + 1 个 Decode 实例）**，生成一个 `request_id`（规则将在后面介绍），将 HTTP 请求消息中的 `max_tokens` 修改为 **1**，然后将请求转发给 **P 实例**。
3. 紧接着，代理/路由器将**原始 HTTP 请求**转发给 **D 实例**。
4. **P 实例**执行 **Prefill**，然后**主动将生成的 KV 缓存**发送给 D 实例（使用 **PUT_ASYNC** 模式）。D 实例的 `zmq_addr` 可以通过 `request_id` 解析。
5. **D 实例**有一个**专用线程**用于接收 KV 缓存（以避免阻塞主进程）。接收到的 KV 缓存保存到 **GPU 内存缓冲区**中，其大小由 vLLM 启动参数 `kv_buffer_size` 决定。当 GPU 缓冲区满时，KV 缓存存储在**本地 Tensor 内存池**中。
6. 在 **Decode** 期间，D 实例的主进程从 **GPU 缓冲区**或**内存池**中获取 KV 缓存（由 P 实例传输），从而**跳过 Prefill**。
7. 完成 **Decode** 后，D 实例将结果返回给**代理/路由器**，然后代理/路由器将其转发给**客户端**。

![image1](https://github.com/user-attachments/assets/fb01bde6-755b-49f7-ad45-48a94b1e10a7)

### 代理/路由器（演示）

一个简单的 HTTP 服务，作为客户端请求的入口点，并启动一个后台线程监听 P/D 实例报告其 HTTP IP 和 PORT，以及 ZMQ IP 和 PORT。它维护一个 `http_addr -> zmq_addr` 的字典。`http_addr` 是 vLLM 实例请求的 IP:PORT，而 `zmq_addr` 是 KV 缓存握手和元数据接收的地址。

代理/路由器负责根据客户端请求的特征（如 prompt）选择 1P1D，并生成对应的 `request_id`，例如：

```text
cmpl-___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0
```

目前，为了快速验证 xPyD 是否能工作，使用轮询方式选择 1P1D。未来计划使用 trie 结合实例的负载状态来选择适当的 P 和 D。

每个 P/D 实例定期向代理/路由器发送心跳包（目前每 3 秒一次）以注册（即报告 `http_addr -> zmq_addr`）并保持连接存活。如果某个实例崩溃并在一段时间内未能发送 ping，代理/路由器将移除超时的实例（此功能尚未开发）。

### KV 缓存传输方式

KVCache 传输有三种方式：PUT、GET 和 PUT_ASYNC。这些方式可以通过 `--kv-transfer-config` 和 `kv_connector_extra_config` 参数指定，具体通过 `send_type` 字段。PUT 和 PUT_ASYNC 都是由 P 实例主动向 D 实例发送 KVCache。区别在于 PUT 是同步传输方式，会阻塞主进程，而 PUT_ASYNC 是异步传输方式。PUT_ASYNC 使用专用线程发送 KVCache，因此不会阻塞主进程。相比之下，GET 方式是 P 实例在计算完 prefill 后将 KVCache 保存到内存缓冲区，然后 D 实例在为 KVCache 分配好空间后主动从 P 实例获取计算好的 KVCache。

实验结果表明，这些方式的性能从高到低依次为：PUT_ASYNC → GET → PUT。

### 通过 ZMQ & NCCL 的 P2P 通信

只要知道对端的地址，就可以进行点对点的 KV 缓存传输（使用 NCCL），不受 rank 和 world size 的约束。这是为了支持 PD 分离下实例的动态扩展（扩缩容）。这意味着添加或移除 P/D 实例不需要完全重启系统。

每个 P/D 实例只需要创建一个 `P2pNcclEngine` 实例。该实例维护一个 ZMQ 服务器，运行一个专用线程监听 `zmq_addr` 地址，并接收来自其他实例的控制流请求。这些请求包括建立 NCCL 连接的请求和发送 KVCache 元数据（如张量形状和数据类型）的请求。但它实际上并不传输 KVCache 数据本身。

当 P 实例和 D 实例首次传输 KVCache 时，它们需要建立 ZMQ 连接和 NCCL 组。对于后续的 KVCache 传输，此 ZMQ 连接和 NCCL 组被重用。NCCL 组仅包含两个 rank，即 world size 等于 2。此设计旨在支持动态扩展，这意味着添加或移除 P/D 实例不需要完全重启系统。只要知道对端的地址，就可以进行点对点的 KVCache 传输，不受 rank 或 world size 的限制。

### NCCL 组拓扑

目前，只支持对称 TP（张量并行）方式进行 KVCache 传输。不对称 TP 和 PP（流水线并行）方式将在未来支持。图 2 展示了 1P2D 设置，其中每个实例的 TP（张量并行）度为 2。总共有 7 个 NCCL 组：三个 vLLM 实例各自有一个 TP=2 的 NCCL 组。此外，P 实例的第 0 块 GPU 卡与每个 D 实例的第 0 块 GPU 卡建立一个 NCCL 组。类似地，P 实例的第 1 块 GPU 卡与每个 D 实例的第 1 块 GPU 卡建立一个 NCCL 组。

![image2](https://github.com/user-attachments/assets/837e61d6-365e-4cbf-8640-6dd7ab295b36)

每个 NCCL 组会占用一定量的 GPU 内存缓冲区用于通信，其大小主要受 `NCCL_MAX_NCHANNELS` 环境变量影响。当 `NCCL_MAX_NCHANNELS=16` 时，一个 NCCL 组通常占用 100MB，而当 `NCCL_MAX_NCHANNELS=8` 时，通常占用 52MB。对于大型 xPyD 配置（如 DeepSeek 的 96P144D），这种实现目前不可行。未来，我们正在考虑使用 RDMA 进行点对点通信，同时也在关注 UCCL。

### GPU 内存缓冲区和 Tensor 内存池

内存缓冲区大小存在权衡：对于 P 实例，在 PUT 和 PUT_ASYNC 模式下不需要内存缓冲区，但在 GET 模式下是必需的。对于 D 实例，在所有三种模式下都需要内存缓冲区。D 实例的内存缓冲区不应太大。同样，对于 GET 模式下的 P 实例，内存缓冲区也不应太大。D 实例的内存缓冲区用于临时存储 P 实例发送的 KVCache。如果过大，将减少 D 实例可用于正常推理的 KVCache 空间，从而降低推理批次大小，最终导致输出吞吐量下降。内存缓冲区的大小由参数 `kv_buffer_size` 配置，以字节为单位，通常设置为内存大小的 5%～10%。

如果 P 实例的 `--max-num-seqs` 参数设置得很大，由于批次较大，P 实例将同时生成大量 KVCache。这可能超出 D 实例内存缓冲区的容量，导致 KVCache 丢失。一旦 KVCache 丢失，D 实例需要重新计算 Prefill，相当于执行两次 Prefill。因此，首 token 延迟（TTFT）将显著增加，导致性能下降。

为了解决上述问题，我受 Linux 内存模块中伙伴系统的启发，设计并开发了一个用于存储 KVCache 的本地 Tensor 内存池。由于服务器上的内存通常足够大（通常在 TB 级别），因此无需考虑使用前缀缓存或基于块的设计来重用内存以节省空间。当内存缓冲区不足时，KVCache 可以直接存储在 Tensor 内存池中，D 实例随后可以从其中获取 KVCache。读写速度为 PCIe 速度，PCIe 4.0 约为 21 GB/s，通常比 Prefill 速度快。否则，像 Mooncake 和 lmcache 这样的解决方案就没有必要了。Tensor 内存池充当泄洪区，通常在突发流量激增之外不会使用。在最坏的情况下，我的解决方案的性能不比使用缓存存储的正常情况差。

## 安装 vLLM

```shell
pip install "vllm>=0.9.2"
```

## 运行 xPyD

### 说明

- 以下示例在 A800（80GB）设备上运行，使用 Meta-Llama-3.1-8B-Instruct 模型。
- 注意 `kv_buffer_size`（以字节为单位）的设置。经验值是 GPU 内存大小的 10%。这与 kvcache 大小有关。如果太小，用于临时存储接收到的 kvcache 的 GPU 内存缓冲区将溢出，导致 kvcache 存储在 tensor 内存池中，增加延迟。如果太大，可用于推理的 kvcache 空间将减少，导致批次大小减小和吞吐量下降。
- 对于 Prefill 实例，当使用非 GET 模式时，`kv_buffer_size` 可以设置为 1，因为 Prefill 目前不需要接收 kvcache。然而，当使用 GET 模式时，需要更大的 `kv_buffer_size`，因为它需要存储发送给 D 实例的 kvcache。
- 您可能需要修改以下命令中的 `kv_buffer_size` 和 `port`（如果有冲突）。
- `PUT_ASYNC` 提供最佳性能，应优先使用。
- `--port` 必须与 `--kv-transfer-config` 中的 `http_port` 一致。
- `disagg_proxy_p2p_nccl_xpyd.py` 脚本将使用端口 10001（用于接收客户端请求）和端口 30001（用于接收 P 和 D 实例的服务发现）。
- 运行代理的节点必须安装 `quart`。
- 支持多节点；只需修改 `--kv-transfer-config` 中的 `proxy_ip` 和 `proxy_port`。
- 在以下示例中，假设**代理的 IP 为 10.0.1.1**。

### 运行 1P3D

#### 代理（例如 10.0.1.1）

```shell
cd {your vllm directory}/examples/disaggregated/p2p_nccl_xpyd/
python3 disagg_proxy_p2p_nccl_xpyd.py &
```

#### Prefill1（例如 10.0.1.2 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=0 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20001 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20001"}}' > /var/vllm.log 2>&1 &
    ```

#### Decode1（例如 10.0.1.3 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=1 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20002 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20002"}}' > /var/vllm.log 2>&1 &
    ```

#### Decode2（例如 10.0.1.4 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=2 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20003 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"23001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20003"}}' > /var/vllm.log 2>&1 &
    ```

#### Decode3（例如 10.0.1.5 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=3 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20004 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"24001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20004"}}' > /var/vllm.log 2>&1 &
    ```

### 运行 3P1D

#### 代理（例如 10.0.1.1）

```shell
cd {your vllm directory}/examples/disaggregated/p2p_nccl_xpyd/
python3 disagg_proxy_p2p_nccl_xpyd.py &
```

#### Prefill1（例如 10.0.1.2 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=0 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20001 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20001"}}' > /var/vllm.log 2>&1 &
    ```

#### Prefill2（例如 10.0.1.3 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=1 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20002 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20002"}}' > /var/vllm.log 2>&1 &
    ```

#### Prefill3（例如 10.0.1.4 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=2 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20003 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"23001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20003"}}' > /var/vllm.log 2>&1 &
    ```

#### Decode1（例如 10.0.1.5 或 10.0.1.1）

??? console "命令"

    ```shell
    CUDA_VISIBLE_DEVICES=3 vllm serve {your model directory} \
        --host 0.0.0.0 \
        --port 20004 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --served-model-name base_model \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"24001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20004"}}' > /var/vllm.log 2>&1 &
    ```

## 单个请求

```shell
curl -X POST -s http://10.0.1.1:10001/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "base_model",
    "prompt": "San Francisco is a",
    "max_tokens": 10,
    "temperature": 0
}'
```

## 基准测试

??? console "命令"

    ```shell
    vllm bench serve \
        --backend vllm \
        --model base_model \
        --tokenizer meta-llama/Llama-3.1-8B-Instruct \
        --dataset-name "random" \
        --host 10.0.1.1 \
        --port 10001 \
        --random-input-len 1024 \
        --random-output-len 1024 \
        --ignore-eos \
        --burstiness 100 \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        --metric-percentiles "90,95,99" \
        --seed $(date +%s) \
        --trust-remote-code \
        --request-rate 3 \
        --num-prompts 1000
    ```

## 关闭

```shell
pgrep python | xargs kill -9 && pkill -f python
```

## 测试数据

### **场景**：1K 输入 & 200 输出 token，E2E P99 延迟 ~2s

![testdata](https://github.com/user-attachments/assets/cef0953b-4567-4bf9-b940-405b92a28eb1)
