# LMCache 示例

此文件夹演示如何使用 LMCache 进行分离式预填充、CPU 卸载和 KV 缓存共享。

## 1. vLLM v1 中的分离式预填充

本示例演示如何使用 NIXL 在单节点上运行 LMCache 进行分离式预填充。

### 前提条件

- 安装 [LMCache](https://github.com/LMCache/LMCache)。可以直接运行 `pip install lmcache`。
- 安装 [NIXL](https://github.com/ai-dynamo/nixl)。
- 至少 2 块 GPU
- 有效的 Hugging Face 令牌（HF_TOKEN），用于 Llama 3.1 8B Instruct。

### 使用方法

运行
`cd disagg_prefill_lmcache_v1`
进入 `disagg_prefill_lmcache_v1` 文件夹，然后运行

```bash
bash disagg_example_nixl.sh
```

即可执行分离式预填充并对性能进行基准测试。

### 组件

#### 服务器脚本

- `disagg_prefill_lmcache_v1/disagg_vllm_launcher.sh` - 启动用于预填充/解码的各个 vLLM 服务器，同时启动代理服务器。
- `disagg_prefill_lmcache_v1/disagg_proxy_server.py` - FastAPI 代理服务器，协调预填充器与解码器之间的通信
- `disagg_prefill_lmcache_v1/disagg_example_nixl.sh` - 运行示例的主脚本

#### 配置

- `disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml` - 预填充服务器的配置
- `disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml` - 解码服务器的配置

#### 日志文件

主脚本会生成多个日志文件：

- `prefiller.log` - 预填充服务器的日志
- `decoder.log` - 解码服务器的日志
- `proxy.log` - 代理服务器的日志

## 2. CPU 卸载示例

- `python cpu_offload_lmcache.py -v v0` - 适用于 vLLM v0 的 CPU 卸载实现
- `python cpu_offload_lmcache.py -v v1` - 适用于 vLLM v1 的 CPU 卸载实现

## 3. KV 缓存共享

`kv_cache_sharing_lmcache_v1.py` 示例演示如何在 vLLM v1 实例之间共享 KV 缓存。

## 4. vLLM v0 中的分离式预填充

`disaggregated_prefill_lmcache_v0.py` 提供了如何在 vLLM v0 中运行分离式预填充的示例。
