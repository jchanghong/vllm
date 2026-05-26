# 休眠模式

vLLM 的休眠模式允许您在不停止服务器或卸载 Docker 容器的情况下，临时释放模型使用的大部分 GPU 内存，包括模型权重和 KV 缓存。这对于 RLHF、训练或需要在推理工作负载之间释放 GPU 资源的成本节约场景特别有用。

主要优势：

- **释放 GPU 内存**：将模型权重卸载到 CPU RAM 并丢弃 KV 缓存，释放高达 90%+ 的 GPU 内存供其他任务使用。
- **快速恢复**：快速唤醒引擎并恢复推理，无需完全重新加载模型。
- **API 端点**：通过 HTTP 端点或 Python API 控制休眠/唤醒状态。
- **支持分布式工作负载**：适用于张量并行、流水线并行等。
- **细粒度控制**：可选择仅唤醒模型权重或 KV 缓存，以避免在权重更新期间出现 OOM。

!!! note
    此功能现在支持 CUDA 和 ROCm 平台。

!!! note
    更多信息，请参阅此[博文](https://blog.vllm.ai/2025/10/26/sleep-mode.html)。

## 休眠级别

级别 1 休眠将卸载模型权重并丢弃 KV 缓存。KV 缓存的内容将被遗忘。级别 1 休眠适用于休眠和唤醒引擎以再次运行同一模型。模型权重备份在 CPU 内存中。请确保有足够的 CPU 内存来存储模型权重。级别 2 休眠将丢弃模型权重和 KV 缓存（而模型的缓冲区保留在 CPU 中，如 rope 缩放张量）。模型权重和 KV 缓存的内容都将被遗忘。级别 2 休眠适用于休眠和唤醒引擎以运行不同的模型或更新模型，其中先前的模型权重不再需要，例如 RLHF 权重更新。

## 用法

### 离线推理

通过向 `LLM` 类传递 `enable_sleep_mode=True` 来启用休眠模式。

```python
from vllm import LLM
llm = LLM("Qwen/Qwen3-0.6B", enable_sleep_mode=True)
```

#### Python API

```python
# 休眠级别 1
# 将引擎置于休眠状态（level=1：将权重卸载到 CPU RAM，丢弃 KV 缓存）
llm.sleep(level=1)

# 唤醒引擎（恢复权重）
llm.wake_up()
```

```python
# 休眠级别 2
# 将引擎置于休眠状态（level=2：丢弃权重和 KV 缓存）
llm.sleep(level=2)

# 仅重新分配权重内存
llm.wake_up(tags=["weights"])

# 原地加载权重
llm.collective_rpc("reload_weights")

# 重新分配 KV 缓存
llm.wake_up(tags=["kv_cache"])
```

#### RLHF 权重更新

在 RLHF 训练期间，vLLM 允许您使用 wake_up() 中的 tags 参数选择性地仅唤醒模型权重或 KV 缓存。这种细粒度控制在使用标签参数时特别有用：通过仅唤醒权重（例如，llm.wake_up(tags=["weights"])），您可以避免在权重更新完成之前为 KV 缓存分配内存。这种方法通过最小化权重同步和更新操作期间的峰值内存使用，帮助防止 GPU 内存不足（OOM）错误，尤其是在大型模型中。

使用 `tags=["weights"]` 或 `tags=["kv_cache"]` 控制恢复哪些资源，对 RLHF 和权重更新很有用。**请注意**，在所有组件都唤醒之前，`is_sleeping` 将报告 `true`。

```python
# 将引擎置于深度休眠（level=2）
llm.sleep(level=2)
# ... 获取新权重
# 仅唤醒权重以避免 OOM
llm.wake_up(tags=["weights"])
# ... 更新权重
# 权重更新后唤醒 KV 缓存
llm.wake_up(tags=["kv_cache"])
```

### 在线服务

要在 vLLM 服务器中启用休眠模式，您需要使用 `VLLM_SERVER_DEV_MODE=1` 标志初始化，并将 `--enable-sleep-mode` 传递给 vLLM 服务器。

#### 开发模式下的服务器

使用 `VLLM_SERVER_DEV_MODE=1` 标志时，您启用了开发端点，这些端点不应暴露给用户。

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \
  --enable-sleep-mode \
  --port 8000
```

以下是在级别 1 中休眠和唤醒模型的示例。

```bash
curl -X POST 'http://localhost:8000/sleep?level=1'
curl -X POST 'http://localhost:8000/wake_up'
```

以下是在级别 2 中休眠和唤醒模型的示例。

```bash
curl -X POST 'http://localhost:8000/sleep?level=2'
# 仅重新分配权重内存
curl -X POST 'http://localhost:8000/wake_up?tags=weights'
# 原地加载权重
curl -X POST 'http://localhost:8000/collective_rpc' -H 'Content-Type: application/json' -d '{"method":"reload_weights"}'
# 重新分配 KV 缓存
curl -X POST 'http://localhost:8000/wake_up?tags=kv_cache'
```

#### HTTP 端点

- `POST /sleep?level=1` — 将模型置于休眠状态（`level=1`）。
- `POST /wake_up` — 唤醒模型。支持可选的 `tags` 查询参数进行部分唤醒（例如，`?tags=weights`）。
- `POST /collective_rpc` — 执行集体远程过程调用（RPC）。
- `GET /is_sleeping` — 检查模型是否处于休眠状态。

!!! note
    这些端点仅在传递 `VLLM_SERVER_DEV_MODE=1` 时可用。

## 限制

在 ROCm 上，ROCm 的虚拟内存分配是通过分块内存分配完成的。您可以通过 `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE`（以 MB 为单位）控制块大小。默认值设置为 256MB。块越大，性能越快。但是，设置过大将导致 OOM。因此，如果在使用休眠模式时遇到 OOM，请尝试减小块大小。建议将块大小定义为 2 的幂。
