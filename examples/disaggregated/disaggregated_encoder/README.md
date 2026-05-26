# 分离式编码器

这些示例脚本演示了 vLLM 的分离式编码器（EPD）功能。

有关 EPD 功能的详细说明，请参阅[分离式编码器功能文档](../../../docs/features/disagg_encoder.md)。

## 文件

- `disagg_epd_proxy.py` - 代理脚本，演示 XeYpZd 设置（X 个编码实例、Y 个预填充实例、Z 个解码实例）。目前在 1e1p1d 配置下稳定运行。

- `disagg_1e1p1d_example.sh` - 设置 1e1p1d 配置，运行 VisionArena 基准测试，并使用本地图像处理单个请求。

- `disagg_1e1pd_example.sh` - 设置 1e1pd 配置，运行 VisionArena 基准测试，并使用本地图像处理单个请求。

### 自定义配置

```bash
# 使用特定的 GPU
GPU_E=0 GPU_PD=1 GPU_P=1 GPU_D=2 bash disagg_1e1p1d_example.sh

# 使用特定的端口
ENDPOINT_PORT=10001 bash disagg_1e1p1d_example.sh

# 使用特定的模型
MODEL="Qwen/Qwen2.5-VL-3B-Instruct" bash disagg_1e1p1d_example.sh

# 使用特定的存储路径
EC_SHARED_STORAGE_PATH="/tmp/my_ec_cache" bash disagg_1e1p1d_example.sh

# 在 XPU 上运行；脚本从 CUDA_VISIBLE_DEVICES 切换到 ZE_AFFINITY_MASK
DEVICE_PLATFORM=xpu GPU_E=0 GPU_PD=1 bash disagg_1e1pd_example.sh
```

`DEVICE_PLATFORM` 默认为 `cuda`。在 Intel GPU 上运行这些示例时，设置 `DEVICE_PLATFORM=xpu`，以便脚本使用 `ZE_AFFINITY_MASK` 代替 `CUDA_VISIBLE_DEVICES` 进行设备选择。

## 编码器实例

编码器引擎应使用以下标志启动：

- `--enforce-eager` **（必需）** – 当前的 EPD 实现仅与以该模式运行的编码器实例兼容。

- `--no-enable-prefix-caching` **（必需）** – 编码器实例不消耗 KV 缓存；禁用前缀缓存以避免与其他功能冲突。

- `--max-num-batched-tokens=<大值>` **（默认值：2048）** – 此标志控制每个解码步骤的令牌调度预算，对纯编码器实例无关。**将其设置为一个非常大的值（实际上无限制）以绕过调度器限制。** 实际的令牌预算由编码器缓存管理器管理。

- `--mm-encoder-only` **（可选）** – 如果可能，在初始化时跳过语言模型以减少设备内存使用。

## 本地媒体输入

要支持本地图像输入（来自你的 `MEDIA_PATH` 目录），请将以下标志添加到编码器实例：

```bash
--allowed-local-media-path $MEDIA_PATH
```

vllm 实例和 `disagg_encoder_proxy` 支持使用 ```{"url": "file://'"$MEDIA_PATH_FILENAME"'}``` 作为多模态输入的本地 URI。每个 URI 从 `disagg_encoder_proxy` 不变地传递给编码器实例，以便编码器可以本地加载媒体。

## EC 连接器和 KV 传输

`ECExampleConnector` 用于将编码器缓存存储在本地磁盘上并便于传输。要启用编码器分离功能，请添加以下配置：

```bash
# 添加到编码器实例：
--ec-transfer-config '{
    "ec_connector": "ECExampleConnector",
    "ec_role": "ec_producer",
    "ec_connector_extra_config": {
        "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
    }
}'

# 添加到预填充/预填充+解码实例：
--ec-transfer-config '{
    "ec_connector": "ECExampleConnector",
    "ec_role": "ec_consumer",
    "ec_connector_extra_config": {
        "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
    }
}'
```

`$EC_SHARED_STORAGE_PATH` 是 EC 连接器临时存储缓存的路径。

如果启用预填充实例（未禁用 `--prefill-servers-urls`），则需要 `--kv-transfer-config` 来支持 PD 分离。目前，我们使用 `NixlConnector` 来实现此目的。请参阅 `tests/v1/kv_connector/nixl_integration` 获取更多关于使用 Nixl 进行 PD 分离的示例代码。

```bash
# 添加到预填充实例：
--kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_producer"
}'

# 添加到解码实例：
--kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_consumer"
}'
```

## 代理实例标志（`disagg_epd_proxy.py`）

| 标志 | 描述 |
| ---- | ----------- |
| `--encode-servers-urls` | 编码器端点的逗号分隔列表。从请求中提取的每个多模态项以轮询方式分发到其中一个 URL。 |
| `--prefill-servers-urls` | 预填充端点的逗号分隔列表。设置为 `disable`、`none` 或 `""` 以跳过专门的预填充阶段并运行 E+PD（编码器 + 组合预填充/解码）。 |
| `--decode-servers-urls` | 解码端点的逗号分隔列表。非流和流路径均在列表中轮询分发。 |
| `--host`、`--port` | 代理本身的绑定地址（默认值：`0.0.0.0:8000`）。 |

使用示例：
E + PD 设置：

```bash
$ python disagg_encoder_proxy.py \
      --encode-servers-urls "http://e1:8001,http://e2:8002" \
      --prefill-servers-urls "disable" \
      --decode-servers-urls "http://pd1:8003,http://pd2:8004"
```

E + P + D 设置：

```bash
$ python disagg_encoder_proxy.py \
      --encode-servers-urls "http://e1:8001,http://e2:8001" \
      --prefill-servers-urls "http://p1:8003,http://p2:8004" \
      --decode-servers-urls "http://d1:8005,http://d2:8006"
```
