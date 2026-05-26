# 安全性

## 节点间通信

多节点 vLLM 部署中所有节点之间的通信**默认不安全**，必须通过将节点放置在隔离网络上来保护。这包括：

1. PyTorch 分布式通信
2. KV 缓存传输通信
3. 张量、流水线和数据并行通信

### 节点间通信的配置选项

以下选项控制 vLLM 中的节点间通信：

#### 1. **环境变量：**

- `VLLM_HOST_IP`：设置 vLLM 进程用于通信的 IP 地址

#### 2. **KV 缓存传输配置：**

- `--kv-ip`：KV 缓存传输通信的 IP 地址（默认：127.0.0.1）
- `--kv-port`：KV 缓存传输通信的端口（默认：14579）

#### 3. **数据并行配置：**

- `data_parallel_master_ip`：数据并行主节点的 IP（默认：127.0.0.1）
- `data_parallel_master_port`：数据并行主节点的端口（默认：29500）

### PyTorch 分布式注意事项

vLLM 使用 PyTorch 的分布式功能进行某些节点间通信。有关 PyTorch 分布式安全注意事项的详细信息，请参阅 [PyTorch 安全指南](https://github.com/pytorch/pytorch/security/policy#using-distributed-features)。

PyTorch 安全指南中的关键要点：

- PyTorch 分布式功能仅用于内部通信
- 它们不适合在不受信任的环境或网络中使用
- 出于性能原因，不包含任何授权协议
- 消息以未加密方式发送
- 连接来自任何地方均无检查

## 安全建议

### 1. **网络隔离：**

- 将 vLLM 节点部署在专用、隔离的网络上
- 使用网络分段以防止未经授权的访问
- 实施适当的防火墙规则

### 2. **配置最佳实践：**

- 始终将 `VLLM_HOST_IP` 设置为特定的 IP 地址，而不是使用默认值
- 配置防火墙，只允许节点之间的必要端口

### 3. **访问控制：**

- 限制对部署环境的物理和网络访问
- 为管理接口实施适当的身份验证和授权
- 对所有系统组件遵循最小权限原则

### 4. **限制媒体 URL 的域名访问：**

通过设置 `--allowed-media-domains` 限制 vLLM 可以访问的媒体 URL 域名，以防止服务端请求伪造（SSRF）攻击。（例如 `--allowed-media-domains upload.wikimedia.org github.com www.bogotobogo.com`）

此保护适用于在线服务 API（多模态输入）和**批处理运行器**（`vllm run-batch`），其中批处理转录/翻译请求中的 `file_url` 值会对照相同的允许列表进行验证。

如果没有域名限制，恶意用户可能提供以下 URL：

- **针对内部服务**：访问内部网络端点、云元数据服务（例如 `169.254.169.254`）或其他不打算公开访问的服务（SSRF）。
- **消耗过多资源**：指向极大的文件或慢速端点，导致服务器下载无限制的数据量，耗尽内存、磁盘或网络带宽。

通过显式地只允许您期望媒体来源的域名，您可以显著减少这类滥用行为的攻击面。

同时，考虑设置 `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0` 以防止 HTTP 重定向被跟随以绕过域名限制。

## 安全与防火墙：保护暴露的 vLLM 系统

虽然 vLLM 设计为允许将不安全的网络服务隔离到私有网络中，但某些组件（如依赖项和底层框架）可能会在所有网络接口上打开不安全服务监听，有时超出了 vLLM 的直接控制。

一个主要问题是 `torch.distributed` 的使用，vLLM 利用它进行分布式通信，包括在单主机上使用 vLLM 时。当 vLLM 使用 TCP 初始化（参见 [PyTorch TCP 初始化文档](https://docs.pytorch.org/docs/stable/distributed.html#tcp-initialization)）时，PyTorch 会创建一个 `TCPStore`，默认情况下监听所有网络接口。这意味着，除非采取额外的保护措施，否则任何可以通过任何网络接口访问您机器的主机都可能访问这些服务。

**从 PyTorch 的角度来看，任何使用 `torch.distributed` 的行为默认都应被视为不安全的。** 这是 PyTorch 团队已知且有意为之的行为。

### 防火墙配置指南

保护 vLLM 系统的最佳方法是仔细配置防火墙，只暴露最小的必要网络面。在大多数情况下，这意味着：

- **阻止所有传入连接，除了 API 服务器正在监听的 TCP 端口。**

- 确保用于内部通信的端口（如 `torch.distributed` 和 KV 缓存传输的端口）只能从受信任的主机或网络访问。

- 切勿将这些内部端口暴露给公共互联网或不受信任的网络。

请查阅您的操作系统或应用程序平台文档以获取具体的防火墙配置说明。

## API 密钥身份验证限制

### 概述

`--api-key` 标志（或 `VLLM_API_KEY` 环境变量）为 vLLM 的 HTTP 服务器提供身份验证，但**仅适用于 `/v1` 路径前缀下的 OpenAI 兼容 API 端点**，以及其他类似的 `/v2`、`/inference` 路径前缀。许多其他敏感端点在同一 HTTP 服务器上暴露，没有任何身份验证强制措施。

**重要提示：** 不要仅依赖 `--api-key` 来保护 vLLM 的访问安全。生产部署需要额外的安全措施。

### 受保护的端点（需要 API 密钥）

配置 `--api-key` 后，以下 `/v1` 端点需要 Bearer token 身份验证：

- `/v1/models` - 列出可用模型
- `/v1/chat/completions` - 聊天补全
- `/v1/chat/completions/batch` - 批量聊天补全
- `/v1/chat/completions/render` - 渲染聊天补全请求
- `/v1/completions` - 文本补全
- `/v1/completions/render` - 渲染补全请求
- `/v1/embeddings` - 生成嵌入
- `/v1/audio/transcriptions` - 音频转录
- `/v1/audio/translations` - 音频翻译
- `/v1/messages` - Anthropic 兼容消息 API
- `/v1/messages/count_tokens` - 对 Anthropic 消息进行令牌计数
- `/v1/responses` - 创建响应
- `/v1/responses/{response_id}` - 检索响应
- `/v1/responses/{response_id}/cancel` - 取消响应
- `/v1/score` - 评分 API
- `/v1/rerank` - 重排序 API
- `/v1/load_lora_adapter` - 加载 LoRA 适配器（可以改变模型行为；仅在设置了 `--enable-lora` 且 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` 时可用）
- `/v1/unload_lora_adapter` - 卸载 LoRA 适配器（可以改变模型行为；仅在设置了 `--enable-lora` 且 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` 时可用）
- `/inference/v1/generate` - 生成补全
- `/v2/embed` - Cohere 嵌入 API
- `/v2/rerank` - Cohere 重排序 API

### 未受保护的端点（无需 API 密钥）

即使配置了 `--api-key`，以下端点**也不需要身份验证**：

**推理端点：**

- `/invocations` - SageMaker 兼容端点（路由到与 `/v1` 端点相同的推理功能）
- `/generative_scoring` - 生成式评分 API
- `/pooling` - 池化 API
- `/classify` - 分类 API
- `/score` - 评分 API（非 `/v1` 变体）
- `/rerank` - 重排序 API（非 `/v1` 变体）

**操作控制端点（仅在支持 `"generate"` 任务时）：**

- `/pause` - 暂停生成（导致拒绝服务）
- `/resume` - 恢复生成
- `/is_paused` - 检查生成是否暂停
- `/scale_elastic_ep` - 触发扩缩操作
- `/is_scaling_elastic_ep` - 检查扩缩是否进行中
- `/init_weight_transfer_engine` - 为 RLHF 初始化权重传输引擎
- `/update_weights` - 更新模型权重（可以改变模型行为）
- `/get_world_size` - 获取分布式世界大小
- `/abort_requests` - 中止正在进行的请求（仅在同时设置了 `--tokens-only` 时）

**实用端点：**

- `/tokenize` - 分词
- `/detokenize` - 解码标记
- `/health` - 健康检查
- `/ping` - SageMaker 健康检查
- `/version` - 版本信息
- `/load` - 服务器负载指标

**分词器信息端点（仅在设置了 `--enable-tokenizer-info-endpoint` 时）：**

此端点**仅当设置了 `--enable-tokenizer-info-endpoint` 标志时才可用**。它可能暴露敏感信息，如聊天模板和分词器配置：

- `/tokenizer_info` - 获取全面的分词器信息，包括聊天模板和配置

**开发端点（仅在 `VLLM_SERVER_DEV_MODE=1` 时）：**

这些端点**仅当环境变量 `VLLM_SERVER_DEV_MODE` 设置为 `1` 时才可用**。它们仅用于开发和调试目的，绝不应在生产环境中启用：

- `/server_info` - 获取详细的服务器配置
- `/reset_prefix_cache` - 重置前缀缓存（可能中断服务）
- `/reset_mm_cache` - 重置多模态缓存（可能中断服务）
- `/reset_encoder_cache` - 重置编码器缓存（可能中断服务）
- `/sleep` - 使引擎休眠（导致拒绝服务）
- `/wake_up` - 唤醒引擎
- `/is_sleeping` - 检查引擎是否在休眠
- `/collective_rpc` - 在引擎上执行任意 RPC 方法（极其危险）

**分析器端点（仅当通过 `--profiler-config` 启用分析时）：**

这些端点仅在启用分析时可用，应仅用于本地开发：

- `/start_profile` - 启动 PyTorch 分析器
- `/stop_profile` - 停止 PyTorch 分析器

**注意：** `/invocations` 端点尤其令人担忧，因为它提供对受保护 `/v1` 端点相同推理功能的未认证访问。

### 安全影响

能够访问 vLLM HTTP 服务器的攻击者可以：

1. **绕过身份验证**：通过使用非 `/v1` 端点（如 `/invocations`、`/inference/v1/generate`、`/generative_scoring`、`/pooling`、`/classify`、`/score` 或 `/rerank`）在没有凭据的情况下运行任意推理
2. **造成拒绝服务**：通过在没有令牌的情况下调用 `/pause`、`/scale_elastic_ep` 或 `/abort_requests`
3. **访问操作控制**：操纵服务器状态（例如，暂停生成、通过 `/update_weights` 更新模型权重）
4. **如果设置了 `--enable-tokenizer-info-endpoint`**：访问敏感的分词器配置（包括聊天模板），可能泄露提示工程策略或其他实现细节
5. **如果设置了 `VLLM_SERVER_DEV_MODE=1`**：通过 `/collective_rpc` 执行任意 RPC 命令、重置缓存、使引擎休眠以及访问详细的服务器配置

### 推荐的安全实践

#### 1. 最小化暴露的端点

**关键：** 切勿在生产环境中设置 `VLLM_SERVER_DEV_MODE=1`。开发端点暴露极其危险的功能，包括：

- 通过 `/collective_rpc` 执行任意 RPC
- 可能中断服务的缓存操作
- 详细的服务器配置披露

同样，切勿在生产环境中启用分析器端点。

**谨慎使用 `--enable-tokenizer-info-endpoint`：** 仅当您需要暴露分词器配置信息时才启用 `/tokenizer_info` 端点。此端点会泄露可能包含敏感实现细节或提示工程策略的聊天模板和分词器设置。

#### 2. 部署在反向代理后面

最有效的方法是将 vLLM 部署在反向代理（如 nginx、Envoy 或 Kubernetes Gateway）后面，该代理能够：

- 明确只允许列出您想向最终用户暴露的端点
- 阻止所有其他端点，包括未认证的推理和操作控制端点
- 在代理层实施额外的身份验证、速率限制和日志记录

## 请求参数资源限制

某些 API 请求参数可能对资源消耗产生重大影响，并可能被滥用以耗尽服务器资源。`/v1/completions` 和 `/v1/chat/completions` 端点中的 `n` 参数控制每个请求生成多少个独立的输出序列。非常大的值会导致引擎分配与 `n` 成比例的内存、CPU 和 GPU 时间，可能导致主机内存不足，并阻止服务器处理其他请求。

为了缓解此问题，vLLM 通过 `VLLM_MAX_N_SEQUENCES` 环境变量（默认值：**16384**）对 `n` 参数实施可配置的上限。超过此限制的请求在被引擎处理之前就会被拒绝。

### 建议

- **面向公众的部署：** 考虑将 `VLLM_MAX_N_SEQUENCES` 设置为适合您工作负载的值（例如 `64` 或 `128`），以限制单个请求的影响范围。
- **反向代理层：** 除了 vLLM 的内置限制外，考虑在反向代理上实施请求正文验证和速率限制，以进一步约束恶意负载。
- **监控：** 监控每个请求的资源消耗，以检测可能表明滥用行为的异常模式。

## 工具服务器和 MCP 安全

vLLM 支持通过 `--tool-server` 参数连接到外部工具服务器。这使得模型可以通过 Responses API（`/v1/responses`）调用工具。工具服务器支持适用于所有模型——不限于特定的模型架构。

**重要提示：** 默认情况下未启用任何工具服务器。必须通过配置显式选择加入。

### 内置演示工具（GPT-OSS）

传递 `--tool-server demo` 可启用内置演示工具，这些工具适用于任何支持工具调用的模型。工具实现不是 vLLM 的一部分——它们由单独安装的 [`gpt-oss`](https://github.com/openai/gpt-oss) 包提供。vLLM 提供委托给 `gpt-oss` 的薄包装器。

- **代码解释器**（`python`）：通过 Docker 执行的 Python（通过 `gpt_oss.tools.python_docker`）
- **Web 浏览器**（`browser`）：通过 Exa API 搜索，需要 `EXA_API_KEY`（通过 `gpt_oss.tools.simple_browser`）

#### 代码解释器（Python 工具）安全风险

代码解释器在 Docker 容器内执行模型生成的代码。然而，容器**默认未配置网络隔离**。它继承主机的 Docker 网络配置（例如，默认桥接网络或 `--network=host`），这意味着：

- 容器可能能够访问主机网络和局域网。
- 从容器可达的内部服务可能通过 SSRF（服务端请求伪造）被利用。
- 云元数据服务（例如 `169.254.169.254`）可能可访问。
- 如果从容器可以访问到有漏洞的内部服务（如 `torch.distributed` 端点），这可能会被用来攻击它们。

这一点尤其令人担忧，因为正在执行的代码是由模型生成的，而模型可能受到对抗性输入（提示注入）的影响。

#### 控制内置工具可用性

内置演示工具由两个设置控制：

1. **`--tool-server demo`**：启用内置演示工具（浏览器和 Python 代码解释器）。

2. **`VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS`**：当通过 Responses API 中的 `mcp` 工具类型请求内置工具时，这个以逗号分隔的允许列表控制允许哪些工具标签。有效值为：
   - `container` - 容器工具
   - `code_interpreter` - Python 代码执行工具
   - `web_search_preview` - Web 搜索/浏览器工具

   如果未设置此变量或为空，则不会启用任何通过 MCP 工具类型请求的内置工具。

要专门禁用 Python 代码解释器，请从 `VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS` 中省略 `code_interpreter`。

**考虑自定义实现**：GPT-OSS Python 工具是一个参考实现。对于生产部署，考虑实现具有更强隔离保证的自定义代码执行沙箱。请参阅 [GPT-OSS 文档](https://github.com/openai/gpt-oss?tab=readme-ov-file#python) 获取指导。

## 动态 LoRA 加载

vLLM 支持通过 `/v1/load_lora_adapter` 和 `/v1/unload_lora_adapter` API 端点动态加载和卸载 LoRA 适配器。此功能**默认不启用**——它需要同时设置 `--enable-lora` 和环境变量 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`。

**警告：** 动态 LoRA 加载不是安全操作，不应在暴露给不受信任客户端的部署中启用。如果您必须启用动态 LoRA 加载，请使用反向代理或网络级访问控制，仅将访问 `/v1/load_lora_adapter` 和 `/v1/unload_lora_adapter` 端点的权限限制给受信任的管理员。不要将这些端点暴露给最终用户。有关配置 LoRA 适配器的详细信息，请参阅 [LoRA 适配器文档](../features/lora.md)。

## 缓存目录安全

vLLM 假定其缓存目录是**私有且受信任的**。缓存内容在加载时不经过加密完整性验证，包括支持任意代码执行的格式。如果不受信任的用户或进程可以写入 vLLM 的缓存目录，他们可能能够使 vLLM 崩溃或使其执行任意代码。

**不要与不受信任的用户共享 vLLM 缓存目录，也不要从不受信任的存储挂载它们。** 对缓存目录的重视程度应与 vLLM 安装本身相同。

### 缓存目录配置

大多数缓存路径默认为单个根目录下的子目录。更改 `VLLM_CACHE_ROOT` 会更改所有继承自它的功能的默认位置。当启用 `torch.compile` 缓存时（默认启用），vLLM 还会将 `TRITON_CACHE_DIR` 重定向到此目录树。如果禁用编译缓存，Triton 将回退到其自己的默认位置（`~/.triton/cache`）。

| 环境变量 | 默认值 | 描述 |
| --- | --- | --- |
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | 基础缓存目录。如果设置了 `XDG_CACHE_HOME`，则遵循该设置。除非显式覆盖，以下所有路径都继承自此目录。 |
| *(torch.compile)* | `$VLLM_CACHE_ROOT/torch_compile_cache/` | 用于 AOT 编译模型、Inductor 图和 Triton 内核的编译缓存。由 `VLLM_DISABLE_COMPILE_CACHE` 控制（设置为 `1` 以禁用）。 |
| `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` | `$VLLM_CACHE_ROOT/flashinfer_autotune_cache/<flashinfer-version>/<arch>/<cache-hash>/` | FlashInfer 自动调优配置缓存。 |
| `VLLM_ASSETS_CACHE` | `$VLLM_CACHE_ROOT/assets/` | 下载的资源（例如，分词器文件）。 |
| `VLLM_XLA_CACHE_PATH` | `$VLLM_CACHE_ROOT/xla_cache/` | XLA/TPU 编译缓存。 |
| `VLLM_MEDIA_CACHE` | *（已禁用）* | 下载媒体的可选缓存（图像、视频、音频）。除非显式设置，否则不启用。 |

### 建议

- **限制文件权限**：限制 `VLLM_CACHE_ROOT`（以及依赖项使用的任何其他缓存目录，如禁用编译缓存时的 `~/.triton`）的文件权限，以便只有 vLLM 进程所有者可以读写。
- **不要从不受信任的来源复制缓存内容**：如果您在环境之间分发缓存工件，请确保它们来自受信任的构建流水线。
- **容器部署：** 如果要将缓存目录挂载到容器中，请确保卷源是受信任的。

## FIPS 兼容性

FIPS 合规性取决于许多因素，因此 vLLM 部署不会自动符合 FIPS 要求。最近的更改提高了 vLLM 在启用 FIPS 的主机上的*耐受性*——即在非批准算法被阻止时避免崩溃——但耐受性并不等同于合规性。部署是否满足 FIPS 要求取决于主机操作系统、为 Python 的 `hashlib` 和 `ssl` 模块提供支持的 OpenSSL 提供程序，以及安装了哪些可选依赖项。

### FIPS 相关配置

在启用 FIPS 的主机上运行 vLLM 的操作员应通过以下旋钮选择符合 FIPS 标准的算法：

- **多模态输入哈希**——`VLLM_MM_HASHER_ALGORITHM` 默认为 `blake3`，它不符合 FIPS 标准。在启用 FIPS 的环境中将其设置为 `sha256` 或 `sha512`。
- **前缀缓存哈希**——将 `--prefix-caching-hash-algo`（配置字段 `prefix_caching_hash_algo`）设置为 `sha256` 或 `sha256_cbor`。`xxhash` 和 `xxhash_cbor` 选项不符合 FIPS 标准。
- **TLS 密码套件**——使用 `--ssl-ciphers` 将 API 服务器的 TLS 握手限制为符合您环境策略的 FIPS 批准密码套件。

### 非安全 MD5 使用的自动回退

vLLM 在少数地方使用 MD5 来派生非安全缓存键（例如，配置哈希）。这些调用点传递了 `usedforsecurity=False`，并且当底层 OpenSSL 提供程序直接拒绝 MD5 时，还会回退到 SHA-256（参见 `vllm/utils/hashing.py` 中的 `safe_hash()`）。无需用户操作；记录此行为是为了让审计员和安全审阅者能够识别 MD5 引用并理解其用途。

### 提供非 FIPS 哈希实现的依赖项

某些依赖项暴露了非 FIPS 批准的哈希实现。vLLM 仅在选择相应算法时调用它们，但具有严格加密控制的操作员可能希望确保这些代码路径不被使用——并且在政策要求时，确保包本身不存在：

- `blake3`——当前列在 `requirements/common.txt` 中，因此标准安装会拉取它。它是延迟导入的，仅在 `VLLM_MM_HASHER_ALGORITHM=blake3`（默认值）时使用。将 `VLLM_MM_HASHER_ALGORITHM` 设置为 `sha256` 或 `sha512` 足以让非 FIPS 代码路径保持休眠状态。如果您的政策还禁止包本身存在，请在 `pip install` 后卸载它（`pip uninstall blake3`）；只要 `VLLM_MM_HASHER_ALGORITHM` 设置为非 blake3 值，vLLM 将继续正常运行。
- `xxhash`——一个真正的可选依赖项（不在 `requirements/common.txt` 中）。仅当选择了基于 `xxhash` 的前缀缓存算法时才会导入。保持不安装并选择基于 `sha256` 的前缀缓存算法。

### 哈希之外：其他 FIPS 考虑因素

哈希是 vLLM 具有显式 FIPS 感知代码的领域，但符合 FIPS 要求的部署取决于 vLLM 本身之外的几个因素。操作员应与其平台和安全团队一起评估以下内容：

- **主机加密提供程序。** Python 的 `hashlib` 和 `ssl` 模块仅在 Python 链接到由主机操作系统提供的经过 FIPS 验证的 OpenSSL（或等效）提供程序时才具有 FIPS 感知能力。vLLM 继承主机配置的任何提供程序——它不捆绑提供程序。
- **API 服务器 TLS。** 用于兼容 OpenAI 的 API 服务器的 TLS 终止通过 Python 的 `ssl` 模块使用主机的 OpenSSL。使用 `--ssl-ciphers` 限制密码套件以匹配您环境的 FIPS 政策，并确保服务器证书使用 FIPS 批准的算法和密钥大小签发。
- **出站 HTTPS。** 模型和资源下载（例如，通过 `huggingface_hub`）使用相同的主机 TLS 栈。相同的提供程序/密码考虑适用。
- **节点间通信默认未加密。** 如[节点间通信](#node-inter-communication)中所述，PyTorch 分布式、KV 缓存传输和数据并行通道不加密流量。要求传输中数据使用 FIPS 批准加密的 FIPS 环境必须在外部提供保护——例如，通过 mTLS sidecar 或由经过 FIPS 验证的模块终止的 IPsec——因为 vLLM 的内部通道本身无法满足此要求。仅网络隔离不是加密，也不满足"传输中数据使用 FIPS 批准加密"的要求，尽管它仍然是一种有用的纵深防御措施。
- **捆绑自己 OpenSSL 的依赖项。** 某些 Python wheel 静态链接的 OpenSSL 构建在启用 FIPS 的主机上无法通过内核 FIPS 自检（`FATAL FIPS SELFTEST FAILURE`）。`opencv-python-headless` 是一个已知示例；其他 manylinux wheel 可能行为类似。在排查 FIPS 启动失败时，审计已安装 wheel 中捆绑的加密库。
- **加速器和机器学习库。** PyTorch、CUDA、cuDNN、NCCL 和类似组件具有独立于 vLLM 的自身加密和 FIPS 状态。NVIDIA 为某些库发布了经过 FIPS 验证的构建；vLLM 不固定使用这些构建，因此选择并验证它们是操作员的责任。
- **vLLM 中不涉及 FIPS 的内容。** 用于令牌采样的随机数生成（Python/NumPy/PyTorch RNG）不是加密用途，不在 FIPS 范围内。Pickled 缓存工件是单独的[缓存目录安全](#cache-directory-security)中涵盖的安全问题。

简而言之：上述配置旋钮让 vLLM 避免使用非批准算法，而自动回退让它能够在启用 FIPS 的主机上运行而不会崩溃。然而，端到端的 FIPS 合规性是整个部署的属性——主机操作系统、加密提供程序、传递依赖项和网络架构——而不仅仅是 vLLM 本身。

## 报告安全漏洞

如果您认为在 vLLM 中发现了安全漏洞，请按照项目的安全政策进行报告。有关如何报告安全问题和项目安全政策的更多信息，请参阅 [vLLM 安全政策](https://github.com/vllm-project/vllm/blob/main/SECURITY.md)。
