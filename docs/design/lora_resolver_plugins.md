# LoRA 解析器插件

本目录包含基于 `LoRAResolver` 框架构建的 vLLM LoRA 解析器插件。
它们自动从指定的本地存储路径发现并加载 LoRA 适配器，无需手动配置或重启服务器。

## 概述

LoRA 解析器插件提供了一种在运行时动态加载 LoRA 适配器的灵活方式。当 vLLM
收到对尚未加载的 LoRA 适配器的请求时，解析器插件将尝试
从配置的存储位置定位并加载该适配器。这实现了：

- **动态 LoRA 加载**：无需重启服务器即可按需加载适配器
- **多种存储后端**：支持文件系统、S3 和自定义后端。内置的 `lora_filesystem_resolver` 需要本地存储路径，而内置的 `hf_hub_resolver` 将从 Huggingface Hub 拉取 LoRA 适配器并以相同方式运行。通常，可以实现自定义解析器从任何源获取。
- **自动发现**：与现有 LoRA 工作流无缝集成
- **可扩展部署**：跨多个 vLLM 实例的集中式适配器管理

## 先决条件

在使用 LoRA 解析器插件之前，请确保已配置以下环境变量：

### 必需的环境变量

1. **`VLLM_ALLOW_RUNTIME_LORA_UPDATING`**：必须设置为 `true` 或 `1` 以启用动态 LoRA 加载
   ```bash
   export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
   ```

2. **`VLLM_PLUGINS`**：必须包含所需的解析器插件（逗号分隔列表）
   ```bash
   export VLLM_PLUGINS=lora_filesystem_resolver
   ```

3. **`VLLM_LORA_RESOLVER_CACHE_DIR`**：必须设置为文件系统解析器的有效目录路径
   ```bash
   export VLLM_LORA_RESOLVER_CACHE_DIR=/path/to/lora/adapters
   ```

### 可选的环境变量

- **`VLLM_PLUGINS`**：如果未设置，将加载所有可用插件。如果设置为空字符串，则不加载任何插件。

## 可用的解析器

### lora_filesystem_resolver

文件系统解析器随 vLLM 默认安装，支持从本地目录结构加载 LoRA 适配器。

#### 设置步骤

1. **创建 LoRA 适配器存储目录**：
   ```bash
   mkdir -p /path/to/lora/adapters
   ```

2. **设置环境变量**：
   ```bash
   export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
   export VLLM_PLUGINS=lora_filesystem_resolver
   export VLLM_LORA_RESOLVER_CACHE_DIR=/path/to/lora/adapters
   ```

3. **启动 vLLM 服务器**：
   您的基础模型可以是 `meta-llama/Llama-2-7b-hf`。请确保在环境变量中设置了 Hugging Face token `export HF_TOKEN=xxx235`。
   ```bash
   vllm serve your-base-model \
       --enable-lora
   ```

#### 目录结构要求

文件系统解析器期望 LoRA 适配器按以下结构组织：

```text
/path/to/lora/adapters/
├── adapter1/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files（如适用）
├── adapter2/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files（如适用）
└── ...
```

每个适配器目录必须包含：

- **`adapter_config.json`**：必需的配置文件，结构如下：
  ```json
  {
    "peft_type": "LORA",
    "base_model_name_or_path": "your-base-model-name",
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
    "modules_to_save": null,
    "use_rslora": false,
    "use_dora": false
  }
  ```

- **`adapter_model.bin`**：LoRA 适配器权重文件

#### 使用示例

1. **准备您的 LoRA 适配器**：
   ```bash
   # 假设您有一个 LoRA 适配器在 /tmp/my_lora_adapter
   cp -r /tmp/my_lora_adapter /path/to/lora/adapters/my_sql_adapter
   ```

2. **验证目录结构**：
   ```bash
   ls -la /path/to/lora/adapters/my_sql_adapter/
   # 应显示：adapter_config.json, adapter_model.bin 等
   ```

3. **使用适配器发起请求**：
   ```bash
   curl http://localhost:8000/v1/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "my_sql_adapter",
           "prompt": "Generate a SQL query for:",
           "max_tokens": 50,
           "temperature": 0.1
       }'
   ```

#### 工作原理

1. 当 vLLM 收到对名为 `my_sql_adapter` 的 LoRA 适配器的请求时
2. 文件系统解析器检查 `/path/to/lora/adapters/my_sql_adapter/` 是否存在
3. 如果找到，它会验证 `adapter_config.json` 文件
4. 如果配置与基础模型匹配且有效，则加载该适配器
5. 请求使用新加载的适配器正常处理
6. 该适配器将继续可用于后续请求

## 高级配置

### 多个解析器

您可以配置多个解析器插件以从不同来源加载适配器：

`lora_s3_resolver` 是一个需要您自行实现的自定义解析器示例

```bash
export VLLM_PLUGINS=lora_filesystem_resolver,lora_s3_resolver
```

所有列出的解析器都已启用；在请求时，vLLM 按顺序尝试它们，直到其中一个成功。

### 自定义解析器实现

要实现您自己的解析器插件：

1. **创建新的解析器类**：
   ```python
   from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
   from vllm.lora.request import LoRARequest
   
   class CustomResolver(LoRAResolver):
       async def resolve_lora(self, base_model_name: str, lora_name: str) -> Optional[LoRARequest]:
           # 您的自定义解析逻辑
           pass
   ```

2. **注册解析器**：
   ```python
   def register_custom_resolver():
       resolver = CustomResolver()
       LoRAResolverRegistry.register_resolver("Custom Resolver", resolver)
   ```

## 故障排除

### 常见问题

1. **"VLLM_LORA_RESOLVER_CACHE_DIR must be set to a valid directory"**
   - 确保目录存在且可访问
   - 检查目录的文件权限

2. **"LoRA adapter not found"**
   - 验证适配器目录名称与请求的模型名称匹配
   - 检查 `adapter_config.json` 是否存在且为有效的 JSON
   - 确保 `adapter_model.bin` 在目录中存在

3. **"Invalid adapter configuration"**
   - 验证 `peft_type` 设置为 "LORA"
   - 检查 `base_model_name_or_path` 与您的基础模型匹配
   - 确保 `target_modules` 配置正确

4. **"LoRA rank exceeds maximum"**
   - 检查 `adapter_config.json` 中的 `r` 值不超过 `max_lora_rank` 设置

### 调试提示

1. **启用调试日志**：
   ```bash
   export VLLM_LOGGING_LEVEL=DEBUG
   ```

2. **验证环境变量**：
   ```bash
   echo $VLLM_ALLOW_RUNTIME_LORA_UPDATING
   echo $VLLM_PLUGINS
   echo $VLLM_LORA_RESOLVER_CACHE_DIR
   ```

3. **测试适配器配置**：
   ```bash
   python -c "
   import json
   with open('/path/to/lora/adapters/my_adapter/adapter_config.json') as f:
       config = json.load(f)
   print('Config valid:', config)
   "
   ```
