# KV 加载失败恢复测试

此示例基于 `examples/disaggregated` 中的 `example_connector` 示例构建。

它演示了 vLLM 在同步和异步加载模式下从 KV 加载失败中恢复的能力。目标是验证 vLLM 能够正确识别无效的 KV 块，重新调度受影响的请求，并确保成功且一致的输出。

## 文件

- `prefill_example.py` – 执行预填充阶段并保存 KV 数据（与 `example_connector` 中相同）。
- `decode_example.py` – 执行解码阶段。接受：
    - `--simulate-failure`：使用自定义连接器模拟 KV 加载失败。
    - `--async-load`：启用异步 KV 加载模式。
- `load_recovery_example_connector.py` – 定义 `LoadRecoveryExampleConnector`，它是 `ExampleConnector` 的子类，通过无法为第一个解码请求加载块来模拟缺失或损坏的外部 KV 块。
- `run.sh` – 编排测试：运行预填充阶段，然后是三个解码阶段：
    1. 正常解码（基线）。
    2. 模拟同步 KV 加载失败的解码。
    3. 模拟异步 KV 加载失败的解码。

    最后，比较基线的输出与恢复后的输出以验证正确性。

## 工作原理

- 测试通过 `KVTransferConfig.kv_connector_module_path` 动态加载 `LoadRecoveryExampleConnector`，从而无需修改原始连接器即可实现对加载失败的可控模拟。
- 模拟失败的解码阶段预期会触发 vLLM 中的恢复逻辑，产生与基线解码相同的输出。
- 如果恢复失败，脚本将打印输出不匹配的统一差异（diff）并以错误退出。

## 使用方法

```bash
./run.sh
```
