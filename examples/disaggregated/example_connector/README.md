# 分离式预填充 V1

此示例包含在 vLLM 离线设置中演示分离式预填充的脚本。

## 文件

- `run.sh` - 一个辅助脚本，将依次运行 `prefill_example.py` 和 `decode_example.py`。
    - 在运行 `run.sh` 之前，请确保你位于 `examples/disaggregated/example_connector` 目录中。
- `prefill_example.py` - 一个仅执行预填充的脚本，将 KV 状态保存到 `local_storage` 目录，并将提示词保存到 `output.txt`。
- `decode_example.py` - 一个仅执行解码的脚本，从 `local_storage` 目录加载 KV 状态，并从 `output.txt` 加载提示词。
