# 基准测试套件

vLLM 提供了全面的基准测试工具，用于性能测试和评估：

- **[基准测试 CLI](./cli.md)**：`vllm bench` CLI 工具和专门的基准测试脚本，用于交互式性能测试。
- **[参数扫描](./sweeps.md)**：自动化 `vllm bench` 在多种配置下运行，可用于[优化和调优](../configuration/optimization.md)。
- **[性能仪表盘](./dashboard.md)**：自动化 CI，每次提交时发布基准测试结果。
