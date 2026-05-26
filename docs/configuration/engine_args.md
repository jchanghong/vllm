---
toc_depth: 3
---

# 引擎参数

引擎参数控制 vLLM 引擎的行为。

- 对于[离线推理](../serving/offline_inference.md)，它们是 [LLM][vllm.LLM] 类参数的一部分。
- 对于[在线服务](../serving/online_serving/README.md)，它们是 `vllm serve` 参数的一部分。

引擎参数类 [EngineArgs][vllm.engine.arg_utils.EngineArgs] 和 [AsyncEngineArgs][vllm.engine.arg_utils.AsyncEngineArgs] 是 [vllm.config][] 中定义的配置类的组合。因此，如果您需要开发者文档，我们建议查看这些配置类，因为它们是类型、默认值和文档字符串的真实来源。

--8<-- "docs/cli/json_tip.inc.md"

## `EngineArgs`

--8<-- "docs/generated/argparse/engine_args.inc.md"

## `AsyncEngineArgs`

--8<-- "docs/generated/argparse/async_engine_args.inc.md"
