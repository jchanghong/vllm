# 发布 vLLM

vLLM 发布提供了可靠的代码库版本，打包为可通过 [PyPI](https://pypi.org/project/vllm) 方便获取的二进制格式。这些发布也是开发团队向社区传达新功能、改进和可能影响用户的即将发生的变更（包括潜在的中断性变更）的关键里程碑。

## 发布节奏和版本管理

我们计划每 2 周进行一次常规发布。自 v0.12.0 起，常规发布递增次版本号而非补丁版本号。历史发布列表可在[此处](https://vllm.ai/releases)查看。

我们的版本号采用 `vX.Y.Z` 的形式，其中 `X` 是主版本号，`Y` 是次版本号，`Z` 是补丁版本号。它们根据以下规则递增：

* _主版本_ 保留给涉及大规模 API 变更的架构性里程碑，类似于 PyTorch 2.0。
* _次版本_ 对应于常规发布，包括新功能、错误修复和其他向后兼容的变更。
* _补丁版本_ 对应于针对新模型的特殊发布，以及针对关键性能、功能和安全问题的紧急补丁。

此版本方案与 [SemVer](https://semver.org/) 类似以实现兼容性，但向后兼容性仅保证有限的次版本数（详情请参阅我们的[弃用策略](https://docs.vllm.ai/en/latest/contributing/deprecation_policy)）。

## 发布分支

每个发布版本都从专用的发布分支构建。

* 对于 _主版本_ 和 _次版本_ 发布，分支切割在发布上线前 1-2 天进行。
* 对于 _补丁版本_ 发布，重复使用之前切割的发布分支。
* 发布构建通过推送到形如 `vX.Y.Z-rc1` 的 RC 标签触发。这使我们能够为每个发布版本构建和测试多个 RC。
* 最终标签：`vX.Y.Z` 不触发构建，但用于发布说明和资产。
* 在创建分支切割后，我们监控主分支上的任何回退，并将这些回退应用到发布分支。

### Cherry-Pick 标准

在分支切割后，我们以明确的标准来最终确定发布分支，确定哪些 cherry pick 可以合入。注意：cherry pick 是在分支切割后将 PR 合并到发布分支的过程。这些操作通常受到限制，以确保团队有足够的时间在稳定的代码库上完成一轮彻底的测试。

* 回归修复 - 解决与最近发布版本（例如 0.7.1 版本的 0.7.0）相比的功能/性能回归问题
* 关键修复 - 针对严重问题的关键修复，如静默不正确、向后兼容性、崩溃、死锁、（大量）内存泄漏
* 修复最近发布版本中引入的新功能（例如 0.7.1 版本的 0.7.0）
* 文档改进
* 发布分支特定的变更（例如更改版本标识符或 CI 修复）

请注意：**Cherry pick 不允许引入新功能**。所有考虑进行 cherry pick 的 PR 都需要先在主干上合并，发布分支特定的变更是唯一例外。

## 手动验证

### 端到端性能验证

在每次发布之前，我们都会进行端到端性能验证，以确保没有引入回归问题。此验证使用 PyTorch CI 上的 [vllm-benchmark 工作流](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-benchmark.yml)。

**当前覆盖范围：**

* 模型：Llama3、Llama4 和 Mixtral
* 硬件：NVIDIA H100 和 AMD MI300x
* _注意：覆盖范围可能根据新模型发布和硬件可用性而变化_

**性能验证流程：**

**步骤 1：获取访问权限**
请求 [pytorch/pytorch-integration-testing](https://github.com/pytorch/pytorch-integration-testing) 仓库的写权限以运行基准测试工作流。

**步骤 2：查看基准测试配置**
熟悉基准测试配置：

* [CUDA 配置](https://github.com/pytorch/pytorch-integration-testing/tree/main/vllm-benchmarks/benchmarks/cuda)
* [ROCm 配置](https://github.com/pytorch/pytorch-integration-testing/tree/main/vllm-benchmarks/benchmarks/rocm)

**步骤 3：运行基准测试**
导航到 [vllm-benchmark 工作流](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-benchmark.yml) 并配置：

* **vLLM 分支**：设置为发布分支（例如 `releases/v0.9.2`）
* **vLLM 提交**：设置为 RC 提交哈希

**步骤 4：查看结果**
工作流完成后，基准测试结果将发布在 [vLLM 基准测试仪表板](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm) 上，对应分支和提交下。

**步骤 5：性能对比**
将当前结果与上一个发布版本进行比较，以验证没有发生性能回归。这是一个 [v0.9.1 vs v0.9.2](https://hud.pytorch.org/benchmark/llms?startTime=Thu%2C%2017%20Apr%202025%2021%3A43%3A50%20GMT&stopTime=Wed%2C%2016%20Jul%202025%2021%3A43%3A50%20GMT&granularity=week&lBranch=releases/v0.9.1&lCommit=b6553be1bc75f046b00046a4ad7576364d03c835&rBranch=releases/v0.9.2&rCommit=a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f&repoName=vllm-project%2Fvllm&benchmarkName=&modelName=All%20Models&backendName=All%20Backends&modeName=All%20Modes&dtypeName=All%20DType&deviceName=All%20Devices&archName=All%20Platforms) 的对比示例。
