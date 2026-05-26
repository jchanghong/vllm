# CI 失败

当我的 PR 上的 CI 作业失败，但我认为我的 PR 没有导致该失败时，我该怎么办？

- 查看当前 CI 测试失败仪表板：  
  👉 [CI 失败仪表板](https://github.com/orgs/vllm-project/projects/20)

- 如果您的失败**已列出**，很可能是与您的 PR 无关的。
  欢迎帮助修复它！
    - 留下带有失败额外实例链接的评论。
    - 使用 👍 反应来标记受影响的人数。

- 如果您的失败**未列出**，您应该**提交一个问题**。

## 提交 CI 测试失败问题

- **提交错误报告：**  
    👉 [新建 CI 失败报告](https://github.com/vllm-project/vllm/issues/new?template=450-ci-failure.yml)

- **使用以下标题格式：**

    ```text
    [CI Failure]: 失败的测试作业 - 正则/匹配/失败:测试
    ```

- **对于环境字段：**

    ```text
    在主分支提交 abcdef123 上仍然失败
    ```

- **在描述中，包括失败的测试：**

    ```text
    FAILED failing/test.py:failing_test1 - 失败描述
    FAILED failing/test.py:failing_test2 - 失败描述
    https://github.com/orgs/vllm-project/projects/20
    https://github.com/vllm-project/vllm/issues/new?template=400-bug-report.yml
    FAILED failing/test.py:failing_test3 - 失败描述
    ```

- **附加日志**（可折叠部分示例）：
    <details>
    <summary>日志：</summary>

    ```text
    ERROR 05-20 03:26:38 [dump_input.py:68] Dumping input data
    --- Logging error ---  
    Traceback (most recent call last):  
      File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 203, in execute_model  
        return self.model_executor.execute_model(scheduler_output)
    ...
    FAILED failing/test.py:failing_test1 - 失败描述
    FAILED failing/test.py:failing_test2 - 失败描述
    FAILED failing/test.py:failing_test3 - 失败描述
    ```

    </details>

## 日志处理

下载作业日志（无需 Buildkite 登录）：

[.buildkite/scripts/ci-fetch-log.sh](../../../.buildkite/scripts/ci-fetch-log.sh)

```bash
# 找到失败的作业。每行的 URL 格式为 .../builds/<N>#<job_uuid>：
gh pr checks <PR> --repo vllm-project/vllm

# 一步完成下载 + 去除时间戳/ANSI：
.buildkite/scripts/ci-fetch-log.sh "https://buildkite.com/vllm/ci/builds/<N>#<job_uuid>"
```

清理已下载的日志：

[.buildkite/scripts/ci-clean-log.sh](../../../.buildkite/scripts/ci-clean-log.sh)

```bash
./ci-clean-log.sh ci.log
```

使用工具 [wl-clipboard](https://github.com/bugaevc/wl-clipboard) 快速复制粘贴：

```bash
tail -525 ci_build.log | wl-copy
```

## 调查 CI 测试失败

1. 前往 👉 [Buildkite 主分支](https://buildkite.com/vllm/ci/builds?branch=main)
2. 通过二分查找找到第一个出现该问题的构建。
3. 将您的发现添加到 GitHub 问题中。
4. 如果您找到了很有嫌疑的 PR，请在问题中提及并联系贡献者。

## 复现失败

CI 测试失败可能是偶发的。使用 bash 循环重复运行：

[.buildkite/scripts/rerun-test.sh](../../../.buildkite/scripts/rerun-test.sh)

```bash
./rerun-test.sh tests/v1/engine/test_engine_core_client.py::test_kv_cache_events[True-tcp]
```

## 提交 PR

如果您提交 PR 以修复 CI 失败：

- 将 PR 链接到问题：
  在 PR 描述中添加 `Closes #12345`。
- 添加 `ci-failure` 标签：
  这有助于在 [CI 失败 GitHub 项目](https://github.com/orgs/vllm-project/projects/20)中进行跟踪。

## 其他资源

- 🔍 [`main` 分支上的测试可靠性](https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests?branch=main&order=ASC&sort_by=reliability)
- 🧪 [最新的 Buildkite CI 运行](https://buildkite.com/vllm/ci/builds?branch=main)

## 每日分类

使用 [Buildkite 分析（2 天视图）](https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests?branch=main&period=2days)来：

- 识别 **`main` 分支上**的近期测试失败。
- 排除 PR 上的合理测试失败。
- （可选）忽略可靠性为 0% 的测试。

与 [CI 失败仪表板](https://github.com/orgs/vllm-project/projects/20)进行比较。
