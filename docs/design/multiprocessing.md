# Python 多进程处理

## 调试

请参阅[故障排除](../usage/troubleshooting.md#python-multiprocessing)
页面了解已知问题及解决方案。

## 引言

!!! important
    源代码引用对应的是撰写本文时（2024 年 12 月）的代码状态。

vLLM 中 Python 多进程处理的使用因以下原因而变得复杂：

- 将 vLLM 作为库使用，限制了对内部代码的控制；
- 某些多进程处理方法与 vLLM 依赖项之间存在不兼容性。

本文档描述了 vLLM 如何处理这些挑战。

## 多进程处理方法

[Python 多进程处理方法](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)包括：

- `spawn` - 生成一个新的 Python 进程。Windows 和 macOS 上的默认方法。
- `fork` - 使用 `os.fork()` 分叉 Python 解释器。Linux 上 Python 3.14 之前版本的默认方法。
- `forkserver` - 生成一个服务器进程，该进程将按需分叉新进程。Linux 上 Python 3.14 及更新版本的默认方法。

### 权衡

`fork` 是最快的方法，但与使用线程的依赖项不兼容。如果您在 macOS 下，使用 `fork` 可能导致进程崩溃。

`spawn` 与依赖项的兼容性更好，但在 vLLM 作为库使用时可能存在问题。如果使用代码没有使用 `__main__` 保护（`if __name__ == "__main__":`），则当 vLLM 生成新进程时，代码将被无意中重新执行。这可能导致无限递归等问题。

`forkserver` 将生成一个新的服务器进程，该进程将按需分叉新进程。不幸的是，当 vLLM 作为库使用时，这存在与 `spawn` 相同的问题。服务器进程作为生成的子进程创建，这将重新执行没有 `__main__` 保护的代码。

对于 `spawn` 和 `forkserver`，进程不能依赖于继承任何全局状态，而 `fork` 则可以。

## 与依赖项的兼容性

多个 vLLM 依赖项表明偏好或需要使用 `spawn`：

- <https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing>
- <https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors>
- <https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders>

在初始化这些依赖项后使用 `fork` 存在已知问题。

## 当前状态（v0）

环境变量 `VLLM_WORKER_MULTIPROC_METHOD` 可用于控制 vLLM 使用的方法。当前默认值是 `fork`。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L339-L342>

如果主进程通过 `vllm` 命令控制，
则使用 `spawn`，因为它具有最广泛的兼容性。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/scripts.py#L123-L140>

`multiproc_xpu_executor` 强制使用 `spawn`。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/executor/multiproc_xpu_executor.py#L14-L18>

还有其他一些地方硬编码了 `spawn` 的使用：

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/distributed/device_communicators/all_reduce_utils.py#L135>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/entrypoints/openai/api_server.py#L184>

相关 PR：

- <https://github.com/vllm-project/vllm/pull/8823>

## v1 中的先前状态

有一个环境变量 `VLLM_ENABLE_V1_MULTIPROCESSING` 用于控制在 v1 引擎核心中是否使用多进程处理。默认情况下是关闭的。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L452-L454>

当启用时，v1 `LLMEngine` 将创建一个新进程来运行引擎核心。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L93-L95>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L70-L77>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/core_client.py#L44-L45>

由于上述所有原因——与依赖项和使用 vLLM 作为库的代码的兼容性——它默认是关闭的。

### v1 中所做的更改

使用 Python 的 `multiprocessing` 没有一种简单的解决方案能够适用于所有情况。作为第一步，我们可以让 v1 进入一种 "尽力而为" 选择多进程处理方法的状态，以最大化兼容性。

- 默认使用 `fork`。
- 当我们确定控制主进程时（执行了 `vllm`），使用 `spawn`。
- 如果我们检测到 `cuda` 先前已初始化，则强制使用 `spawn` 并发出警告。我们知道 `fork` 会出问题，所以这是我们能做的最佳选择。

在这种场景下已知仍然会出问题的情况是，使用 vLLM 作为库的代码在调用 vLLM 之前初始化了 `cuda`。我们发出的警告应指示用户添加 `__main__` 保护或禁用多进程处理。

如果发生这种已知故障情况，用户将看到两条说明正在发生什么的消息。首先，来自 vLLM 的日志消息：

```console
WARNING 12-11 14:50:37 multiproc_worker_utils.py:281] CUDA was previously
    initialized. We must use the `spawn` multiprocessing start method. Setting
    VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See
    https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
    for more information.
```

其次，Python 本身将引发一个异常，并附带清晰的说明：

```console
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
```

## 考虑的替代方案

### 检测是否存在 `__main__` 保护

有人建议，如果我们能够检测到将 vLLM 作为库使用的代码是否具有 `__main__` 保护，我们可能会表现更好。这篇 [Stack Overflow 上的帖子](https://stackoverflow.com/questions/77220442/multiprocessing-pool-in-a-python-class-without-name-main-guard)来自一位面临同样问题的库作者。

检测我们是在原始的 `__main__` 进程中还是在后续生成的子进程中是有可能的。然而，检测代码中是否存在 `__main__` 保护似乎并不直接。

这个选项已被认为不切实际而被放弃。

### 使用 `forkserver`

起初看起来 `forkserver` 是这个问题的一个不错的解决方案。然而，它的工作方式在 vLLM 作为库使用时带来了与 `spawn` 相同的挑战。

### 始终强制使用 `spawn`

清理这个问题的一种方法就是始终强制使用 `spawn`，并记录在将 vLLM 作为库使用时需要使用 `__main__` 保护。但这会破坏现有代码，使 vLLM 更难使用，违背了让 `LLM` 类尽可能易于使用的愿望。

与其将这个问题推给用户，我们将保留复杂性，尽最大努力使事情正常运行。

## 未来工作

未来我们可能会考虑采用不同的工作器管理方法来绕过这些挑战。

1. 我们可以实现类似 `forkserver` 的方案，但进程管理器是我们通过运行自己的子进程和自定义工作器管理入口点（启动一个 `vllm-manager` 进程）初始启动的。

2. 我们可以探索其他可能更适合我们需求的库。需要考虑的示例：

    - <https://github.com/joblib/loky>
