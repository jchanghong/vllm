<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

vLLM 实验性地支持 Apple Silicon 上的 macOS。目前，用户必须从源码构建才能在 macOS 上原生运行。

目前，macOS 的 CPU 实现支持 FP32 和 FP16 数据类型。

!!! tip "使用 vLLM-Metal 进行 GPU 加速推理"
    要在 Apple Silicon 上使用 Metal 进行 GPU 加速推理，请查看 [vllm-metal](https://github.com/vllm-project/vllm-metal)，这是一个社区维护的硬件插件，使用 MLX 作为计算后端。

--8<-- [end:installation]
--8<-- [start:requirements]

- 操作系统：`macOS Sonoma` 或更高版本
- SDK：`XCode 15.4` 或更高版本（含 Command Line Tools）
- 编译器：`Apple Clang >= 15.0.0`

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

目前，没有预构建的 Apple silicon CPU wheel 包。

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

安装 XCode 和包含 Apple Clang 的 Command Line Tools 后，执行以下命令从源码构建并安装 vLLM。

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
uv pip install -e .
```

!!! tip
    需要使用 `--index-strategy unsafe-best-match` 标志来解析跨多个包索引（PyTorch CPU 索引和 PyPI）的依赖关系。没有此标志，您可能会遇到 `typing-extensions` 版本冲突。

    "unsafe" 一词指的是包解析策略，而非安全性。默认情况下，`uv` 仅在找到包的第一个索引中搜索，以防止依赖混淆攻击。此标志允许 `uv` 搜索所有配置的索引以找到最佳兼容版本。由于 PyTorch 和 PyPI 都是可信的包源，使用此策略对于 vLLM 安装是安全且合适的。

!!! note
    在 macOS 上，`VLLM_TARGET_DEVICE` 会自动设置为 `cpu`，这是目前唯一支持的设备。

!!! example "故障排除"
    如果构建因标准 C++ 头文件无法找到而失败，出现如下错误，请尝试移除并重新安装 [Xcode 的 Command Line Tools](https://developer.apple.com/download/all/)。

    ```text
    [...] fatal error: 'map' file not found
            1 | #include <map>
                |          ^~~~~
        1 error generated.
        [2/8] Building CXX object CMakeFiles/_C.dir/csrc/cpu/pos_encoding.cpp.o

    [...] fatal error: 'cstddef' file not found
            10 | #include <cstddef>
                |          ^~~~~~~~~
        1 error generated.
    ```

    ---

    如果构建因 C++11/C++17 兼容性错误而失败，如下所示，问题在于构建系统默认使用了较旧的 C++ 标准：

    ```text
    [...] error: 'constexpr' is not a type
    [...] error: expected ';' before 'constexpr'
    [...] error: 'constexpr' does not name a type
    ```

    **解决方案**：您的编译器可能使用了较旧的 C++ 标准。编辑 `cmake/cpu_extension.cmake` 并在 `set(CMAKE_CXX_STANDARD_REQUIRED ON)` 之前添加 `set(CMAKE_CXX_STANDARD 17)`。

    要检查编译器的 C++ 标准支持：
    ```bash
    clang++ -std=c++17 -pedantic -dM -E -x c++ /dev/null | grep __cplusplus
    ```
    在 Apple Clang 16 上您应看到：`#define __cplusplus 201703L`

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

目前，没有预构建的 Arm silicon CPU 镜像。

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

--8<-- [end:build-image-from-source]
--8<-- [start:extra-information]
--8<-- [end:extra-information]
