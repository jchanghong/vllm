# 交错思考

## 介绍

交错思考允许模型在工具调用之间进行推理，从而在接收到工具结果后能够做出更复杂的决策。此功能帮助模型将多个工具调用与中间的推理步骤串联起来，并基于中间结果做出细致的决策。

重要提示：交错思考会增加 token 使用量和响应延迟。在启用此功能时，请考虑您的预算和性能要求。

## 交错思考的工作原理

使用交错思考，模型可以：

- 在决定下一步操作之前，思考工具调用的结果
- 将多个工具调用与中间的推理步骤串联起来
- 基于中间结果做出更细致的决策
- 为其工具选择过程提供透明的推理

## 支持的模型

vLLM 目前支持以下交错思考模型：

| 模型系列 | 推理解析器名称 |
| ------------ | --------------------- |
| moonshotai/Kimi-K2-Thinking | kimi_k2 |
| MiniMaxAI/MiniMax-M2 | minimax_m2 |

## 示例用法

要使用带有工具调用的交错思考，请指定一个支持此功能的模型，并在聊天补全请求中启用工具调用。以下是一个示例：

??? code

    ```python
    """
    vllm serve MiniMaxAI/MiniMax-M2 \
      --tensor-parallel-size 4 \
      --tool-call-parser minimax_m2 \
      --reasoning-parser minimax_m2 \
      --enable-auto-tool-choice
    """
    import json
    
    from openai import OpenAI
    
    client = OpenAI(base_url="http://localhost:8000/v1",     api_key="dummy")
    
    
    def get_current_weather(location: str, unit: "str"):
        """Get the current weather in a given location"""
        if unit == "celsius":
            return f"The current temperature in {location} is 22°C."
        else:
            return f"The current temperature in {location} is 72°F."
    
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given     location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g.,     'San Francisco, CA'",
                        },
                        "unit": {"type": "string", "enum":     ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What's the weather in Fahrenheit like in San Francisco?"}]
    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    tool_call = response.choices[0].message.tool_calls[0].function
    
    messages.append(
        {
            "role": "assistant",
            "tool_calls": response.choices[0].message.tool_calls,
            "reasoning": response.choices[0].message.reasoning, # 追加推理内容
        }
    )
    
    # 模拟工具执行
    available_tools = {"get_weather": get_current_weather}
    
    completion_tool_calls = response.choices[0].message.tool_calls
    for call in completion_tool_calls:
        tool_to_call = available_tools[call.function.name]
        args = json.loads(call.function.arguments)
        result = tool_to_call(**args)
        messages.append(
            {
                "role": "tool",
                "content": result,
                "tool_call_id": call.id,
                "name": call.function.name,
            }
        )
    response_2 = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    print(response_2.choices[0].message.content)
    ```
此示例演示了如何使用天气查询函数设置带有工具调用的交错思考。模型在生成最终响应之前会思考工具结果。
