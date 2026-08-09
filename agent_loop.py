import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_wether",
            "description": "Get the current wether",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或地区名称，例如 'Beijing' 或 'Tokyo'"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

def get_current_wether(location: str) -> str:
    return f"{location}：24摄氏度，天气晴天"

tool_registry = {
    "get_current_wether": get_current_wether
}


def run_agent_loop(user_message: str, llm, tool_registry: dict, max_iterations: int = 10) -> str:


    messages = [{"role": "user", "content": user_message}]

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        response = llm.chat(messages,tools)
        tool_calls = response.get("tool_calls") or []

        messages.append({
            "role": "assistant",
            "content": response.get("content") or "",
            "tool_calls": tool_calls,
        })

        if tool_calls:
            for tc in tool_calls:
                function = tc["function"]
                name = function["name"]
                arguments = json.loads(function["arguments"])  # JSON 字符串 → dict
                func = tool_registry[name]
                result = func(**arguments)


                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],      # ← 加这个，provider 用来配对
                    "content": str(result),
                })

            continue

        return response["content"]

    # 达到最大调用次数做总结
    messages.append({
        "role": "user",
        "content": "Iteration limit reached. Summarize the progress and give the best possible final answer. Do not call tools."
    })
    response = llm.chat(messages)
    return response.get("content")
