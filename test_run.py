"""Agent Loop 验收测试。"""

from agent_loop import run_agent_loop, tool_registry


class FakeLLM:
    """返回预设响应的测试替身，不负责调用真实模型。"""

    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.call_count = 0
        self.seen_messages: list[list[dict]] = []

    def chat(self, messages: list[dict], tools=None) -> dict:
        self.seen_messages.append([m.copy() for m in messages])
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


def test1_tool_call():
    """工具调用闭环：模型调工具 → 结果进 messages → 模型带着结果回答"""
    llm = FakeLLM([
        {"content": "我来查一下", "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "get_current_wether",
                    "arguments": '{"location": "北京"}',
                },
            }
        ]},
        {"content": "北京天气晴朗，24摄氏度", "tool_calls": []},
    ])
    answer = run_agent_loop("北京天气怎么样？", llm, tool_registry)
    assert "24摄氏度" in answer, f"FAIL: 最终回答异常: {answer}"

    # 核心考察点：llm 第二次被调用时，messages 里必须包含工具返回的时间。
    # 这就是 Observation——模型是状态机，工具结果是它下一轮的输入。
    second_call_msgs = llm.seen_messages[1]
    assert any("24摄氏度" in str(m) for m in second_call_msgs), \
        "FAIL: 工具结果没有 append 进 messages，模型根本看不到工具干了什么"
    print("PASS test1: 工具调用闭环（Observation append 正确）")


def test2_direct_answer():
    """无工具调用时一次退出，不多转"""
    llm = FakeLLM([
        {"content": "你好！有什么可以帮你的？", "tool_calls": []},
    ])
    answer = run_agent_loop("你好", llm, tool_registry)
    assert "你好" in answer, f"FAIL: {answer}"
    assert llm.call_count == 1, \
        f"FAIL: 没有工具调用时 loop 应该一次就结束，实际调了 {llm.call_count} 次"
    print("PASS test2: 直接回答（Termination 正确）")


def test3_infinite_tool_calls():
    """退化模型永远想调工具 → loop 必须自己停下来。

    MockLLM 序列用完后会一直重复最后一个响应（永远带 tool_calls）。
    如果你的循环没有防护，这里会死循环。
    """
    llm = FakeLLM([
        {"content": "我还要再查一次", "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "get_current_wether",
                    "arguments": '{"location": "北京"}',
                },
            }
        ]},
    ])
    answer = run_agent_loop("查天气", llm, tool_registry, max_iterations=5)
    assert llm.call_count <= 5, \
        f"FAIL: loop 没有停下来，调了 {llm.call_count} 次"
    # 达到上限后返回什么由你设计——只要函数能正常 return 就算过
    assert isinstance(answer, str), "FAIL: 达到上限后也应该返回一个字符串"
    print("PASS test3: 无限循环防护（max_iterations 生效）")


if __name__ == "__main__":
    test1_tool_call()
    test2_direct_answer()
    test3_infinite_tool_calls()
    print("\n✅ 全部通过。喊我做 review，然后我们对照 Hermes 源码。")
