"""Mock 工具 —— 测试环境，不要改。"""


def get_current_time() -> str:
    return "2026-08-06 15:45:00 (mock)"


# 工具注册表：名字 → 可调用对象。
# 你的 loop 应该通过名字查表调用，而不是 if/else 硬编码工具名。
# （第二阶段你会体会到为什么：新增工具时不用改 loop。）
TOOL_REGISTRY = {
    "get_current_time": get_current_time,
}
