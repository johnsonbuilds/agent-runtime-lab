"""Run the real OpenAI-compatible agent loop from the command line."""

import sys

from dotenv import load_dotenv

from agent_loop import run_agent_loop, tool_registry
from llm import OpenAICompatibleLLM


def main() -> None:
    load_dotenv()
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        prompt = input("请输入问题：").strip()
    if not prompt:
        raise SystemExit("问题不能为空")

    llm = OpenAICompatibleLLM()
    answer = run_agent_loop(prompt, llm, tool_registry)
    print(answer)


if __name__ == "__main__":
    main()
