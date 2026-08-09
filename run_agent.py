"""Run the real OpenAI-compatible agent loop from the command line."""

import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent / "src" / "agent_runtime"
sys.path.insert(0, str(SOURCE_DIR))

from dotenv import load_dotenv

from agent.loop import run_agent_loop
from providers.llm import OpenAICompatibleLLM
from tools.tools import create_default_registry


def main() -> None:
    load_dotenv()
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        prompt = input("请输入问题：").strip()
    if not prompt:
        raise SystemExit("问题不能为空")

    llm = OpenAICompatibleLLM()
    tools = create_default_registry()
    print(run_agent_loop(prompt, llm, tools))


if __name__ == "__main__":
    main()
