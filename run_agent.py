"""Run the real OpenAI-compatible agent loop from the command line."""

import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_DIR))

from dotenv import load_dotenv

from agent_runtime.agent import Conversation, run_turn
from agent_runtime.providers import OpenAICompatibleLLM
from agent_runtime.tools import create_default_registry


def main() -> None:
    load_dotenv()
    llm = OpenAICompatibleLLM()
    tools = create_default_registry()
    conversation = Conversation()
    prompt = " ".join(sys.argv[1:]).strip()

    while True:
        if not prompt:
            try:
                prompt = input("> ").strip()
            except EOFError:
                print()
                break
        if prompt.lower() in {"exit", "quit"}:
            break
        if prompt:
            print(run_turn(prompt, llm, tools, conversation=conversation))
        prompt = ""


if __name__ == "__main__":
    main()
