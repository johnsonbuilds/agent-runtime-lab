"""Run the real OpenAI-compatible agent loop from the command line."""

import argparse
import asyncio
import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_DIR))

from dotenv import load_dotenv

from agent_runtime.agent import Conversation, RunTrace, run_turn
from agent_runtime.providers import OpenAICompatibleLLM
from agent_runtime.tools import create_default_registry


async def _run() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        metavar="PATH",
        help="append run events as JSONL to PATH",
    )
    parser.add_argument("prompt", nargs="*", help="prompt to send; omit for interactive mode")
    args = parser.parse_args()

    llm = OpenAICompatibleLLM()
    tools = create_default_registry()
    conversation = Conversation()
    trace = RunTrace(output_path=args.trace) if args.trace else None
    prompt = " ".join(args.prompt).strip()

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
            print(await run_turn(prompt, llm, tools, conversation=conversation, trace=trace))
        prompt = ""


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
