from __future__ import annotations

import os
import sys
from typing import List

from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage

try:
    # When executed as a module inside the package
    from .agent import build_agent
except Exception:  # pragma: no cover - fallback for direct script execution
    # If run via `python 05_GenAI/langgraph_agent/run.py`, relative imports may fail.
    # Add the parent folder (05_GenAI) to sys.path so `langgraph_agent` is importable.
    import sys
    from pathlib import Path
    pkg_dir = Path(__file__).resolve().parent
    parent_dir = pkg_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from langgraph_agent.agent import build_agent


def main():
    # Load .env from repo root or nearest parent reliably
    try:
        path = find_dotenv()
        if not path:
            raise RuntimeError(".env not found via find_dotenv")
        load_dotenv(path, override=False)
    except Exception:
        load_dotenv('.env', override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment or .env.")
        sys.exit(1)

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    app = build_agent(model=model)

    print("LangGraph Agent (type 'exit' to quit)\n")
    messages: List = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not user:
            continue

        messages.append(HumanMessage(content=user))
        state = {"messages": messages}
        state = app.invoke(state)
        messages = state["messages"]

        # Find last AI message and print
        ai_msgs = [m for m in messages if m.type == "ai"]
        if ai_msgs:
            print(f"Agent: {ai_msgs[-1].content}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
