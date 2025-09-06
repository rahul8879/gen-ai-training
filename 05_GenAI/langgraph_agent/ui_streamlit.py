from __future__ import annotations

import os
from typing import List

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage

try:
    # When executed as a module inside the package
    from .agent import build_agent
except Exception:  # pragma: no cover - fallback for direct script execution
    # If run via `streamlit run .../ui_streamlit.py`, relative imports may fail.
    # Add the parent folder (05_GenAI) to sys.path so `langgraph_agent` is importable.
    import sys
    from pathlib import Path
    pkg_dir = Path(__file__).resolve().parent
    parent_dir = pkg_dir.parent  # points to 05_GenAI
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from langgraph_agent.agent import build_agent


def init_env():
    # Load .env from repo root or nearest parent reliably
    try:
        path = find_dotenv()
        if not path:
            raise RuntimeError(".env not found via find_dotenv")
        load_dotenv(path, override=False)
    except Exception:
        # Fallback to repo-root relative
        load_dotenv('.env', override=False)
    return os.getenv("OPENAI_API_KEY"), os.getenv("LLM_MODEL", "gpt-4o-mini")


def get_role(msg: BaseMessage) -> str:
    if msg.type == "human":
        return "user"
    if msg.type == "ai":
        return "assistant"
    if msg.type in {"tool", "system"}:
        return "assistant"
    return "assistant"


def render_message(msg: BaseMessage, show_tools: bool = True):
    role = get_role(msg)
    content = getattr(msg, "content", "")
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None) and show_tools:
        # Display the AI thought and tool calls succinctly
        with st.chat_message(role):
            st.markdown(content or "")
            tcalls = msg.tool_calls or []
            if tcalls:
                with st.expander("Tool calls", expanded=False):
                    for i, tc in enumerate(tcalls, 1):
                        st.markdown(f"{i}. `{tc.get('name')}` → args: `{tc.get('args')}`")
        return
    if isinstance(msg, ToolMessage) and show_tools:
        with st.chat_message(role):
            st.caption(f"Tool `{msg.name}` result:")
            st.code(content)
        return
    with st.chat_message(role):
        st.markdown(content or "")


def main():
    st.set_page_config(page_title="LangGraph Agent", page_icon="🤖", layout="wide")
    st.title("LangGraph Agentic Chat")
    st.caption("Tool-using agent with calculator + FAQ lookup")

    api_key, default_model = init_env()

    with st.sidebar:
        st.header("Settings")
        if not api_key:
            st.error("OPENAI_API_KEY is not set. Add it to .env or your environment.")
        model = st.text_input("LLM model", value=st.session_state.get("model", default_model))
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(st.session_state.get("temperature", 0.0)), step=0.05)
        show_tools = st.checkbox("Show tool events", value=st.session_state.get("show_tools", True))
        if st.button("Apply & Reset Chat"):
            st.session_state.clear()
            st.session_state["model"] = model
            st.session_state["temperature"] = temperature
            st.session_state["show_tools"] = show_tools
            st.experimental_rerun()

        st.markdown("---")
        st.caption("Env Info")
        st.code(f"MODEL={model}\nTEMP={temperature}\nAPI_KEY={'set' if api_key else 'missing'}")

    # Initialize app and state
    if "model" not in st.session_state:
        st.session_state["model"] = model
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = temperature
    if "show_tools" not in st.session_state:
        st.session_state["show_tools"] = show_tools
    if "app" not in st.session_state:
        st.session_state["app"] = build_agent(model=st.session_state["model"], temperature=st.session_state["temperature"])
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # type: List[BaseMessage]

    # Render chat history
    for msg in st.session_state["messages"]:
        render_message(msg, show_tools=st.session_state["show_tools"])

    # User input
    prompt = st.chat_input("Ask a question…")
    if prompt:
        st.session_state["messages"].append(HumanMessage(content=prompt))
        render_message(st.session_state["messages"][-1])

        state = {"messages": st.session_state["messages"]}
        state = st.session_state["app"].invoke(state)
        st.session_state["messages"] = state["messages"]

        # Render only the new messages after the user input
        # Find the last AI message to display succinctly
        last_ai = None
        for m in reversed(st.session_state["messages"]):
            if isinstance(m, AIMessage):
                last_ai = m
                break
        if last_ai:
            render_message(last_ai, show_tools=st.session_state["show_tools"])


if __name__ == "__main__":  # pragma: no cover
    main()
