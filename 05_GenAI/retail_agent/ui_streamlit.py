from __future__ import annotations

import os
from typing import List

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage

try:
    from .agent_retail import build_retail_agent
except Exception:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    pkg_dir = Path(__file__).resolve().parent
    parent_dir = pkg_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from retail_agent.agent_retail import build_retail_agent


def init_env():
    load_dotenv(find_dotenv(), override=False)
    return os.getenv("OPENAI_API_KEY"), os.getenv("LLM_MODEL", "gpt-4o-mini")


def main():
    st.set_page_config(page_title="Retail Analyst Agent", page_icon="🛒", layout="wide")
    st.title("Retail Analyst Agent")
    st.caption("Sales, inventory, and pricing tools with agentic planning")

    api_key, default_model = init_env()
    print(f"Using model: {default_model}")
    print(f"API key set: {'yes' if api_key else 'no'}")
    print('api value',api_key)

    with st.sidebar:
        st.header("Settings")
        if not api_key:
            st.error("OPENAI_API_KEY not set. Add it to .env.")
        model = st.text_input("LLM model", value=st.session_state.get("model", default_model))
        temperature = st.slider("Temperature", 0.0, 1.0, float(st.session_state.get("temperature", 0.0)), 0.05)
        if st.button("Apply & Reset"):
            st.session_state.clear()
            st.session_state["model"] = model
            st.session_state["temperature"] = temperature
            # Correct API: rerun the app after changing settings
            st.experimental_rerun()

        st.markdown("---")
        st.caption("Env Info")
        st.code(f"MODEL={model}\nTEMP={temperature}\nAPI_KEY={'set' if api_key else 'missing'}")

    # Validate env (avoid non-ASCII in headers causing httpx errors)
    def _is_ascii(s: str) -> bool:
        try:
            s.encode("ascii")
            return True
        except Exception:
            return False

    if api_key and not _is_ascii(api_key):
        st.error("OPENAI_API_KEY contains non-ASCII characters. Please re-copy a plain ASCII key into .env.")
        return
    for var in ("OPENAI_ORGANIZATION", "OPENAI_PROJECT"):
        val = os.getenv(var)
        if val and not _is_ascii(val):
            st.error(f"{var} contains non-ASCII characters. Remove or replace with plain ASCII.")
            return

    # Init agent and history
    if "app" not in st.session_state:
        st.session_state["app"] = build_retail_agent(model=model, temperature=temperature)
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # type: List[BaseMessage]

    # Render past
    for m in st.session_state["messages"]:
        with st.chat_message("assistant" if m.type != "human" else "user"):
            st.markdown(getattr(m, "content", ""))

    # Input
    prompt = st.chat_input("Ask for a retail analysis or report…")
    if prompt:
        st.session_state["messages"].append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        state = {"messages": st.session_state["messages"]}
        state = st.session_state["app"].invoke(state)
        st.session_state["messages"] = state["messages"]
        # Show the latest AI message
        for m in reversed(st.session_state["messages"]):
            if isinstance(m, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(getattr(m, "content", ""))
                break


if __name__ == "__main__":  # pragma: no cover
    main()
