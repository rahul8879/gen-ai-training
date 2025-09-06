# LangGraph Agentic AI (Tool-Using Agent)

This mini-project demonstrates a professional, tool-using agent built with LangGraph + LangChain. The agent can decide when to use tools (calculator, FAQ lookup) and when to answer directly. It’s a clean, minimal template you can extend with RAG, web search, or custom APIs.

## Features

- Agent graph using `langgraph` with a standard “Agent -> Tools -> Agent” loop
- Two tools:
  - `calculator`: safe evaluation of arithmetic expressions
  - `faq_lookup`: answers from a local curated FAQ JSON
- CLI chat with conversation memory

## Setup

1) Create and activate a virtual environment, then install dependencies from the repo root:

```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) Configure environment variables in `.env` at the repo root:

```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

3) Run the agent CLI:

```
python 05_GenAI/langgraph_agent/run.py
```

Type questions; `exit` to quit.

4) Run the Streamlit UI:

```
streamlit run 05_GenAI/langgraph_agent/ui_streamlit.py
```

Use the sidebar to change model/temperature and to reset the chat.

## Docker

Build and run with Docker (uses `.env` via compose):

```
# From repo root
docker compose up --build
```

Open http://localhost:8501 and chat. The container mounts the repo for live code reloads. Set `OPENAI_API_KEY` in `.env` before running.

## Files

- `agent.py`: builds the LangGraph agent with tools
- `tools.py`: defines `calculator` and `faq_lookup`
- `run.py`: simple CLI loop maintaining chat history
- `data/faq.json`: sample professional FAQ content

## Extend It

- Add a RAG tool: embed local docs and retrieve sources
- Add a web/search tool: carefully sandbox I/O and cite sources
- Add guardrails: system policy, sensitive topics, and output validation

## Notes

- The agent requires an LLM; this template uses OpenAI via `langchain_openai`. Swap in your preferred provider if needed.
- Keep secrets in `.env` and never commit API keys.
