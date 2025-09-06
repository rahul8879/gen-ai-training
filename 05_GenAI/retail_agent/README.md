# Retail Analyst Agent (LangGraph)

A realistic, tool-using agent for retail analytics and decision support. The agent plans, calls domain tools (sales analysis, inventory status, pricing optimization), and produces a concise markdown report.

## Capabilities

- Sales summary by date range: revenue, units, top SKUs/categories
- Inventory alerts: low stock vs. reorder point
- Pricing optimization: simulate ±10% price changes with simple elasticity
- Report generation: consolidated markdown with findings and recommendations

## Quick Start

1) Install deps from repo root:
```
pip install -r requirements.txt
```
2) Configure `.env`:
```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```
3) Run the Streamlit UI:
```
streamlit run 05_GenAI/retail_agent/ui_streamlit.py
```

Try asking:
- "Summarize last 14 days sales and flag low inventory."
- "Top 5 SKUs by revenue last month and suggest price changes."

## Files
- `agent_retail.py`: builds the multi-tool agent graph
- `tools_retail.py`: retail domain tools
- `ui_streamlit.py`: chat/report UI
- `data/sales.csv`: sample daily transactions
- `data/inventory.csv`: current stock & reorder points

## Notes
- Data is sample-only; replace with your own CSVs (same columns) or wire to a DB tool.
- The pricing model is a simple elasticity simulation for demos; calibrate with real experiments.
