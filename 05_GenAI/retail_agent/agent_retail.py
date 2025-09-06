from __future__ import annotations

import json
from typing import Dict, List

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from .tools_retail import (
    retail_inventory_status,
    retail_markdown_report,
    retail_price_optimize,
    retail_sales_summary,
)


def build_retail_agent(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """Retail agent with planning and tool use.

    Tools available:
      - retail_sales_summary
      - retail_inventory_status
      - retail_price_optimize
      - retail_markdown_report
    """

    tools = [
        retail_sales_summary,
        retail_inventory_status,
        retail_price_optimize,
        retail_markdown_report,
    ]

    system = (
        "You are a senior retail analytics assistant. \n"
        "When asked for a business summary, first gather facts using tools (sales, inventory),"
        " optionally run price optimization, then produce a concise markdown report. \n"
        "Cite which tools were used and ensure a 'Final Answer:' section.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("messages"),
    ])

    llm = ChatOpenAI(model=model, temperature=temperature)
    runnable = prompt | llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        ai = runnable.invoke(state)
        return {"messages": [ai]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")

    # Route to tools if requested, otherwise end.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")

    return graph.compile()

