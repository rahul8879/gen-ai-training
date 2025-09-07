from __future__ import annotations

from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from tools import calculator, faq_lookup


def build_agent(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """Build a LangGraph agent with tools bound.

    Returns a compiled graph that accepts a `MessagesState`-compatible dict:
      {"messages": List[BaseMessage]}
    """

    tools = [calculator, faq_lookup]

    system = (
        "You are a pragmatic AI assistant. Use tools when they help produce a more accurate,"
        " verifiable answer. Be concise, cite which tool you used when applicable, and return"
        " final answers clearly marked with 'Final Answer:'."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("messages"),
        ]
    )

    llm = ChatOpenAI(model=model, temperature=temperature)
    runnable = prompt | llm.bind_tools(tools)

    def agent_node(state: MessagesState) -> Dict[str, List]:
        result = runnable.invoke(state)
        return {"messages": [result]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")

    # If the agent requested tools, go to tools; else we are done.
    # Newer LangGraph returns "__end__" when the agent should terminate.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")

    return graph.compile()
