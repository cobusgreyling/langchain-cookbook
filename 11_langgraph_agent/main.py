"""
LangGraph Agent
================
Build a stateful, graph-based agent using LangGraph:
1. Define nodes (functions) for reasoning and tool execution
2. Wire them into a StateGraph with conditional edges
3. Run the agent with checkpointing
"""

import os
import sys
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Tools ----------

@tool
def search_knowledge_base(query: str) -> str:
    """Search an internal knowledge base for information."""
    kb = {
        "refund": "Refund policy: Full refund within 30 days of purchase. After 30 days, store credit only.",
        "shipping": "Standard shipping: 5-7 business days. Express: 1-2 business days ($15 extra).",
        "hours": "Customer support hours: Mon-Fri 9am-6pm EST. Weekend: 10am-4pm EST.",
        "warranty": "All products come with a 1-year manufacturer warranty covering defects.",
    }
    query_lower = query.lower()
    for key, value in kb.items():
        if key in query_lower:
            return value
    return "No relevant information found. Please contact support@example.com."


@tool
def create_ticket(subject: str, priority: str) -> str:
    """Create a support ticket. Priority must be 'low', 'medium', or 'high'."""
    if priority not in ("low", "medium", "high"):
        return "Invalid priority. Use 'low', 'medium', or 'high'."
    ticket_id = hash(subject) % 10000
    return f"Ticket #{ticket_id} created — subject: '{subject}', priority: {priority}"


# ---------- Graph state ----------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------- Graph nodes ----------

def chatbot_node(state: AgentState):
    """The LLM reasons about the conversation and decides whether to call tools."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([search_knowledge_base, create_ticket])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_use_tools(state: AgentState) -> str:
    """Conditional edge: route to tools if the last message has tool calls."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# ---------- Build the graph ----------

def build_agent():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("chatbot", chatbot_node)
    graph.add_node("tools", ToolNode([search_knowledge_base, create_ticket]))

    # Add edges
    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges("chatbot", should_use_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "chatbot")  # After tools, go back to chatbot

    # Compile with checkpointing
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def main():
    check_api_key()

    agent = build_agent()
    config = {"configurable": {"thread_id": "demo-session"}}

    conversations = [
        "What is your refund policy?",
        "I'd like to return something I bought 45 days ago. Can you create a ticket for me?",
        "What are your support hours?",
    ]

    print("=== LangGraph Agent ===\n")

    for user_msg in conversations:
        print(f"User: {user_msg}")
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_msg)]},
            config=config,
        )
        ai_msg = response["messages"][-1]
        print(f"Agent: {ai_msg.content}\n")

    print("Done!")


if __name__ == "__main__":
    main()
