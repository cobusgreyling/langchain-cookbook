"""
Multi-Modal Agents (Vision + Tools)
====================================
An agent that can reason over images and call tools:
1. Accept image descriptions (simulated vision input)
2. Analyze visual content with a vision-capable LLM
3. Call tools based on visual analysis
"""

import os
import sys
import base64
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Tools ----------

PRODUCT_DB = {
    "laptop": {"name": "ProBook 15", "price": 999.99, "stock": 23, "category": "electronics"},
    "headphones": {"name": "SoundMax Pro", "price": 249.99, "stock": 57, "category": "electronics"},
    "chair": {"name": "ErgoSeat Plus", "price": 549.99, "stock": 12, "category": "furniture"},
    "monitor": {"name": "UltraView 27", "price": 449.99, "stock": 8, "category": "electronics"},
    "keyboard": {"name": "TypeMaster MK", "price": 149.99, "stock": 34, "category": "electronics"},
}


@tool
def lookup_product(product_type: str) -> str:
    """Look up product details by type (e.g., 'laptop', 'headphones')."""
    product_type = product_type.lower().strip()
    for key, info in PRODUCT_DB.items():
        if key in product_type or product_type in key:
            return (
                f"Product: {info['name']}\n"
                f"Price: ${info['price']}\n"
                f"In stock: {info['stock']} units\n"
                f"Category: {info['category']}"
            )
    return f"No product found matching '{product_type}'."


@tool
def compare_products(product_a: str, product_b: str) -> str:
    """Compare two products side by side."""
    results = []
    for name in [product_a, product_b]:
        key = name.lower().strip()
        found = None
        for k, info in PRODUCT_DB.items():
            if k in key or key in k:
                found = info
                break
        if found:
            results.append(f"{found['name']}: ${found['price']} ({found['stock']} in stock)")
        else:
            results.append(f"{name}: not found")
    return " vs ".join(results)


@tool
def calculate_total(items: list[str]) -> str:
    """Calculate the total price for a list of product types."""
    total = 0.0
    found_items = []
    for item in items:
        for key, info in PRODUCT_DB.items():
            if key in item.lower():
                total += info["price"]
                found_items.append(f"{info['name']}: ${info['price']}")
                break
    if not found_items:
        return "No matching products found."
    return "\n".join(found_items) + f"\n\nTotal: ${total:.2f}"


# ---------- State ----------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------- Agent nodes ----------

TOOLS = [lookup_product, compare_products, calculate_total]


def agent_node(state: AgentState):
    """Vision-capable agent that can analyze images and use tools."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful shopping assistant with vision capabilities. "
            "You can analyze images of products and use tools to look up details, "
            "compare products, and calculate totals. "
            "When a user describes or shows you products, identify them and use your tools."
        ),
    }

    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_use_tools(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# ---------- Build graph ----------

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_use_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def main():
    check_api_key()

    agent = build_agent()

    # Simulate multi-modal interactions (text descriptions standing in for images)
    scenarios = [
        {
            "description": "User shows a photo of a desk with a laptop and headphones",
            "message": (
                "I see a workspace photo with a laptop and wireless headphones on the desk. "
                "Can you look up both of these products and tell me their prices?"
            ),
        },
        {
            "description": "User shows a photo of two monitors side by side",
            "message": (
                "I'm looking at a photo comparing a monitor and a keyboard setup. "
                "Can you compare these two products for me?"
            ),
        },
        {
            "description": "User shows a shopping cart screenshot",
            "message": (
                "I want to buy a laptop, headphones, and a keyboard. "
                "Can you calculate the total for all three items?"
            ),
        },
    ]

    print("=== Multi-Modal Agent (Vision + Tools) ===\n")

    for scenario in scenarios:
        print(f"Scenario: {scenario['description']}")
        print(f"User: {scenario['message']}")

        result = agent.invoke({
            "messages": [HumanMessage(content=scenario["message"])]
        })

        ai_msg = result["messages"][-1]
        print(f"Agent: {ai_msg.content}\n")
        print("-" * 60 + "\n")

    print("Done!")


if __name__ == "__main__":
    main()
