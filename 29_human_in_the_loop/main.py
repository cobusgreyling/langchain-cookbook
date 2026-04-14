"""
Human-in-the-Loop with LangGraph
==================================
An agent that pauses for human approval before taking sensitive actions:
1. Classify incoming requests
2. For sensitive operations, pause and ask for human approval
3. Execute approved actions or provide alternative suggestions
"""

import os
import sys
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Simulated database ----------

ACCOUNTS = {
    "alice": {"balance": 5000.00, "email": "alice@example.com"},
    "bob": {"balance": 3200.00, "email": "bob@example.com"},
}


# ---------- Tools ----------

@tool
def check_balance(account: str) -> str:
    """Check the balance of an account. This is a read-only operation."""
    account = account.lower()
    if account in ACCOUNTS:
        return f"Account '{account}' balance: ${ACCOUNTS[account]['balance']:.2f}"
    return f"Account '{account}' not found."


@tool
def transfer_funds(from_account: str, to_account: str, amount: float) -> str:
    """Transfer funds between accounts. This is a SENSITIVE operation requiring approval."""
    from_account = from_account.lower()
    to_account = to_account.lower()

    if from_account not in ACCOUNTS:
        return f"Source account '{from_account}' not found."
    if to_account not in ACCOUNTS:
        return f"Destination account '{to_account}' not found."
    if ACCOUNTS[from_account]["balance"] < amount:
        return f"Insufficient funds. Balance: ${ACCOUNTS[from_account]['balance']:.2f}"

    ACCOUNTS[from_account]["balance"] -= amount
    ACCOUNTS[to_account]["balance"] += amount
    return (
        f"Transferred ${amount:.2f} from {from_account} to {to_account}. "
        f"New balances — {from_account}: ${ACCOUNTS[from_account]['balance']:.2f}, "
        f"{to_account}: ${ACCOUNTS[to_account]['balance']:.2f}"
    )


@tool
def delete_account(account: str) -> str:
    """Delete an account permanently. This is a SENSITIVE operation requiring approval."""
    account = account.lower()
    if account not in ACCOUNTS:
        return f"Account '{account}' not found."
    del ACCOUNTS[account]
    return f"Account '{account}' has been permanently deleted."


# ---------- Sensitivity classification ----------

SENSITIVE_TOOLS = {"transfer_funds", "delete_account"}

TOOLS = [check_balance, transfer_funds, delete_account]


# ---------- State ----------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_approval: bool
    approved: bool


# ---------- Nodes ----------

def agent_node(state: AgentState):
    """Agent that reasons about requests and calls tools."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    system_msg = {
        "role": "system",
        "content": (
            "You are a banking assistant. You can check balances, transfer funds, "
            "and delete accounts. Be helpful and concise."
        ),
    }
    messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def check_sensitivity(state: AgentState) -> str:
    """Check if the last message contains sensitive tool calls."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END

    sensitive = any(tc["name"] in SENSITIVE_TOOLS for tc in last_message.tool_calls)
    if sensitive:
        return "request_approval"
    return "tools"


def request_approval(state: AgentState):
    """Flag that human approval is needed."""
    last_message = state["messages"][-1]
    tool_names = [tc["name"] for tc in last_message.tool_calls]
    tool_args = [tc["args"] for tc in last_message.tool_calls]

    summary = "; ".join(
        f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})"
        for name, args in zip(tool_names, tool_args)
    )
    print(f"\n  ⚠ APPROVAL REQUIRED: {summary}")
    return {"pending_approval": True}


def approval_gate(state: AgentState) -> str:
    """Route based on approval status."""
    if state.get("approved"):
        return "tools"
    return "denied"


def denied_node(state: AgentState):
    """Handle denied actions."""
    return {
        "messages": [AIMessage(content="The requested action was denied by the human reviewer. Is there anything else I can help with?")],
        "pending_approval": False,
        "approved": False,
    }


# ---------- Build graph ----------

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("request_approval", request_approval)
    graph.add_node("denied", denied_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", check_sensitivity, {
        "request_approval": "request_approval",
        "tools": "tools",
        END: END,
    })

    # After approval request, check if approved
    graph.add_conditional_edges("request_approval", approval_gate, {
        "tools": "tools",
        "denied": "denied",
    })

    graph.add_edge("tools", "agent")
    graph.add_edge("denied", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ---------- Simulation ----------

def simulate_human_approval(action_description: str) -> bool:
    """Simulate human approval (auto-approve transfers, deny deletions)."""
    if "delete" in action_description.lower():
        print("  ✗ Human DENIED the action (deletion not allowed)")
        return False
    print("  ✓ Human APPROVED the action")
    return True


def main():
    check_api_key()

    agent = build_agent()

    scenarios = [
        ("Check Alice's balance", True),
        ("Transfer $500 from Alice to Bob", True),
        ("Delete Bob's account", False),
    ]

    print("=== Human-in-the-Loop Agent ===\n")

    for i, (request, should_approve) in enumerate(scenarios):
        config = {"configurable": {"thread_id": f"session-{i}"}}

        print(f"User: {request}")
        result = agent.invoke(
            {
                "messages": [HumanMessage(content=request)],
                "pending_approval": False,
                "approved": False,
            },
            config=config,
        )

        # Check if approval was requested
        if result.get("pending_approval"):
            # Simulate human decision
            approved = simulate_human_approval(request)
            result = agent.invoke(
                {"approved": approved, "pending_approval": True},
                config=config,
            )

        ai_msg = result["messages"][-1]
        print(f"Agent: {ai_msg.content}\n")
        print("-" * 60 + "\n")

    print("Done!")


if __name__ == "__main__":
    main()
