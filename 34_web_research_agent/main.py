"""
Web Research Agent
===================
An agent that searches the web, reads pages, and synthesises findings:
1. Plan research queries from a user question
2. Search the web (simulated for portability)
3. Extract and summarise information
4. Produce a final research report
"""

import os
import sys
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Simulated web data ----------
# In production, replace with Tavily, SerpAPI, or httpx calls.

WEB_DATA = {
    "langchain framework 2024": [
        {
            "title": "LangChain v0.3: What's New",
            "url": "https://blog.langchain.dev/v0-3",
            "content": (
                "LangChain v0.3 introduced major improvements: first-class support for "
                "Pydantic v2, streamlined package structure with langchain-core, and better "
                "integration with LangGraph for agent workflows. The LCEL (LangChain Expression "
                "Language) was refined for clearer chain composition."
            ),
        },
        {
            "title": "LangChain Ecosystem Overview",
            "url": "https://docs.langchain.com/ecosystem",
            "content": (
                "The LangChain ecosystem includes: langchain-core (base abstractions), "
                "langchain (chains, agents), langchain-community (third-party integrations), "
                "LangGraph (stateful agents), LangSmith (observability), and LangServe (deployment, "
                "now deprecated in favor of direct FastAPI usage)."
            ),
        },
    ],
    "langgraph agents": [
        {
            "title": "Building Agents with LangGraph",
            "url": "https://langchain-ai.github.io/langgraph/",
            "content": (
                "LangGraph is a framework for building stateful, multi-step agent applications. "
                "Key features: StateGraph for defining agent logic, checkpointing for persistence, "
                "human-in-the-loop support, and multi-agent collaboration patterns. "
                "LangGraph is the recommended way to build agents in the LangChain ecosystem."
            ),
        },
    ],
    "python web frameworks comparison": [
        {
            "title": "FastAPI vs Django vs Flask in 2024",
            "url": "https://example.com/framework-comparison",
            "content": (
                "FastAPI excels at async APIs with automatic docs. Django offers batteries-included "
                "web development with ORM and admin. Flask provides a minimalist core with "
                "extensibility. In 2024, FastAPI is the fastest-growing, Django remains the most "
                "used for full-stack, and Flask is popular for microservices."
            ),
        },
    ],
}


# ---------- Tools ----------

@tool
def web_search(query: str) -> str:
    """Search the web for information. Returns a list of results with titles, URLs, and snippets."""
    query_lower = query.lower()

    # Find best matching results
    best_results = []
    for key, results in WEB_DATA.items():
        # Simple keyword matching
        key_words = set(key.split())
        query_words = set(query_lower.split())
        overlap = len(key_words & query_words)
        if overlap > 0:
            for r in results:
                best_results.append((overlap, r))

    if not best_results:
        return "No results found for this query."

    best_results.sort(key=lambda x: -x[0])
    output = []
    for _, r in best_results[:3]:
        output.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:150]}...")
    return "\n\n".join(output)


@tool
def read_page(url: str) -> str:
    """Read the full content of a web page given its URL."""
    for results in WEB_DATA.values():
        for r in results:
            if r["url"] == url:
                return f"# {r['title']}\n\n{r['content']}"
    return f"Could not fetch content from {url}."


@tool
def take_notes(notes: str) -> str:
    """Save research notes for later synthesis. Returns confirmation."""
    research_notes.append(notes)
    return f"Notes saved ({len(notes)} chars). Total notes: {len(research_notes)}."


# Global notes storage
research_notes: list[str] = []


# ---------- State ----------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------- Nodes ----------

TOOLS = [web_search, read_page, take_notes]


def agent_node(state: AgentState):
    """Research agent that searches, reads, and synthesises."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    system = SystemMessage(content=(
        "You are a web research agent. Your workflow:\n"
        "1. Search the web for relevant information using web_search()\n"
        "2. Read promising pages with read_page()\n"
        "3. Take notes on key findings with take_notes()\n"
        "4. After gathering enough information (2-3 searches), synthesise a final report\n\n"
        "Be thorough but concise. Cite sources with URLs when possible."
    ))

    messages = [system] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_use_tools(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
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
    global research_notes
    check_api_key()

    agent = build_agent()

    questions = [
        "What are the latest developments in the LangChain ecosystem?",
        "Compare the top Python web frameworks for building APIs.",
    ]

    print("=== Web Research Agent ===\n")

    for q in questions:
        research_notes = []  # Reset for each question
        print(f"Research question: {q}\n")

        result = agent.invoke({"messages": [HumanMessage(content=q)]})

        ai_msg = result["messages"][-1]
        print(f"Report:\n{ai_msg.content}\n")
        print("=" * 60 + "\n")

    print("Done!")


if __name__ == "__main__":
    main()
