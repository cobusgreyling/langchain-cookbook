"""
Long-Term Agent Memory
=======================
An agent with semantic memory that persists across conversations:
1. Store facts and user preferences in a vector store
2. Automatically retrieve relevant memories for each interaction
3. Update and consolidate memories over time
"""

import os
import sys
import json
from datetime import datetime
from typing import Annotated

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Memory store (vector-backed) ----------

class SemanticMemory:
    """A semantic memory store backed by a vector database."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(
            collection_name="agent_memory",
            embedding_function=self.embeddings,
        )
        self.memory_count = 0

    def store(self, content: str, memory_type: str = "fact") -> str:
        """Store a new memory."""
        self.memory_count += 1
        doc = Document(
            page_content=content,
            metadata={
                "type": memory_type,
                "timestamp": datetime.now().isoformat(),
                "id": f"mem_{self.memory_count}",
            },
        )
        self.vectorstore.add_documents([doc])
        return f"Stored memory: '{content}'"

    def recall(self, query: str, k: int = 3) -> list[str]:
        """Recall relevant memories based on a query."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        memories = []
        for doc, score in results:
            if score < 1.5:  # Relevance threshold
                memories.append(
                    f"[{doc.metadata.get('type', 'fact')}] {doc.page_content}"
                )
        return memories

    def cleanup(self):
        self.vectorstore.delete_collection()


# Global memory instance
memory_store = SemanticMemory()


# ---------- Tools ----------

@tool
def remember(fact: str, memory_type: str = "fact") -> str:
    """Store a fact or preference in long-term memory.
    memory_type can be 'fact', 'preference', or 'context'."""
    return memory_store.store(fact, memory_type)


@tool
def recall(query: str) -> str:
    """Search long-term memory for information relevant to a query."""
    memories = memory_store.recall(query)
    if not memories:
        return "No relevant memories found."
    return "Recalled memories:\n" + "\n".join(f"  - {m}" for m in memories)


# ---------- State ----------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------- Nodes ----------

TOOLS = [remember, recall]


def enrich_with_memory(state: AgentState):
    """Before the agent responds, automatically recall relevant memories."""
    last_user_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if last_user_msg:
        memories = memory_store.recall(last_user_msg)
        if memories:
            memory_context = "Relevant memories:\n" + "\n".join(f"  - {m}" for m in memories)
            print(f"  [Memory] Auto-recalled {len(memories)} memories")
            return {
                "messages": [SystemMessage(content=memory_context)]
            }

    return {"messages": []}


def agent_node(state: AgentState):
    """Agent with access to long-term memory tools."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    system_msg = SystemMessage(content=(
        "You are a helpful assistant with long-term memory. "
        "You can remember facts about the user and recall them later. "
        "Proactively remember important details the user shares (preferences, facts, context). "
        "When answering, check your memory for relevant information. "
        "Be conversational and reference things you remember."
    ))

    messages = [system_msg] + state["messages"]
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

    graph.add_node("enrich", enrich_with_memory)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "enrich")
    graph.add_edge("enrich", "agent")
    graph.add_conditional_edges("agent", should_use_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def main():
    check_api_key()

    agent = build_agent()

    # Simulate a multi-turn conversation across "sessions"
    conversations = [
        # Session 1: User shares preferences
        {
            "session": "session-1",
            "messages": [
                "Hi! I'm a Python developer and I prefer using async code over threads.",
                "I also really like the FastAPI framework for building APIs.",
            ],
        },
        # Session 2: New session — agent should remember
        {
            "session": "session-2",
            "messages": [
                "What framework should I use for my next API project?",
                "Can you recall what you know about my coding preferences?",
            ],
        },
    ]

    print("=== Long-Term Agent Memory ===\n")

    for convo in conversations:
        session_id = convo["session"]
        print(f"--- {session_id} ---\n")
        config = {"configurable": {"thread_id": session_id}}

        for user_msg in convo["messages"]:
            print(f"User: {user_msg}")
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_msg)]},
                config=config,
            )
            ai_msg = result["messages"][-1]
            print(f"Agent: {ai_msg.content}\n")

        print("-" * 60 + "\n")

    memory_store.cleanup()
    print("Done!")


if __name__ == "__main__":
    main()
