"""
Multi-Agent Collaboration
==========================
A supervisor agent that routes tasks to specialised worker agents:
1. Researcher — searches for information
2. Writer — drafts content based on research
3. Supervisor — orchestrates the workflow
"""

import os
import sys

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Simulated knowledge base ----------

KNOWLEDGE = {
    "python": (
        "Python is a high-level programming language created by Guido van Rossum in 1991. "
        "Key features: dynamic typing, garbage collection, multiple paradigms (OOP, functional, procedural). "
        "Popular for web dev (Django, Flask), data science (pandas, NumPy), ML (PyTorch, TensorFlow), and scripting."
    ),
    "rust": (
        "Rust is a systems programming language focused on safety, speed, and concurrency. "
        "Created by Graydon Hoare at Mozilla, first stable release in 2015. "
        "Key features: ownership system, zero-cost abstractions, no garbage collector, pattern matching."
    ),
    "go": (
        "Go (Golang) was designed at Google by Robert Griesemer, Rob Pike, and Ken Thompson, released in 2009. "
        "Key features: static typing, garbage collection, built-in concurrency (goroutines), fast compilation."
    ),
}


# ---------- Worker agents ----------

def researcher_agent(topic: str) -> str:
    """Simulated researcher that gathers information on a topic."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Look up local knowledge first
    context = ""
    for key, info in KNOWLEDGE.items():
        if key in topic.lower():
            context += info + "\n"

    if not context:
        context = "No specific knowledge found. Use general knowledge."

    prompt = ChatPromptTemplate.from_template(
        "You are a research assistant. Provide key facts and talking points about this topic.\n\n"
        "Available context:\n{context}\n\n"
        "Topic: {topic}\n\n"
        "Research notes (bullet points):"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"topic": topic, "context": context})


def writer_agent(topic: str, research: str) -> str:
    """Writer that drafts content based on research notes."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    prompt = ChatPromptTemplate.from_template(
        "You are a technical writer. Write a concise, engaging summary (3-4 sentences) "
        "based on these research notes.\n\n"
        "Topic: {topic}\n\n"
        "Research notes:\n{research}\n\n"
        "Summary:"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"topic": topic, "research": research})


# ---------- Supervisor ----------

class SupervisorDecision(BaseModel):
    """Supervisor routing decision."""
    next_agent: str = Field(description="Which agent to call next: 'researcher', 'writer', or 'done'")
    task: str = Field(description="The task to assign to the next agent")


def supervisor_agent(user_request: str) -> str:
    """Supervisor that orchestrates researcher and writer agents."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Step 1: Research phase
    print(f"  [Supervisor] Assigning research task...")
    research = researcher_agent(user_request)
    print(f"  [Researcher] Gathered notes ({len(research)} chars)")

    # Step 2: Writing phase
    print(f"  [Supervisor] Assigning writing task...")
    draft = writer_agent(user_request, research)
    print(f"  [Writer] Draft complete ({len(draft)} chars)")

    # Step 3: Supervisor review
    review_prompt = ChatPromptTemplate.from_template(
        "You are a supervisor reviewing a draft. Make minor improvements if needed, "
        "otherwise return it as-is. Keep it concise.\n\n"
        "Original request: {request}\n"
        "Draft:\n{draft}\n\n"
        "Final version:"
    )
    review_chain = review_prompt | llm | StrOutputParser()
    final = review_chain.invoke({"request": user_request, "draft": draft})
    print(f"  [Supervisor] Review complete")

    return final


def main():
    check_api_key()

    tasks = [
        "Write a brief overview of Python as a programming language",
        "Compare Rust and Go for systems programming",
    ]

    print("=== Multi-Agent Collaboration ===\n")

    for task in tasks:
        print(f"Request: {task}")
        result = supervisor_agent(task)
        print(f"\nFinal output:\n{result}\n")
        print("-" * 60 + "\n")

    print("Done!")


if __name__ == "__main__":
    main()
