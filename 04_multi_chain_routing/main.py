"""
Multi-Chain Routing
====================
Route user queries to specialised chains based on detected topic.
A classifier chain picks the right expert, then the expert chain answers.
"""

import os
import sys

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Classification schema ----------

class RouteDecision(BaseModel):
    """Decides which expert chain should handle the query."""
    topic: str = Field(
        description="One of: 'coding', 'science', 'history', 'general'"
    )


# ---------- Expert prompts ----------

EXPERT_PROMPTS = {
    "coding": ChatPromptTemplate.from_template(
        "You are an expert software engineer. Give a clear, practical answer "
        "with code examples when appropriate.\n\nQuestion: {question}"
    ),
    "science": ChatPromptTemplate.from_template(
        "You are a science educator. Explain concepts clearly, using analogies "
        "and real-world examples.\n\nQuestion: {question}"
    ),
    "history": ChatPromptTemplate.from_template(
        "You are a historian. Provide accurate, engaging answers with relevant "
        "dates and context.\n\nQuestion: {question}"
    ),
    "general": ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the question clearly and concisely."
        "\n\nQuestion: {question}"
    ),
}


def main():
    check_api_key()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    output_parser = StrOutputParser()

    # Router: classify the query topic
    router_prompt = ChatPromptTemplate.from_template(
        "Classify this question into exactly one category: coding, science, history, or general.\n\n"
        "Question: {question}"
    )
    router_chain = router_prompt | llm.with_structured_output(RouteDecision)

    questions = [
        "How do I reverse a linked list in Python?",
        "Why is the sky blue?",
        "What caused the fall of the Roman Empire?",
        "What's a good recipe for banana bread?",
    ]

    print("=== Multi-Chain Router ===\n")

    for q in questions:
        # Step 1: classify
        route = router_chain.invoke({"question": q})
        topic = route.topic if route.topic in EXPERT_PROMPTS else "general"

        # Step 2: route to expert chain
        expert_chain = EXPERT_PROMPTS[topic] | llm | output_parser
        answer = expert_chain.invoke({"question": q})

        print(f"Q: {q}")
        print(f"Routed to: {topic}")
        print(f"A: {answer[:200]}...\n")

    print("Done!")


if __name__ == "__main__":
    main()
