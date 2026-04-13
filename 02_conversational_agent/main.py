"""
Conversational Agent with Tool Use
====================================
An agent that reasons about when to use tools (calculator, dictionary)
and maintains a multi-turn conversation.
"""

import os
import sys
import math

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Tools ----------

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports basic arithmetic, powers, sqrt, etc.
    Examples: '2 + 3', '10 ** 2', 'sqrt(144)', '(5 + 3) * 2'
    """
    allowed_names = {
        "sqrt": math.sqrt,
        "abs": abs,
        "pow": pow,
        "round": round,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def dictionary_lookup(word: str) -> str:
    """Look up the definition of a word. Returns a brief definition."""
    definitions = {
        "langchain": "A framework for building applications powered by large language models.",
        "embedding": "A dense vector representation of text that captures semantic meaning.",
        "retrieval": "The process of finding relevant documents from a collection based on a query.",
        "agent": "An AI system that can reason about and decide which tools to use to accomplish tasks.",
        "hallucination": "When an AI model generates information that is not grounded in its training data or context.",
        "tokenization": "The process of breaking text into smaller units (tokens) for processing by a language model.",
    }
    word_lower = word.lower().strip()
    if word_lower in definitions:
        return f"{word}: {definitions[word_lower]}"
    return f"No definition found for '{word}'. Available words: {', '.join(definitions.keys())}"


def main():
    check_api_key()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [calculator, dictionary_lookup]
    agent = create_react_agent(llm, tools)

    # Multi-turn conversation
    conversations = [
        "What is the square root of 256 plus 15 squared?",
        "What does the word 'embedding' mean in AI?",
        "Now multiply my first result by 2",
    ]

    message_history = []
    print("=== Conversational Agent ===\n")

    for user_msg in conversations:
        print(f"User: {user_msg}")
        message_history.append(HumanMessage(content=user_msg))

        response = agent.invoke({"messages": message_history})
        ai_msg = response["messages"][-1]

        print(f"Agent: {ai_msg.content}\n")
        message_history = response["messages"]

    print("Done!")


if __name__ == "__main__":
    main()
