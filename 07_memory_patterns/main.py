"""
Memory Patterns
================
Demonstrates different memory strategies for conversation:
1. Full buffer memory (keep everything)
2. Sliding window memory (last N exchanges)
3. Summary memory (compress old messages)
"""

import os
import sys

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Pattern 1: Full Buffer Memory ----------

def example_buffer_memory():
    """Keep all messages — simple but grows without bound."""
    print("=== Pattern 1: Full Buffer Memory ===")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()

    history = []
    exchanges = [
        "My name is Alice and I'm a data scientist.",
        "I work at Acme Corp on recommendation systems.",
        "What do you know about me so far?",
    ]

    for msg in exchanges:
        print(f"  User: {msg}")
        response = chain.invoke({"input": msg, "history": history})
        print(f"  AI: {response}")
        history.append(HumanMessage(content=msg))
        history.append(AIMessage(content=response))

    print(f"  [Messages stored: {len(history)}]\n")


# ---------- Pattern 2: Sliding Window Memory ----------

def example_window_memory():
    """Keep only the last N exchanges to bound context size."""
    print("=== Pattern 2: Sliding Window Memory (last 2 exchanges) ===")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    window_size = 4  # 2 exchanges = 4 messages (human + AI each)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()

    history = []
    exchanges = [
        "My favorite color is blue.",
        "My favorite food is sushi.",
        "My favorite movie is Inception.",
        "What is my favorite color?",  # Should NOT remember (outside window)
        "What is my favorite movie?",  # Should remember (inside window)
    ]

    for msg in exchanges:
        # Trim to window
        windowed_history = history[-window_size:]
        print(f"  User: {msg}")
        response = chain.invoke({"input": msg, "history": windowed_history})
        print(f"  AI: {response}")
        history.append(HumanMessage(content=msg))
        history.append(AIMessage(content=response))

    print(f"  [Total messages: {len(history)}, Window: {window_size}]\n")


# ---------- Pattern 3: Summary Memory ----------

def example_summary_memory():
    """Periodically summarise older messages to compress history."""
    print("=== Pattern 3: Summary Memory ===")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Be concise.\n\n"
         "Summary of earlier conversation:\n{summary}"),
        MessagesPlaceholder("recent_history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()

    summary = "No prior conversation."
    recent_history = []
    summarize_threshold = 4  # Summarise after 4 messages

    exchanges = [
        "I'm planning a trip to Japan in April.",
        "I want to visit Tokyo, Kyoto, and Osaka.",
        "My budget is around $3000 for two weeks.",
        "I love Japanese food, especially ramen.",
        "Can you summarise my trip plans so far?",
    ]

    for msg in exchanges:
        # Summarise if recent history is too long
        if len(recent_history) >= summarize_threshold:
            summarize_prompt = ChatPromptTemplate.from_template(
                "Summarise this conversation in 2-3 sentences:\n\n"
                "Previous summary: {old_summary}\n\n"
                "Recent messages:\n{messages}"
            )
            messages_text = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in recent_history
            )
            summarize_chain = summarize_prompt | llm | StrOutputParser()
            summary = summarize_chain.invoke({
                "old_summary": summary,
                "messages": messages_text,
            })
            recent_history = []
            print(f"  [Summarised: {summary[:80]}...]")

        print(f"  User: {msg}")
        response = chain.invoke({
            "input": msg,
            "summary": summary,
            "recent_history": recent_history,
        })
        print(f"  AI: {response}")
        recent_history.append(HumanMessage(content=msg))
        recent_history.append(AIMessage(content=response))

    print(f"  [Recent messages: {len(recent_history)}, has summary: {summary != 'No prior conversation.'}]\n")


def main():
    check_api_key()
    example_buffer_memory()
    example_window_memory()
    example_summary_memory()
    print("Done!")


if __name__ == "__main__":
    main()
