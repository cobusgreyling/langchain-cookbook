"""
Caching and Cost Optimization
===============================
Reduce LLM costs and latency:
1. In-memory response caching
2. SQLite persistent cache
3. Token budgeting and prompt compression
"""

import os
import sys
import time
import tempfile

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Example 1: In-Memory Cache ----------

def example_inmemory_cache():
    """Cache LLM responses in memory to avoid redundant calls."""
    print("=== Example 1: In-Memory Cache ===\n")

    set_llm_cache(InMemoryCache())

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")
    chain = prompt | llm | StrOutputParser()

    queries = [
        "France",
        "Japan",
        "France",  # Should be cached
        "Japan",   # Should be cached
        "Brazil",
    ]

    for country in queries:
        start = time.time()
        result = chain.invoke({"country": country})
        elapsed = time.time() - start
        cached = "CACHED" if elapsed < 0.1 else "API"
        print(f"  [{cached}] {country}: {result} ({elapsed:.3f}s)")

    # Clear cache
    set_llm_cache(None)
    print()


# ---------- Example 2: Token Budgeting ----------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English text."""
    return len(text) // 4


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget."""
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    # Truncate proportionally
    max_chars = max_tokens * 4
    return text[:max_chars] + "... [truncated]"


def example_token_budgeting():
    """Demonstrate token budgeting for cost control."""
    print("=== Example 2: Token Budgeting ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Simulated long document
    long_document = (
        "LangChain is a framework for developing applications powered by language models. "
        "It provides modular components for prompts, chains, agents, memory, and retrieval. "
        "LangChain supports multiple LLM providers and can be used with both cloud and local models. "
    ) * 20  # Repeat to simulate a long document

    token_budgets = [50, 100, 200]

    for budget in token_budgets:
        truncated = truncate_to_budget(long_document, budget)
        estimated = estimate_tokens(truncated)

        prompt = ChatPromptTemplate.from_template(
            "Summarise this text in one sentence:\n\n{text}"
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"text": truncated})

        print(f"  Budget: {budget} tokens | Used: ~{estimated} tokens")
        print(f"  Summary: {result}\n")


# ---------- Example 3: Prompt Compression ----------

def compress_prompt(text: str, llm: ChatOpenAI) -> str:
    """Use an LLM to compress a verbose prompt while preserving meaning."""
    prompt = ChatPromptTemplate.from_template(
        "Compress this text to be as short as possible while keeping all key information. "
        "Remove filler words, redundancy, and unnecessary detail.\n\n"
        "Original text:\n{text}\n\n"
        "Compressed text:"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


def example_prompt_compression():
    """Demonstrate prompt compression for cost savings."""
    print("=== Example 3: Prompt Compression ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    verbose_prompts = [
        (
            "I would really appreciate it if you could please help me understand "
            "what exactly the concept of machine learning is all about, and if you "
            "could explain it in a way that is simple and easy for beginners to understand, "
            "that would be absolutely wonderful and very helpful."
        ),
        (
            "Could you please provide me with a comprehensive and detailed explanation "
            "of the differences and similarities between Python lists and Python tuples, "
            "including when you would want to use one over the other in your code?"
        ),
    ]

    for original in verbose_prompts:
        compressed = compress_prompt(original, llm)
        original_tokens = estimate_tokens(original)
        compressed_tokens = estimate_tokens(compressed)
        savings = (1 - compressed_tokens / original_tokens) * 100

        print(f"  Original ({original_tokens} tokens): {original[:80]}...")
        print(f"  Compressed ({compressed_tokens} tokens): {compressed}")
        print(f"  Savings: {savings:.0f}%\n")


def main():
    check_api_key()
    example_inmemory_cache()
    example_token_budgeting()
    example_prompt_compression()
    print("Done!")


if __name__ == "__main__":
    main()
