"""
Error Handling and Fallbacks
==============================
Build resilient chains with retry logic and model fallbacks:
1. Automatic retries with exponential backoff
2. Model fallback chains (primary → fallback)
3. Graceful degradation with default responses
"""

import os
import sys
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Example 1: Retry with backoff ----------

def example_retry():
    """Demonstrate built-in retry behaviour on LLM calls."""
    print("=== Example 1: Retry Logic ===\n")

    # LangChain's ChatOpenAI has built-in retry via max_retries
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_retries=3,        # Retry up to 3 times on transient errors
        request_timeout=30,   # Timeout per request
    )

    prompt = ChatPromptTemplate.from_template("What is {topic}? Answer in one sentence.")
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"topic": "the Pythagorean theorem"})
    print(f"  Result (with retry enabled): {result}")
    print(f"  Config: max_retries=3, timeout=30s\n")


# ---------- Example 2: Model fallback ----------

def example_model_fallback():
    """Fall back to a cheaper/different model if the primary fails."""
    print("=== Example 2: Model Fallback Chain ===\n")

    # Primary model (simulated as working)
    primary = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Fallback model
    fallback = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    prompt = ChatPromptTemplate.from_template(
        "Explain {concept} clearly in 2 sentences."
    )

    # Chain with fallback: if primary fails, try fallback
    primary_chain = prompt | primary | StrOutputParser()
    fallback_chain = prompt | fallback | StrOutputParser()

    chain_with_fallback = primary_chain.with_fallbacks([fallback_chain])

    concepts = ["quantum entanglement", "blockchain consensus"]
    for concept in concepts:
        result = chain_with_fallback.invoke({"concept": concept})
        print(f"  {concept}: {result}\n")


# ---------- Example 3: Graceful degradation ----------

def example_graceful_degradation():
    """Return a default response when all else fails."""
    print("=== Example 3: Graceful Degradation ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def safe_invoke(chain, inputs: dict, default: str = "I'm unable to answer right now.") -> str:
        """Invoke a chain with a safety net."""
        try:
            return chain.invoke(inputs)
        except Exception as e:
            print(f"    [Error caught: {type(e).__name__}: {e}]")
            return default

    # Working chain
    prompt = ChatPromptTemplate.from_template("What is {topic}? One sentence.")
    chain = prompt | llm | StrOutputParser()

    result = safe_invoke(chain, {"topic": "gravity"})
    print(f"  Normal call: {result}\n")

    # Chain that will fail (bad model name) — caught gracefully
    bad_llm = ChatOpenAI(model="gpt-nonexistent-model", temperature=0, max_retries=0)
    bad_chain = prompt | bad_llm | StrOutputParser()

    result = safe_invoke(bad_chain, {"topic": "gravity"}, default="[Fallback] Gravity is a fundamental force of nature.")
    print(f"  Failed call: {result}\n")


# ---------- Example 4: Custom retry with backoff ----------

def with_retry_backoff(fn, max_retries: int = 3, base_delay: float = 1.0):
    """Decorator-style retry with exponential backoff."""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    raise
                delay = base_delay * (2 ** attempt)
                print(f"    [Retry {attempt + 1}/{max_retries}] {type(e).__name__}, waiting {delay:.1f}s...")
                time.sleep(delay)
    return wrapper


def example_custom_retry():
    """Custom retry logic with exponential backoff."""
    print("=== Example 4: Custom Retry with Backoff ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template("Define '{term}' in one sentence.")
    chain = prompt | llm | StrOutputParser()

    @with_retry_backoff
    def resilient_call(term: str) -> str:
        return chain.invoke({"term": term})

    terms = ["idempotency", "eventual consistency"]
    for term in terms:
        result = resilient_call(term)
        print(f"  {term}: {result}\n")


def main():
    check_api_key()
    example_retry()
    example_model_fallback()
    example_graceful_degradation()
    example_custom_retry()
    print("Done!")


if __name__ == "__main__":
    main()
