"""
Callbacks and Observability
============================
Monitor and debug LangChain chains with callbacks:
1. Custom callback handler for logging
2. Token counting and cost tracking
3. Chain execution tracing
"""

import os
import sys
import time
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Custom callback handler ----------

class ObservabilityHandler(BaseCallbackHandler):
    """Tracks LLM calls, latency, token usage, and chain execution."""

    def __init__(self):
        self.calls = []
        self.total_tokens = 0
        self.total_cost = 0.0
        self._start_times = {}

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
        run_id = kwargs.get("run_id", "unknown")
        self._start_times[str(run_id)] = time.time()
        print(f"    [LLM Start] Model call initiated (run_id: {str(run_id)[:8]})")

    def on_llm_end(self, response: LLMResult, **kwargs):
        run_id = str(kwargs.get("run_id", "unknown"))
        elapsed = time.time() - self._start_times.pop(run_id, time.time())

        # Extract token usage if available
        token_usage = {}
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})

        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total = prompt_tokens + completion_tokens

        # Estimate cost (gpt-4o-mini pricing)
        cost = (prompt_tokens * 0.15 + completion_tokens * 0.6) / 1_000_000

        self.total_tokens += total
        self.total_cost += cost
        self.calls.append({
            "latency_ms": round(elapsed * 1000),
            "tokens": total,
            "cost_usd": cost,
        })

        print(f"    [LLM End] {elapsed:.2f}s | {total} tokens | ${cost:.6f}")

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        chain_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        print(f"    [Chain Start] {chain_name}")

    def on_chain_end(self, outputs: dict, **kwargs):
        pass

    def on_llm_error(self, error: BaseException, **kwargs):
        print(f"    [LLM Error] {error}")

    def summary(self) -> str:
        return (
            f"Total calls: {len(self.calls)} | "
            f"Total tokens: {self.total_tokens} | "
            f"Total cost: ${self.total_cost:.6f} | "
            f"Avg latency: {sum(c['latency_ms'] for c in self.calls) / max(len(self.calls), 1):.0f}ms"
        )


# ---------- Token counter callback ----------

class TokenCounter(BaseCallbackHandler):
    """Simple callback that just counts tokens per call."""

    def __init__(self):
        self.log = []

    def on_llm_end(self, response: LLMResult, **kwargs):
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.log.append({
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
            })


def main():
    check_api_key()

    handler = ObservabilityHandler()
    counter = TokenCounter()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        callbacks=[handler, counter],
    )

    # --- Example 1: Simple chain with observability ---
    print("=== Example 1: Observed Chain Execution ===\n")
    prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
    chain = prompt | llm | StrOutputParser()

    topics = ["recursion", "hash tables", "Big O notation"]
    for topic in topics:
        print(f"  Topic: {topic}")
        result = chain.invoke({"topic": topic}, config={"callbacks": [handler, counter]})
        print(f"  Result: {result}\n")

    # --- Example 2: Multi-step chain ---
    print("=== Example 2: Multi-Step Chain ===\n")
    step1 = ChatPromptTemplate.from_template(
        "Generate a quiz question about {topic}."
    ) | llm | StrOutputParser()

    step2 = ChatPromptTemplate.from_template(
        "Answer this quiz question concisely:\n{question}"
    ) | llm | StrOutputParser()

    print("  Topic: Python decorators")
    question = step1.invoke({"topic": "Python decorators"}, config={"callbacks": [handler, counter]})
    print(f"  Generated Q: {question}")
    answer = step2.invoke({"question": question}, config={"callbacks": [handler, counter]})
    print(f"  Answer: {answer}\n")

    # --- Summary ---
    print("=== Observability Summary ===")
    print(f"  {handler.summary()}")
    print(f"  Token breakdown per call:")
    for i, entry in enumerate(counter.log):
        print(f"    Call {i+1}: {entry['prompt']} prompt + {entry['completion']} completion")

    print("\nDone!")


if __name__ == "__main__":
    main()
