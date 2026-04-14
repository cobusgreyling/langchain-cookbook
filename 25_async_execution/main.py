"""
Async Execution
================
Run LangChain chains concurrently for better throughput:
1. Async chain invocation with ainvoke
2. Parallel batch processing with asyncio.gather
3. Async streaming
"""

import os
import sys
import asyncio
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Example 1: Sync vs Async comparison ----------

async def example_sync_vs_async():
    """Compare sequential (sync) vs concurrent (async) execution."""
    print("=== Example 1: Sync vs Async ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template("What is {topic}? One sentence.")
    chain = prompt | llm | StrOutputParser()

    topics = [
        {"topic": "photosynthesis"},
        {"topic": "the Doppler effect"},
        {"topic": "machine learning"},
        {"topic": "the Krebs cycle"},
    ]

    # Sequential
    start = time.time()
    sync_results = []
    for t in topics:
        result = chain.invoke(t)
        sync_results.append(result)
    sync_time = time.time() - start
    print(f"  Sequential: {sync_time:.2f}s for {len(topics)} calls")

    # Concurrent
    start = time.time()
    async_results = await asyncio.gather(*[chain.ainvoke(t) for t in topics])
    async_time = time.time() - start
    print(f"  Concurrent: {async_time:.2f}s for {len(topics)} calls")
    print(f"  Speedup: {sync_time / async_time:.1f}x\n")

    for topic, result in zip(topics, async_results):
        print(f"  {topic['topic']}: {result[:80]}...")
    print()


# ---------- Example 2: Async batch processing ----------

async def example_batch_processing():
    """Process a batch of items concurrently."""
    print("=== Example 2: Async Batch Processing ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Different chains for different tasks
    translate_prompt = ChatPromptTemplate.from_template(
        "Translate to {language}: {text}"
    )
    translate_chain = translate_prompt | llm | StrOutputParser()

    items = [
        {"text": "Hello, how are you?", "language": "French"},
        {"text": "Hello, how are you?", "language": "Spanish"},
        {"text": "Hello, how are you?", "language": "Japanese"},
        {"text": "Hello, how are you?", "language": "German"},
        {"text": "Hello, how are you?", "language": "Portuguese"},
    ]

    start = time.time()
    results = await asyncio.gather(*[translate_chain.ainvoke(item) for item in items])
    elapsed = time.time() - start

    for item, result in zip(items, results):
        print(f"  {item['language']}: {result}")
    print(f"\n  Translated {len(items)} languages in {elapsed:.2f}s\n")


# ---------- Example 3: Async streaming ----------

async def example_async_streaming():
    """Stream tokens asynchronously."""
    print("=== Example 3: Async Streaming ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Write a haiku about {subject}."
    )
    chain = prompt | llm | StrOutputParser()

    print("  Subject: programming\n  ", end="")
    async for chunk in chain.astream({"subject": "programming"}):
        print(chunk, end="", flush=True)
    print("\n")


# ---------- Example 4: Async with multiple chains ----------

async def example_multi_chain_async():
    """Run entirely different chains concurrently."""
    print("=== Example 4: Multi-Chain Concurrent Execution ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Three different chains
    summary_chain = (
        ChatPromptTemplate.from_template("Summarise in one sentence: {text}")
        | llm | StrOutputParser()
    )
    keyword_chain = (
        ChatPromptTemplate.from_template("Extract 3 keywords from: {text}")
        | llm | StrOutputParser()
    )
    sentiment_chain = (
        ChatPromptTemplate.from_template("What is the sentiment (positive/negative/neutral) of: {text}")
        | llm | StrOutputParser()
    )

    text = (
        "LangChain has become one of the most popular frameworks for building LLM applications. "
        "Its modular design and excellent documentation make it accessible to developers of all levels."
    )

    start = time.time()
    summary, keywords, sentiment = await asyncio.gather(
        summary_chain.ainvoke({"text": text}),
        keyword_chain.ainvoke({"text": text}),
        sentiment_chain.ainvoke({"text": text}),
    )
    elapsed = time.time() - start

    print(f"  Input: {text[:60]}...")
    print(f"  Summary: {summary}")
    print(f"  Keywords: {keywords}")
    print(f"  Sentiment: {sentiment}")
    print(f"  All 3 chains completed in {elapsed:.2f}s\n")


async def async_main():
    await example_sync_vs_async()
    await example_batch_processing()
    await example_async_streaming()
    await example_multi_chain_async()


def main():
    check_api_key()
    asyncio.run(async_main())
    print("Done!")


if __name__ == "__main__":
    main()
