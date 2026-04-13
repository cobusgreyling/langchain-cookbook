"""
Streaming Responses
====================
Demonstrates three streaming patterns:
1. Direct LLM streaming
2. Chain streaming (prompt → LLM → parser)
3. Agent event streaming
"""

import os
import sys

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


def example_1_direct_streaming():
    """Stream tokens directly from the LLM."""
    print("=== Example 1: Direct LLM Streaming ===")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    print("Streaming: ", end="", flush=True)
    for chunk in llm.stream([HumanMessage(content="Write a haiku about programming.")]):
        print(chunk.content, end="", flush=True)
    print("\n")


def example_2_chain_streaming():
    """Stream through a full chain."""
    print("=== Example 2: Chain Streaming ===")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in exactly 3 bullet points. Be concise."
    )
    chain = prompt | llm | StrOutputParser()

    print("Streaming: ", end="", flush=True)
    for chunk in chain.stream({"topic": "how neural networks learn"}):
        print(chunk, end="", flush=True)
    print("\n")


def example_3_async_stream_events():
    """Stream events from a chain to see intermediate steps."""
    import asyncio

    async def run():
        print("=== Example 3: Stream Events ===")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        prompt = ChatPromptTemplate.from_template(
            "List 3 fun facts about {topic}."
        )
        chain = prompt | llm | StrOutputParser()

        print("Events:")
        async for event in chain.astream_events(
            {"topic": "octopuses"}, version="v2"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="", flush=True)
            elif kind == "on_chain_start":
                print(f"  [Chain started: {event['name']}]")
            elif kind == "on_chain_end":
                if event["name"] == "RunnableSequence":
                    print(f"\n  [Chain finished]")
        print()

    asyncio.run(run())


def main():
    check_api_key()
    example_1_direct_streaming()
    example_2_chain_streaming()
    example_3_async_stream_events()
    print("Done!")


if __name__ == "__main__":
    main()
