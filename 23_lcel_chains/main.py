"""
LCEL (LangChain Expression Language)
======================================
Compose chains declaratively using the pipe operator:
1. Basic chain composition
2. Parallel execution with RunnableParallel
3. Branching with RunnableBranch
4. Passthrough and lambda runnables
"""

import os
import sys

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch,
)


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Example 1: Basic pipe composition ----------

def example_basic_pipe():
    """Chain components together with the | operator."""
    print("=== Example 1: Basic Pipe Composition ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Prompt | LLM | Parser
    chain = (
        ChatPromptTemplate.from_template("Explain {concept} in one sentence for a {audience}.")
        | llm
        | StrOutputParser()
    )

    inputs = [
        {"concept": "recursion", "audience": "5-year-old"},
        {"concept": "recursion", "audience": "senior engineer"},
    ]

    for inp in inputs:
        result = chain.invoke(inp)
        print(f"  [{inp['audience']}] {result}")

    print()


# ---------- Example 2: Parallel execution ----------

def example_parallel():
    """Run multiple chains in parallel and combine results."""
    print("=== Example 2: RunnableParallel ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    pros_chain = (
        ChatPromptTemplate.from_template("List 2 pros of {technology}. Be brief.")
        | llm | StrOutputParser()
    )
    cons_chain = (
        ChatPromptTemplate.from_template("List 2 cons of {technology}. Be brief.")
        | llm | StrOutputParser()
    )
    summary_prompt = ChatPromptTemplate.from_template(
        "Given these pros and cons, write a one-sentence verdict.\n\nPros: {pros}\nCons: {cons}"
    )

    # Run pros and cons in parallel, then summarise
    parallel = RunnableParallel(pros=pros_chain, cons=cons_chain)

    full_chain = (
        {"technology": RunnablePassthrough()}
        | parallel
        | summary_prompt
        | llm
        | StrOutputParser()
    )

    techs = ["microservices", "monoliths"]
    for tech in techs:
        # Show intermediate results
        intermediate = parallel.invoke({"technology": tech})
        verdict = full_chain.invoke(tech)
        print(f"  {tech}:")
        print(f"    Pros: {intermediate['pros'][:80]}...")
        print(f"    Cons: {intermediate['cons'][:80]}...")
        print(f"    Verdict: {verdict}\n")


# ---------- Example 3: Branching ----------

def example_branching():
    """Route inputs to different chains based on conditions."""
    print("=== Example 3: RunnableBranch ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Different chains for different input types
    math_chain = (
        ChatPromptTemplate.from_template("Solve this math problem step by step:\n{input}")
        | llm | StrOutputParser()
    )
    code_chain = (
        ChatPromptTemplate.from_template("Write Python code for this:\n{input}")
        | llm | StrOutputParser()
    )
    general_chain = (
        ChatPromptTemplate.from_template("Answer concisely:\n{input}")
        | llm | StrOutputParser()
    )

    def classify(inp: dict) -> str:
        text = inp["input"].lower()
        if any(w in text for w in ["calculate", "solve", "math", "sum", "multiply"]):
            return "math"
        if any(w in text for w in ["code", "function", "program", "implement", "write a"]):
            return "code"
        return "general"

    branch = RunnableBranch(
        (lambda x: classify(x) == "math", math_chain),
        (lambda x: classify(x) == "code", code_chain),
        general_chain,  # Default
    )

    queries = [
        "Calculate the sum of first 10 prime numbers",
        "Write a function to check if a string is a palindrome",
        "What is the capital of Australia?",
    ]

    for q in queries:
        category = classify({"input": q})
        result = branch.invoke({"input": q})
        print(f"  [{category}] Q: {q}")
        print(f"  A: {result[:120]}...\n")


# ---------- Example 4: Lambda runnables ----------

def example_lambda():
    """Use RunnableLambda for custom transformations in a chain."""
    print("=== Example 4: RunnableLambda ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Custom preprocessing
    def preprocess(input_dict: dict) -> dict:
        text = input_dict["text"]
        return {
            "text": text.strip().lower(),
            "word_count": len(text.split()),
            "char_count": len(text),
        }

    # Custom postprocessing
    def postprocess(result: str) -> dict:
        return {
            "summary": result,
            "summary_length": len(result),
        }

    chain = (
        RunnableLambda(preprocess)
        | ChatPromptTemplate.from_template(
            "Summarise this text ({word_count} words, {char_count} chars) in one sentence:\n\n{text}"
        )
        | llm
        | StrOutputParser()
        | RunnableLambda(postprocess)
    )

    text = "  LangChain Expression Language allows you to compose chains using the pipe operator. It supports parallel execution, branching, and custom transformations.  "
    result = chain.invoke({"text": text})
    print(f"  Input: {text.strip()[:60]}...")
    print(f"  Summary: {result['summary']}")
    print(f"  Summary length: {result['summary_length']} chars\n")


def main():
    check_api_key()
    example_basic_pipe()
    example_parallel()
    example_branching()
    example_lambda()
    print("Done!")


if __name__ == "__main__":
    main()
