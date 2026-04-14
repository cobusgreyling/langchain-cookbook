"""
Few-Shot Prompting
===================
Teach LLMs by example:
1. Static few-shot prompts
2. Dynamic example selection based on similarity
3. Few-shot with structured output
"""

import os
import sys

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from pydantic import BaseModel, Field


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Example 1: Static few-shot ----------

def example_static_few_shot():
    """Provide fixed examples in the prompt."""
    print("=== Example 1: Static Few-Shot ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "fast", "output": "slow"},
        {"input": "bright", "output": "dim"},
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You give the antonym of every word the user provides."),
        few_shot_prompt,
        ("human", "{input}"),
    ])

    chain = final_prompt | llm | StrOutputParser()

    test_words = ["brave", "generous", "ancient"]
    for word in test_words:
        result = chain.invoke({"input": word})
        print(f"  {word} → {result}")

    print()


# ---------- Example 2: Dynamic example selection ----------

def example_dynamic_selection():
    """Select the most relevant examples based on input similarity."""
    print("=== Example 2: Dynamic Example Selection ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    examples = [
        {"input": "What is 2+2?", "output": "The answer is 4."},
        {"input": "What is the capital of France?", "output": "The capital of France is Paris."},
        {"input": "How do I reverse a list in Python?", "output": "Use my_list[::-1] or my_list.reverse()."},
        {"input": "What is photosynthesis?", "output": "Photosynthesis is the process by which plants convert sunlight into energy."},
        {"input": "How do I read a file in Python?", "output": "Use open('file.txt', 'r') with a context manager: with open('file.txt') as f: content = f.read()"},
        {"input": "What is gravity?", "output": "Gravity is a fundamental force that attracts objects with mass toward each other."},
    ]

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        embeddings,
        Chroma,
        k=2,
    )

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selector,
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer questions concisely, following the style of the examples."),
        few_shot_prompt,
        ("human", "{input}"),
    ])

    chain = final_prompt | llm | StrOutputParser()

    test_questions = [
        "How do I sort a dictionary in Python?",
        "What is the speed of light?",
        "What is 15 * 7?",
    ]

    for q in test_questions:
        # Show which examples were selected
        selected = example_selector.select_examples({"input": q})
        selected_inputs = [ex["input"] for ex in selected]
        result = chain.invoke({"input": q})
        print(f"  Q: {q}")
        print(f"  Selected examples: {selected_inputs}")
        print(f"  A: {result}\n")


# ---------- Example 3: Few-shot with structured output ----------

class SentimentResult(BaseModel):
    """Sentiment analysis result."""
    sentiment: str = Field(description="'positive', 'negative', or 'neutral'")
    confidence: float = Field(description="Confidence from 0.0 to 1.0")
    key_phrase: str = Field(description="The phrase that most indicates the sentiment")


def example_few_shot_structured():
    """Combine few-shot prompting with structured output."""
    print("=== Example 3: Few-Shot + Structured Output ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    examples = [
        {"input": "This product is absolutely amazing!", "output": "sentiment: positive, confidence: 0.95, key_phrase: 'absolutely amazing'"},
        {"input": "Terrible experience, would not recommend.", "output": "sentiment: negative, confidence: 0.90, key_phrase: 'Terrible experience'"},
        {"input": "It's okay, nothing special.", "output": "sentiment: neutral, confidence: 0.70, key_phrase: 'nothing special'"},
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a sentiment analyser. Analyse the sentiment of the given text."),
        few_shot_prompt,
        ("human", "{input}"),
    ])

    chain = final_prompt | llm.with_structured_output(SentimentResult)

    test_texts = [
        "I love how easy this is to use!",
        "The delivery was late and the item was damaged.",
        "The product works as described.",
    ]

    for text in test_texts:
        result = chain.invoke({"input": text})
        print(f"  Text: {text}")
        print(f"  Result: {result.sentiment} ({result.confidence:.0%}) — '{result.key_phrase}'\n")


def main():
    check_api_key()
    example_static_few_shot()
    example_dynamic_selection()
    example_few_shot_structured()
    print("Done!")


if __name__ == "__main__":
    main()
