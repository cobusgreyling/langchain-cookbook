"""
Structured Output Parsing
==========================
Extract structured, typed data from free-form text using Pydantic models
and LangChain's with_structured_output.
"""

import os
import sys
from typing import Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Pydantic schemas ----------

class Person(BaseModel):
    """Information about a person mentioned in text."""
    name: str = Field(description="Full name of the person")
    age: Optional[int] = Field(default=None, description="Age if mentioned")
    occupation: Optional[str] = Field(default=None, description="Job or role if mentioned")


class MovieReview(BaseModel):
    """Structured movie review extracted from text."""
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating out of 10")
    genre: str = Field(description="Primary genre")
    pros: list[str] = Field(description="Positive aspects mentioned")
    cons: list[str] = Field(description="Negative aspects mentioned")
    recommended: bool = Field(description="Whether the reviewer recommends it")


class ExtractedEntities(BaseModel):
    """Named entities extracted from a text passage."""
    people: list[str] = Field(default_factory=list, description="People mentioned")
    organizations: list[str] = Field(default_factory=list, description="Organizations mentioned")
    locations: list[str] = Field(default_factory=list, description="Locations mentioned")
    dates: list[str] = Field(default_factory=list, description="Dates or time references")


def main():
    check_api_key()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # --- Example 1: Extract person info ---
    print("=== Example 1: Person Extraction ===")
    person_llm = llm.with_structured_output(Person)
    prompt1 = ChatPromptTemplate.from_template(
        "Extract person information from this text:\n{text}"
    )
    chain1 = prompt1 | person_llm

    text1 = "Dr. Sarah Chen, a 34-year-old neuroscientist at MIT, published groundbreaking research on neural plasticity."
    person = chain1.invoke({"text": text1})
    print(f"  Name: {person.name}")
    print(f"  Age: {person.age}")
    print(f"  Occupation: {person.occupation}")

    # --- Example 2: Movie review ---
    print("\n=== Example 2: Movie Review ===")
    review_llm = llm.with_structured_output(MovieReview)
    prompt2 = ChatPromptTemplate.from_template(
        "Extract a structured review from this text:\n{text}"
    )
    chain2 = prompt2 | review_llm

    text2 = (
        "Just saw Dune Part Two and wow — an 8.5/10 for me. The cinematography and world-building "
        "are absolutely stunning, and Timothée Chalamet really delivers. The score by Hans Zimmer "
        "is phenomenal. On the downside, some side characters feel underdeveloped and the pacing "
        "drags a bit in the middle. Still, this sci-fi epic is a must-watch."
    )
    review = chain2.invoke({"text": text2})
    print(f"  Title: {review.title}")
    print(f"  Rating: {review.rating}/10")
    print(f"  Genre: {review.genre}")
    print(f"  Pros: {review.pros}")
    print(f"  Cons: {review.cons}")
    print(f"  Recommended: {review.recommended}")

    # --- Example 3: Named entity extraction ---
    print("\n=== Example 3: Entity Extraction ===")
    entity_llm = llm.with_structured_output(ExtractedEntities)
    prompt3 = ChatPromptTemplate.from_template(
        "Extract all named entities from this text:\n{text}"
    )
    chain3 = prompt3 | entity_llm

    text3 = (
        "On March 15, 2024, Sundar Pichai announced that Google would open a new AI research "
        "lab in Tokyo, Japan. The lab will collaborate with the University of Tokyo and is "
        "expected to be operational by December 2024."
    )
    entities = chain3.invoke({"text": text3})
    print(f"  People: {entities.people}")
    print(f"  Organizations: {entities.organizations}")
    print(f"  Locations: {entities.locations}")
    print(f"  Dates: {entities.dates}")

    print("\nDone!")


if __name__ == "__main__":
    main()
