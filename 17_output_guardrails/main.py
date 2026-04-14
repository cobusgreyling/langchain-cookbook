"""
Output Guardrails and Moderation
==================================
Validate and filter LLM outputs:
1. PII detection and redaction
2. Content moderation (toxicity check)
3. Output format validation with retry
"""

import os
import sys
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Guardrail 1: PII Detection ----------

PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
}


def detect_pii(text: str) -> list[dict]:
    """Detect PII patterns in text."""
    findings = []
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        for match in matches:
            findings.append({"type": pii_type, "value": match})
    return findings


def redact_pii(text: str) -> str:
    """Replace PII with redaction markers."""
    redacted = text
    for pii_type, pattern in PII_PATTERNS.items():
        redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)
    return redacted


# ---------- Guardrail 2: Content Moderation ----------

class ModerationResult(BaseModel):
    """Content moderation assessment."""
    is_safe: bool = Field(description="Whether the content is safe to display")
    category: str = Field(description="Category: 'safe', 'toxic', 'harmful', 'inappropriate'")
    reason: str = Field(description="Brief explanation")


def moderate_content(text: str, llm: ChatOpenAI) -> ModerationResult:
    """Use an LLM to check if content is appropriate."""
    prompt = ChatPromptTemplate.from_template(
        "You are a content moderator. Assess whether this text is safe for a general audience.\n\n"
        "Text: {text}\n\n"
        "Evaluate for: toxicity, harmful advice, inappropriate content, or bias."
    )
    chain = prompt | llm.with_structured_output(ModerationResult)
    return chain.invoke({"text": text})


# ---------- Guardrail 3: Output Validation with Retry ----------

class ValidatedAnswer(BaseModel):
    """A validated, structured answer."""
    answer: str = Field(description="The answer text")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    sources_cited: bool = Field(description="Whether the answer cites sources")


def answer_with_validation(question: str, llm: ChatOpenAI, max_retries: int = 2) -> str:
    """Generate an answer and validate its quality, retrying if needed."""
    prompt = ChatPromptTemplate.from_template(
        "Answer this question accurately and cite your reasoning.\n\n"
        "Question: {question}\n\n"
        "Provide a confident, well-sourced answer."
    )

    chain = prompt | llm.with_structured_output(ValidatedAnswer)

    for attempt in range(max_retries + 1):
        result = chain.invoke({"question": question})

        if result.confidence >= 0.7:
            return f"[Confidence: {result.confidence:.0%}] {result.answer}"

        print(f"    [Retry {attempt + 1}] Low confidence ({result.confidence:.0%}), retrying...")

    return f"[Low confidence: {result.confidence:.0%}] {result.answer}"


def main():
    check_api_key()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # --- Example 1: PII Detection and Redaction ---
    print("=== Guardrail 1: PII Detection ===\n")
    test_texts = [
        "Contact John at john.doe@example.com or call 555-123-4567.",
        "My SSN is 123-45-6789 and my card is 4111 1111 1111 1111.",
        "The meeting is at 3pm in conference room B.",
    ]

    for text in test_texts:
        pii = detect_pii(text)
        redacted = redact_pii(text)
        status = f"Found {len(pii)} PII items" if pii else "Clean"
        print(f"  [{status}] Original:  {text}")
        if pii:
            print(f"           Redacted:  {redacted}")
            for item in pii:
                print(f"           Detected:  {item['type']}: {item['value']}")
        print()

    # --- Example 2: Content Moderation ---
    print("=== Guardrail 2: Content Moderation ===\n")
    contents = [
        "Python is a great programming language for beginners and experts alike.",
        "Here are some tips for improving your code quality and maintainability.",
    ]

    for content in contents:
        result = moderate_content(content, llm)
        status = "SAFE" if result.is_safe else "FLAGGED"
        print(f"  [{status}] {content[:60]}...")
        print(f"           Category: {result.category} — {result.reason}\n")

    # --- Example 3: Validated Output with Retry ---
    print("=== Guardrail 3: Output Validation ===\n")
    questions = [
        "What is the speed of light in a vacuum?",
        "When was the Python programming language first released?",
    ]

    for q in questions:
        print(f"  Q: {q}")
        answer = answer_with_validation(q, llm)
        print(f"  A: {answer}\n")

    print("Done!")


if __name__ == "__main__":
    main()
