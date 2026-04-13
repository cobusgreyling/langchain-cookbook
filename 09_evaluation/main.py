"""
Evaluation and Testing
=======================
A lightweight evaluation framework for LangChain chains:
1. Exact-match and keyword checks
2. LLM-as-judge scoring
3. Batch evaluation with reporting
"""

import os
import sys

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Test dataset ----------

TEST_CASES = [
    {
        "input": "What is the capital of France?",
        "expected_keywords": ["paris"],
        "category": "factual",
    },
    {
        "input": "Explain photosynthesis in one sentence.",
        "expected_keywords": ["light", "energy", "plant"],
        "category": "explanation",
    },
    {
        "input": "What is 15 * 23?",
        "expected_keywords": ["345"],
        "category": "math",
    },
    {
        "input": "List three programming languages.",
        "expected_keywords": [],  # Will use LLM judge
        "category": "listing",
    },
    {
        "input": "Is the Earth flat?",
        "expected_keywords": ["no", "not flat", "sphere", "round", "oblate"],
        "category": "factual",
    },
]


# ---------- Evaluation functions ----------

def keyword_check(response: str, keywords: list[str]) -> bool:
    """Check if any of the expected keywords appear in the response."""
    if not keywords:
        return True  # No keywords to check — skip
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in keywords)


class QualityScore(BaseModel):
    """LLM judge quality assessment."""
    score: int = Field(description="Quality score from 1 to 5")
    reasoning: str = Field(description="Brief explanation for the score")


def llm_judge(question: str, answer: str, judge_llm: ChatOpenAI) -> QualityScore:
    """Use an LLM to judge the quality of an answer."""
    prompt = ChatPromptTemplate.from_template(
        "Rate this answer on a scale of 1-5 for correctness and helpfulness.\n\n"
        "Question: {question}\n"
        "Answer: {answer}\n\n"
        "Score 1 = wrong/unhelpful, 5 = perfect."
    )
    chain = prompt | judge_llm.with_structured_output(QualityScore)
    return chain.invoke({"question": question, "answer": answer})


def run_evaluation():
    """Run the full evaluation pipeline."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Build a simple Q&A chain
    prompt = ChatPromptTemplate.from_template(
        "Answer this question concisely:\n{question}"
    )
    chain = prompt | llm | StrOutputParser()

    results = []

    print("=== Running Evaluation ===\n")

    for i, test in enumerate(TEST_CASES, 1):
        question = test["input"]
        response = chain.invoke({"question": question})

        # Keyword check
        kw_pass = keyword_check(response, test["expected_keywords"])

        # LLM judge
        score = llm_judge(question, response, judge)

        result = {
            "question": question,
            "response": response[:100],
            "keyword_pass": kw_pass,
            "judge_score": score.score,
            "judge_reasoning": score.reasoning,
            "category": test["category"],
        }
        results.append(result)

        status = "PASS" if kw_pass and score.score >= 3 else "FAIL"
        print(f"  [{status}] Test {i}: {question}")
        print(f"         Response: {response[:80]}...")
        print(f"         Keywords: {'PASS' if kw_pass else 'FAIL'} | Judge: {score.score}/5")
        print(f"         Reason: {score.reasoning}\n")

    # Summary
    print("=== Summary ===")
    total = len(results)
    passed = sum(1 for r in results if r["keyword_pass"] and r["judge_score"] >= 3)
    avg_score = sum(r["judge_score"] for r in results) / total

    print(f"  Passed: {passed}/{total} ({100*passed//total}%)")
    print(f"  Average judge score: {avg_score:.1f}/5")

    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r["judge_score"])

    print("  By category:")
    for cat, scores in sorted(by_category.items()):
        avg = sum(scores) / len(scores)
        print(f"    {cat}: {avg:.1f}/5 ({len(scores)} tests)")


def main():
    check_api_key()
    run_evaluation()
    print("\nDone!")


if __name__ == "__main__":
    main()
