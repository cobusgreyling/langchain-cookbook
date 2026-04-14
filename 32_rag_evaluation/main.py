"""
RAG Evaluation with RAGAS-style Metrics
=========================================
Systematically evaluate a RAG pipeline:
1. Build a simple RAG chain
2. Run it on a test set with known answers
3. Measure faithfulness, relevance, and correctness
4. Report scores
"""

import os
import sys

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Knowledge base ----------

DOCUMENTS = [
    Document(page_content="Python 3.12 was released on October 2, 2023. Key features include improved error messages with more helpful suggestions, a new type parameter syntax (PEP 695), and the removal of the distutils package.", metadata={"source": "python_releases"}),
    Document(page_content="FastAPI is a modern Python web framework for building APIs. It uses Python type hints for automatic validation, serialization, and documentation. FastAPI is built on Starlette and Pydantic.", metadata={"source": "fastapi_overview"}),
    Document(page_content="Django is a high-level Python web framework that encourages rapid development. It follows the model-template-view (MTV) pattern. Django includes an ORM, admin interface, and authentication system out of the box.", metadata={"source": "django_overview"}),
    Document(page_content="Flask is a lightweight Python web framework. It is classified as a microframework because it does not require particular tools or libraries. Flask supports extensions for adding functionality like database integration and form validation.", metadata={"source": "flask_overview"}),
]


# ---------- Test dataset ----------

TEST_SET = [
    {
        "question": "When was Python 3.12 released?",
        "ground_truth": "Python 3.12 was released on October 2, 2023.",
        "expected_source": "python_releases",
    },
    {
        "question": "What is FastAPI built on?",
        "ground_truth": "FastAPI is built on Starlette and Pydantic.",
        "expected_source": "fastapi_overview",
    },
    {
        "question": "What pattern does Django follow?",
        "ground_truth": "Django follows the model-template-view (MTV) pattern.",
        "expected_source": "django_overview",
    },
    {
        "question": "Why is Flask called a microframework?",
        "ground_truth": "Flask is called a microframework because it does not require particular tools or libraries.",
        "expected_source": "flask_overview",
    },
]


# ---------- Build RAG chain ----------

def build_rag_chain():
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(DOCUMENTS)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n"
        "{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever, vectorstore


# ---------- Evaluation metrics (LLM-as-judge) ----------

class FaithfulnessScore(BaseModel):
    score: float = Field(description="Score from 0.0 to 1.0")
    reason: str = Field(description="Brief explanation")


class RelevanceScore(BaseModel):
    score: float = Field(description="Score from 0.0 to 1.0")
    reason: str = Field(description="Brief explanation")


class CorrectnessScore(BaseModel):
    score: float = Field(description="Score from 0.0 to 1.0")
    reason: str = Field(description="Brief explanation")


def evaluate_faithfulness(question: str, answer: str, context: str) -> FaithfulnessScore:
    """Is the answer grounded in the provided context (no hallucination)?"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(FaithfulnessScore)

    prompt = ChatPromptTemplate.from_template(
        "Rate how faithfully the answer is grounded in the context (0.0 = hallucinated, 1.0 = fully grounded).\n\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer: {answer}\n"
    )
    return structured_llm.invoke(prompt.format(context=context, question=question, answer=answer))


def evaluate_relevance(question: str, context: str) -> RelevanceScore:
    """Is the retrieved context relevant to the question?"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(RelevanceScore)

    prompt = ChatPromptTemplate.from_template(
        "Rate how relevant the retrieved context is to the question (0.0 = irrelevant, 1.0 = perfectly relevant).\n\n"
        "Question: {question}\n"
        "Retrieved context: {context}\n"
    )
    return structured_llm.invoke(prompt.format(question=question, context=context))


def evaluate_correctness(answer: str, ground_truth: str) -> CorrectnessScore:
    """Does the answer match the ground truth?"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(CorrectnessScore)

    prompt = ChatPromptTemplate.from_template(
        "Rate how correct the answer is compared to the ground truth (0.0 = wrong, 1.0 = correct).\n\n"
        "Ground truth: {ground_truth}\n"
        "Answer: {answer}\n"
    )
    return structured_llm.invoke(prompt.format(answer=answer, ground_truth=ground_truth))


# ---------- Main ----------

def main():
    check_api_key()

    print("Building RAG pipeline...")
    chain, retriever, vectorstore = build_rag_chain()

    print("Running evaluation...\n")
    print(f"{'Question':<45} {'Faith':>6} {'Relev':>6} {'Correct':>8}")
    print("-" * 70)

    totals = {"faithfulness": 0, "relevance": 0, "correctness": 0}

    for test in TEST_SET:
        question = test["question"]
        ground_truth = test["ground_truth"]

        # Get answer and retrieved context
        answer = chain.invoke(question)
        docs = retriever.invoke(question)
        context = "\n".join(doc.page_content for doc in docs)

        # Evaluate
        faith = evaluate_faithfulness(question, answer, context)
        relev = evaluate_relevance(question, context)
        correct = evaluate_correctness(answer, ground_truth)

        totals["faithfulness"] += faith.score
        totals["relevance"] += relev.score
        totals["correctness"] += correct.score

        q_short = question[:42] + "..." if len(question) > 45 else question
        print(f"{q_short:<45} {faith.score:>6.2f} {relev.score:>6.2f} {correct.score:>8.2f}")

    n = len(TEST_SET)
    print("-" * 70)
    print(f"{'AVERAGE':<45} {totals['faithfulness']/n:>6.2f} {totals['relevance']/n:>6.2f} {totals['correctness']/n:>8.2f}")

    print("\nDetailed breakdown:")
    print(f"  Faithfulness: {totals['faithfulness']/n:.2f} — Are answers grounded in context?")
    print(f"  Relevance:    {totals['relevance']/n:.2f} — Is retrieved context on-topic?")
    print(f"  Correctness:  {totals['correctness']/n:.2f} — Do answers match ground truth?")

    vectorstore.delete_collection()
    print("\nDone!")


if __name__ == "__main__":
    main()
