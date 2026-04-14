"""
RAG with Reranking
===================
Retrieval-Augmented Generation with a reranking step:
1. Retrieve a broad set of candidates via vector search
2. Rerank candidates using an LLM to pick the most relevant
3. Answer using only the top reranked documents
"""

import os
import sys

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Sample documents ----------

DOCUMENTS = [
    Document(page_content="Python 3.12 introduced several performance improvements, including a new specializing adaptive interpreter.", metadata={"source": "python_release"}),
    Document(page_content="Django 5.0 added facet filters in the admin, simplified form field rendering, and database-generated model fields.", metadata={"source": "django_release"}),
    Document(page_content="FastAPI is a modern web framework for building APIs with Python, based on standard Python type hints.", metadata={"source": "fastapi_overview"}),
    Document(page_content="Flask is a lightweight WSGI web framework. It is designed with simplicity and extensibility in mind.", metadata={"source": "flask_overview"}),
    Document(page_content="Python's GIL (Global Interpreter Lock) prevents true multi-threading for CPU-bound tasks. Python 3.13 experiments with a no-GIL build.", metadata={"source": "python_gil"}),
    Document(page_content="NumPy 2.0 introduced a new string dtype, improved type promotion rules, and removed many deprecated features.", metadata={"source": "numpy_release"}),
    Document(page_content="LangChain's expression language (LCEL) allows composing chains with the pipe operator for declarative workflows.", metadata={"source": "langchain_lcel"}),
    Document(page_content="Pydantic V2 is a ground-up rewrite in Rust, offering 5-50x speed improvements over V1 for data validation.", metadata={"source": "pydantic_v2"}),
]


# ---------- LLM-based reranker ----------

class RerankedDoc(BaseModel):
    """A reranked document with a relevance score."""
    index: int = Field(description="Original index of the document")
    relevance: float = Field(description="Relevance score from 0.0 to 1.0")


class RerankedResults(BaseModel):
    """Collection of reranked documents."""
    documents: list[RerankedDoc] = Field(description="Documents ranked by relevance")


def rerank_documents(query: str, docs: list[Document], llm: ChatOpenAI, top_k: int = 3) -> list[Document]:
    """Rerank documents by relevance to the query using an LLM."""
    doc_descriptions = "\n".join(
        f"[{i}] {doc.page_content[:200]}" for i, doc in enumerate(docs)
    )

    prompt = ChatPromptTemplate.from_template(
        "Given the query, score each document's relevance from 0.0 to 1.0.\n\n"
        "Query: {query}\n\n"
        "Documents:\n{documents}\n\n"
        "Return all documents with their relevance scores."
    )

    chain = prompt | llm.with_structured_output(RerankedResults)
    result = chain.invoke({"query": query, "documents": doc_descriptions})

    # Sort by relevance and take top_k
    ranked = sorted(result.documents, key=lambda d: d.relevance, reverse=True)
    return [(docs[r.index], r.relevance) for r in ranked[:top_k] if r.index < len(docs)]


def main():
    check_api_key()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Build vector store
    vectorstore = Chroma.from_documents(DOCUMENTS, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    queries = [
        "What performance improvements were made to Python recently?",
        "Which web frameworks are available for Python?",
    ]

    print("=== RAG with Reranking ===\n")

    for query in queries:
        print(f"Q: {query}\n")

        # Step 1: Broad retrieval
        candidates = retriever.invoke(query)
        print(f"  Retrieved {len(candidates)} candidates:")
        for i, doc in enumerate(candidates):
            print(f"    [{i}] {doc.page_content[:80]}...")

        # Step 2: Rerank
        reranked = rerank_documents(query, candidates, llm, top_k=2)
        print(f"\n  Top {len(reranked)} after reranking:")
        for doc, score in reranked:
            print(f"    [{score:.2f}] {doc.page_content[:80]}...")

        # Step 3: Answer with reranked docs
        context = "\n\n".join(doc.page_content for doc, _ in reranked)
        answer_prompt = ChatPromptTemplate.from_template(
            "Answer the question using only the context below.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
        answer_chain = answer_prompt | llm | StrOutputParser()
        answer = answer_chain.invoke({"context": context, "question": query})
        print(f"\n  Answer: {answer}\n")
        print("-" * 60 + "\n")

    vectorstore.delete_collection()
    print("Done!")


if __name__ == "__main__":
    main()
