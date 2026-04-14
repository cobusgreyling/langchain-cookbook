"""
Hybrid Search (Vector + Keyword)
=================================
Combines BM25 keyword search with vector similarity search:
1. Build both a vector store and a BM25 retriever
2. Merge results using Reciprocal Rank Fusion
3. Use the fused results for RAG
"""

import os
import sys

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Sample documents ----------

DOCUMENTS = [
    Document(page_content="The Python GIL (Global Interpreter Lock) is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecodes at once.", metadata={"id": 1}),
    Document(page_content="asyncio is Python's built-in library for writing concurrent code using the async/await syntax. It is ideal for I/O-bound and structured network code.", metadata={"id": 2}),
    Document(page_content="Threading in Python allows concurrent execution but is limited by the GIL for CPU-bound tasks. Use multiprocessing for true parallelism.", metadata={"id": 3}),
    Document(page_content="FastAPI uses async/await natively and can handle thousands of concurrent connections efficiently using uvicorn and asyncio.", metadata={"id": 4}),
    Document(page_content="The multiprocessing module spawns separate processes, each with its own GIL, enabling true parallel execution for CPU-intensive workloads.", metadata={"id": 5}),
    Document(page_content="Celery is a distributed task queue for Python that supports scheduling, retries, and multiple message brokers like Redis and RabbitMQ.", metadata={"id": 6}),
    Document(page_content="Python 3.13 introduces an experimental free-threaded mode that removes the GIL entirely, allowing true multi-threaded parallelism.", metadata={"id": 7}),
]


# ---------- Simple BM25 retriever ----------

def bm25_search(query: str, documents: list[Document], k: int = 3) -> list[Document]:
    """Simple keyword-based search using term frequency scoring."""
    query_terms = set(query.lower().split())
    scored = []
    for doc in documents:
        doc_terms = doc.page_content.lower().split()
        # Simple TF score: count query term occurrences
        score = sum(doc_terms.count(term) for term in query_terms)
        scored.append((doc, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored[:k] if score > 0]


def reciprocal_rank_fusion(results_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.page_content[:50]  # Use content prefix as ID
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rank + k)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in sorted_docs]


def main():
    check_api_key()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Build vector store
    vectorstore = Chroma.from_documents(DOCUMENTS, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    queries = [
        "How does the GIL affect Python threading?",
        "What are the options for async programming in Python?",
        "How can I do parallel processing in Python?",
    ]

    print("=== Hybrid Search (Vector + Keyword) ===\n")

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context below.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    for query in queries:
        print(f"Q: {query}\n")

        # Vector search
        vector_results = vector_retriever.invoke(query)
        print(f"  Vector results ({len(vector_results)}):")
        for doc in vector_results:
            print(f"    - {doc.page_content[:70]}...")

        # BM25 keyword search
        bm25_results = bm25_search(query, DOCUMENTS, k=3)
        print(f"  BM25 results ({len(bm25_results)}):")
        for doc in bm25_results:
            print(f"    - {doc.page_content[:70]}...")

        # Fuse results
        fused = reciprocal_rank_fusion([vector_results, bm25_results])[:3]
        print(f"  Fused results ({len(fused)}):")
        for doc in fused:
            print(f"    - {doc.page_content[:70]}...")

        # Answer
        context = "\n\n".join(doc.page_content for doc in fused)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query})
        print(f"\n  Answer: {answer}\n")
        print("-" * 60 + "\n")

    vectorstore.delete_collection()
    print("Done!")


if __name__ == "__main__":
    main()
