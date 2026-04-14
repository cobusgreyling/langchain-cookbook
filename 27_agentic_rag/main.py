"""
Agentic RAG
============
An agent that decides when and how to retrieve information:
1. Query rewriting — reformulate vague questions
2. Retrieval — fetch relevant documents
3. Grading — judge whether retrieved docs are relevant
4. Re-query or answer — loop back or generate final answer
"""

import os
import sys
from typing import Annotated

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Sample documents ----------

DOCUMENTS = [
    Document(page_content="The Python GIL (Global Interpreter Lock) prevents multiple native threads from executing Python bytecodes simultaneously. This means CPU-bound Python programs cannot achieve true parallelism with threads. However, I/O-bound programs can still benefit from threading.", metadata={"source": "python_concurrency"}),
    Document(page_content="asyncio is Python's built-in library for writing concurrent code using async/await syntax. It uses a single-threaded event loop and is ideal for I/O-bound and network-bound applications. asyncio was introduced in Python 3.4.", metadata={"source": "asyncio_overview"}),
    Document(page_content="multiprocessing in Python spawns separate OS processes, each with its own GIL. This allows true parallelism for CPU-bound tasks. The multiprocessing module provides Pool for parallel execution and Queue/Pipe for inter-process communication.", metadata={"source": "multiprocessing_overview"}),
    Document(page_content="Python's concurrent.futures module provides ThreadPoolExecutor and ProcessPoolExecutor for managing pools of threads or processes. It offers a simple, high-level interface via the submit() and map() methods.", metadata={"source": "concurrent_futures"}),
]


# ---------- Build vector store ----------

def build_vectorstore():
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    chunks = splitter.split_documents(DOCUMENTS)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(chunks, embeddings)


# ---------- Agent state ----------

class AgentState(TypedDict):
    question: str
    rewritten_query: str
    documents: list[str]
    relevance_score: str
    answer: str
    attempts: int


# ---------- Nodes ----------

def rewrite_query(state: AgentState) -> dict:
    """Rewrite the user question to be more search-friendly."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Rewrite this question to be more specific and search-friendly. "
        "Output only the rewritten question, nothing else.\n\n"
        "Original: {question}"
    )
    chain = prompt | llm | StrOutputParser()

    query = state.get("rewritten_query") or state["question"]
    rewritten = chain.invoke({"question": query})
    attempts = state.get("attempts", 0) + 1
    print(f"  [Rewrite] '{query}' -> '{rewritten}' (attempt {attempts})")
    return {"rewritten_query": rewritten, "attempts": attempts}


def retrieve(state: AgentState) -> dict:
    """Retrieve documents from the vector store."""
    query = state["rewritten_query"]
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)
    doc_texts = [doc.page_content for doc in docs]
    print(f"  [Retrieve] Found {len(doc_texts)} documents")
    return {"documents": doc_texts}


class RelevanceGrade(BaseModel):
    grade: str = Field(description="'relevant' or 'not_relevant'")
    reason: str = Field(description="Brief explanation")


def grade_documents(state: AgentState) -> dict:
    """Grade whether retrieved documents are relevant to the question."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(RelevanceGrade)

    prompt = ChatPromptTemplate.from_template(
        "Are these documents relevant to answering the question?\n\n"
        "Question: {question}\n\n"
        "Documents:\n{documents}\n\n"
        "Grade as 'relevant' or 'not_relevant'."
    )

    docs_text = "\n---\n".join(state["documents"])
    result = structured_llm.invoke(
        prompt.format(question=state["question"], documents=docs_text)
    )
    print(f"  [Grade] {result.grade} — {result.reason}")
    return {"relevance_score": result.grade}


def generate_answer(state: AgentState) -> dict:
    """Generate an answer from the retrieved documents."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the following context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    chain = prompt | llm | StrOutputParser()
    context = "\n\n".join(state["documents"])
    answer = chain.invoke({"context": context, "question": state["question"]})
    print(f"  [Generate] Answer produced ({len(answer)} chars)")
    return {"answer": answer}


# ---------- Routing ----------

def route_after_grading(state: AgentState) -> str:
    """Route based on relevance grade and attempt count."""
    if state["relevance_score"] == "relevant":
        return "generate"
    if state.get("attempts", 0) >= 2:
        return "generate"  # Give best-effort answer after 2 attempts
    return "rewrite"


# ---------- Build graph ----------

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("rewrite", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade_documents)
    graph.add_node("generate", generate_answer)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grading, {
        "generate": "generate",
        "rewrite": "rewrite",
    })
    graph.add_edge("generate", END)

    return graph.compile()


# ---------- Main ----------

vectorstore = None


def main():
    global vectorstore
    check_api_key()

    print("Building vector store...")
    vectorstore = build_vectorstore()

    agent = build_agent()

    questions = [
        "How do I run stuff in parallel in Python?",
        "What's the deal with the GIL?",
        "How does async work in Python?",
    ]

    print("\n=== Agentic RAG ===\n")

    for q in questions:
        print(f"Q: {q}")
        result = agent.invoke({"question": q, "attempts": 0})
        print(f"A: {result['answer']}\n")
        print("-" * 60 + "\n")

    vectorstore.delete_collection()
    print("Done!")


if __name__ == "__main__":
    main()
