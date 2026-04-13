"""
Basic RAG with ChromaDB
=======================
A minimal Retrieval-Augmented Generation pipeline:
1. Load documents (in-memory text for portability)
2. Split into chunks
3. Embed and store in ChromaDB
4. Query with a retrieval chain
"""

import os
import sys

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Sample documents ----------

DOCUMENTS = [
    Document(
        page_content=(
            "Python was created by Guido van Rossum and first released in 1991. "
            "It emphasises code readability and supports multiple programming paradigms, "
            "including procedural, object-oriented, and functional programming. "
            "Python's design philosophy is summarised by the document The Zen of Python."
        ),
        metadata={"source": "python_overview"},
    ),
    Document(
        page_content=(
            "LangChain is a framework for developing applications powered by large language "
            "models. It provides tools for prompt management, chains, agents, memory, and "
            "retrieval-augmented generation. LangChain supports multiple LLM providers "
            "including OpenAI, Anthropic, and local models via Ollama."
        ),
        metadata={"source": "langchain_overview"},
    ),
    Document(
        page_content=(
            "ChromaDB is an open-source vector database designed for AI applications. "
            "It stores embeddings alongside metadata and documents, enabling fast "
            "similarity search. Chroma can run in-memory for development or as a "
            "persistent server for production workloads."
        ),
        metadata={"source": "chromadb_overview"},
    ),
]


def build_rag_chain():
    """Build and return a RAG chain."""
    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_documents(DOCUMENTS)

    # Create vector store (in-memory)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n"
        "{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    # Build chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, vectorstore


def main():
    check_api_key()

    print("Building RAG pipeline...")
    chain, vectorstore = build_rag_chain()

    questions = [
        "Who created Python?",
        "What is ChromaDB used for?",
        "What LLM providers does LangChain support?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = chain.invoke(q)
        print(f"A: {answer}")

    # Cleanup
    vectorstore.delete_collection()
    print("\nDone!")


if __name__ == "__main__":
    main()
