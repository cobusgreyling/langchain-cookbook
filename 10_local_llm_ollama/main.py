"""
Local LLM with Ollama
======================
Run LangChain entirely locally — no API keys required.
Requires Ollama installed with a model pulled (e.g. llama3.2).

1. Basic chat with local LLM
2. Chain with local LLM
3. Local RAG with local embeddings
"""

import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except ImportError:
    print("Error: Install langchain-ollama: pip install langchain-ollama")
    sys.exit(1)

MODEL_NAME = "llama3.2"


def check_ollama():
    """Verify Ollama is running and the model is available."""
    try:
        llm = ChatOllama(model=MODEL_NAME)
        llm.invoke("hi")
        return True
    except Exception as e:
        print(f"Error: Could not connect to Ollama. Make sure it's running")
        print(f"  Install: https://ollama.com")
        print(f"  Then run: ollama pull {MODEL_NAME}")
        print(f"  Detail: {e}")
        return False


def example_1_basic_chat():
    """Basic question answering with a local model."""
    print("=== Example 1: Basic Local Chat ===")
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    response = llm.invoke("What are three benefits of open-source software? Be brief.")
    print(f"  Response: {response.content[:200]}\n")


def example_2_chain():
    """Run a chain with a local model."""
    print("=== Example 2: Local Chain ===")
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    prompt = ChatPromptTemplate.from_template(
        "You are a {role}. Answer this question in 2-3 sentences:\n{question}"
    )
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "role": "friendly science teacher",
        "question": "Why do leaves change color in autumn?",
    })
    print(f"  Response: {result[:200]}\n")


def example_3_local_rag():
    """Full RAG pipeline using only local models."""
    print("=== Example 3: Local RAG (no API keys!) ===")

    # Documents
    docs = [
        Document(page_content=(
            "Ollama makes it easy to run large language models locally. "
            "It supports models like Llama, Mistral, and Gemma. "
            "Models run on your own hardware with no data sent to the cloud."
        )),
        Document(page_content=(
            "Local LLMs offer privacy advantages since data never leaves your machine. "
            "They work offline and have predictable costs. The trade-off is that they "
            "require sufficient RAM and GPU resources."
        )),
        Document(page_content=(
            "Popular local models include Llama 3.2 (Meta), Mistral (Mistral AI), "
            "and Gemma (Google). These models range from 1B to 70B+ parameters. "
            "Smaller models like 7B-8B run well on consumer hardware."
        )),
    ]

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    # Use local embeddings + Chroma
    try:
        from langchain_chroma import Chroma
    except ImportError:
        print("  Skipping RAG (install chromadb: pip install langchain-chroma chromadb)")
        return

    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # RAG chain
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Answer based on this context only:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = "What are the benefits of running LLMs locally?"
    print(f"  Q: {question}")
    answer = chain.invoke(question)
    print(f"  A: {answer[:200]}")

    vectorstore.delete_collection()
    print()


def main():
    if not check_ollama():
        sys.exit(1)

    example_1_basic_chat()
    example_2_chain()
    example_3_local_rag()
    print("Done!")


if __name__ == "__main__":
    main()
