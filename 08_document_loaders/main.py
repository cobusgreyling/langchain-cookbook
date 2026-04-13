"""
Document Loaders
=================
Demonstrates loading documents from multiple sources:
1. Plain text (in-memory)
2. CSV data
3. Web pages
4. Combined pipeline with splitting
"""

import os
import sys
import csv
import tempfile

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    WebBaseLoader,
)


# ---------- Example 1: Text files ----------

def example_text_loader():
    """Load a plain text file."""
    print("=== Example 1: Text Loader ===")

    # Create a temp file to demonstrate
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            "LangChain is a framework for building LLM applications.\n"
            "It provides abstractions for chains, agents, and retrieval.\n"
            "LangChain supports multiple language model providers.\n"
            "The framework is available in Python and JavaScript."
        )
        tmp_path = f.name

    loader = TextLoader(tmp_path)
    docs = loader.load()

    print(f"  Loaded {len(docs)} document(s)")
    print(f"  Content preview: {docs[0].page_content[:80]}...")
    print(f"  Metadata: {docs[0].metadata}")

    os.unlink(tmp_path)
    return docs


# ---------- Example 2: CSV files ----------

def example_csv_loader():
    """Load structured data from a CSV file."""
    print("\n=== Example 2: CSV Loader ===")

    # Create a sample CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "language", "stars", "description"])
        writer.writerow(["LangChain", "Python", "75000", "Framework for LLM apps"])
        writer.writerow(["LlamaIndex", "Python", "30000", "Data framework for LLMs"])
        writer.writerow(["Semantic Kernel", "C#", "18000", "Microsoft LLM orchestration"])
        writer.writerow(["Haystack", "Python", "14000", "NLP framework by deepset"])
        tmp_path = f.name

    loader = CSVLoader(tmp_path)
    docs = loader.load()

    print(f"  Loaded {len(docs)} rows as documents")
    for doc in docs[:2]:
        print(f"  Row: {doc.page_content[:60]}...")

    os.unlink(tmp_path)
    return docs


# ---------- Example 3: Web pages ----------

def example_web_loader():
    """Load content from a web page."""
    print("\n=== Example 3: Web Loader ===")

    try:
        loader = WebBaseLoader("https://en.wikipedia.org/wiki/Large_language_model")
        docs = loader.load()

        print(f"  Loaded {len(docs)} document(s) from web")
        print(f"  Content length: {len(docs[0].page_content)} characters")
        print(f"  Preview: {docs[0].page_content[100:200].strip()}...")
        return docs
    except Exception as e:
        print(f"  Web loading skipped (requires internet): {e}")
        # Return a fallback document
        return [Document(
            page_content="Large language models (LLMs) are AI models trained on vast text data.",
            metadata={"source": "fallback"},
        )]


# ---------- Example 4: Combined pipeline ----------

def example_combined_pipeline(all_docs):
    """Combine documents from multiple sources and split them."""
    print("\n=== Example 4: Combined Pipeline ===")

    print(f"  Total documents before splitting: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        length_function=len,
    )
    chunks = splitter.split_documents(all_docs)

    print(f"  Total chunks after splitting: {len(chunks)}")
    print(f"  Sources: {set(doc.metadata.get('source', 'unknown') for doc in chunks)}")

    # Show chunk size distribution
    sizes = [len(c.page_content) for c in chunks]
    print(f"  Chunk sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")

    return chunks


def main():
    text_docs = example_text_loader()
    csv_docs = example_csv_loader()
    web_docs = example_web_loader()

    all_docs = text_docs + csv_docs + web_docs
    chunks = example_combined_pipeline(all_docs)

    print("\nDone!")


if __name__ == "__main__":
    main()
