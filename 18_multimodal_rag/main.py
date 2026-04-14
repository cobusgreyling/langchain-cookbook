"""
PDF and Multimodal RAG
=======================
Load and query PDF documents:
1. Extract text from PDFs using pypdf
2. Chunk and embed the content
3. Query with a RAG chain
4. Demonstrate multimodal prompting (image URL + text)
"""

import os
import sys
import tempfile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Part 1: PDF RAG (simulated) ----------

# Simulated PDF content (avoids needing a real PDF file for the demo)
SIMULATED_PDF_PAGES = [
    {
        "page": 1,
        "content": (
            "Annual Report 2024 — Acme Corp\n\n"
            "Executive Summary\n"
            "Acme Corp achieved record revenue of $4.2 billion in 2024, representing "
            "a 15% year-over-year increase. Our AI division grew by 45%, driven by "
            "strong enterprise adoption of our platform products."
        ),
    },
    {
        "page": 2,
        "content": (
            "Financial Highlights\n\n"
            "Revenue: $4.2B (up 15% YoY)\n"
            "Net Income: $890M (up 22% YoY)\n"
            "R&D Spending: $1.1B (26% of revenue)\n"
            "Employee Count: 12,500 (up from 10,800)\n"
            "The company maintained strong margins despite increased investment in "
            "research and development, particularly in generative AI and cloud infrastructure."
        ),
    },
    {
        "page": 3,
        "content": (
            "Product Updates\n\n"
            "1. Acme AI Platform v3.0 launched with multi-modal capabilities\n"
            "2. Cloud infrastructure expanded to 8 new regions\n"
            "3. Enterprise customer base grew to 2,400 companies\n"
            "4. Developer API usage increased 3x year-over-year\n"
            "Our focus remains on making AI accessible and safe for enterprise use cases."
        ),
    },
]


def load_pdf_documents() -> list[Document]:
    """Simulate loading pages from a PDF document."""
    docs = []
    for page in SIMULATED_PDF_PAGES:
        docs.append(Document(
            page_content=page["content"],
            metadata={"source": "annual_report_2024.pdf", "page": page["page"]},
        ))
    return docs


def pdf_rag_demo():
    """Demonstrate RAG over PDF documents."""
    print("=== Part 1: PDF RAG ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load and split
    docs = load_pdf_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"  Loaded {len(docs)} pages, split into {len(chunks)} chunks")

    # Build vector store
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # RAG chain
    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context from the PDF.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs
        )

    questions = [
        "What was Acme Corp's revenue in 2024?",
        "How much did the company spend on R&D?",
        "What new product capabilities were launched?",
    ]

    for q in questions:
        retrieved = retriever.invoke(q)
        context = format_docs(retrieved)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": q})
        pages = [str(d.metadata.get("page", "?")) for d in retrieved]
        print(f"  Q: {q}")
        print(f"  A: {answer}")
        print(f"  Sources: pages {', '.join(pages)}\n")

    vectorstore.delete_collection()


# ---------- Part 2: Multimodal (Vision) ----------

def multimodal_demo():
    """Demonstrate multimodal prompting with image URLs."""
    print("=== Part 2: Multimodal Vision ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Vision with a public image URL
    message = HumanMessage(content=[
        {"type": "text", "text": "Describe this image in one sentence. What does it show?"},
        {
            "type": "image_url",
            "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/200px-Python-logo-notext.svg.png"},
        },
    ])

    print("  Sending image + text prompt to GPT-4o-mini...")
    response = llm.invoke([message])
    print(f"  Response: {response.content}\n")


def main():
    check_api_key()
    pdf_rag_demo()
    multimodal_demo()
    print("Done!")


if __name__ == "__main__":
    main()
