"""
Deployment with FastAPI
========================
Serve a LangChain chain as a REST API:
1. Wrap a chain in a FastAPI app
2. Add health check and metadata endpoints
3. Demonstrate sync and async invocation
"""

import os
import sys
import json
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# Note: To run as a real server, install: pip install fastapi uvicorn
# This demo runs the chain directly and shows how to structure the API.


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Chain definitions ----------

def build_qa_chain():
    """Build a simple Q&A chain."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Answer this question concisely:\n{question}"
    )
    return prompt | llm | StrOutputParser()


def build_summarize_chain():
    """Build a text summarization chain."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Summarise this text in one sentence:\n\n{text}"
    )
    return prompt | llm | StrOutputParser()


# ---------- FastAPI app definition ----------

def create_app_code() -> str:
    """Return the FastAPI app code that would be used in production."""
    return '''
# server.py — run with: uvicorn server:app --reload
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="LangChain API", version="1.0.0")

# Build chains
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = (
    ChatPromptTemplate.from_template("Answer concisely: {question}")
    | llm | StrOutputParser()
)

summarize_chain = (
    ChatPromptTemplate.from_template("Summarise in one sentence: {text}")
    | llm | StrOutputParser()
)


class QARequest(BaseModel):
    question: str

class SummarizeRequest(BaseModel):
    text: str

class ChainResponse(BaseModel):
    result: str


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/qa", response_model=ChainResponse)
async def qa(request: QARequest):
    result = await qa_chain.ainvoke({"question": request.question})
    return ChainResponse(result=result)

@app.post("/summarize", response_model=ChainResponse)
async def summarize(request: SummarizeRequest):
    result = await summarize_chain.ainvoke({"text": request.text})
    return ChainResponse(result=result)
'''


# ---------- Demo: simulate API calls ----------

class QARequest(BaseModel):
    question: str


class SummarizeRequest(BaseModel):
    text: str


async def simulate_api():
    """Simulate calling the API endpoints."""
    qa_chain = build_qa_chain()
    summarize_chain = build_summarize_chain()

    # Simulate /qa endpoint
    print("  POST /qa")
    qa_requests = [
        QARequest(question="What is LangChain?"),
        QARequest(question="What is FastAPI used for?"),
    ]

    for req in qa_requests:
        result = await qa_chain.ainvoke({"question": req.question})
        response = {"result": result}
        print(f"    Request: {json.dumps(req.model_dump())}")
        print(f"    Response: {json.dumps(response)}\n")

    # Simulate /summarize endpoint
    print("  POST /summarize")
    text = (
        "FastAPI is a modern, fast, web framework for building APIs with Python 3.7+ "
        "based on standard Python type hints. It is one of the fastest Python frameworks "
        "available, on par with NodeJS and Go. It provides automatic interactive API "
        "documentation and is built on Starlette and Pydantic."
    )
    result = await summarize_chain.ainvoke({"text": text})
    print(f"    Request: {{text: '{text[:50]}...'}}")
    print(f"    Response: {json.dumps({'result': result})}\n")


def main():
    check_api_key()

    print("=== Deployment with FastAPI ===\n")

    # Show the server code
    print("--- Server Code (server.py) ---")
    print(create_app_code())
    print("--- End Server Code ---\n")

    # Simulate API calls
    print("=== Simulated API Calls ===\n")
    asyncio.run(simulate_api())

    print("To run as a real server:")
    print("  pip install fastapi uvicorn")
    print("  uvicorn server:app --reload")
    print("  Then visit http://localhost:8000/docs for interactive API docs\n")
    print("Done!")


if __name__ == "__main__":
    main()
