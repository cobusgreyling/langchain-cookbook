# 14 — RAG with Reranking

Improve RAG quality by adding an LLM-based reranking step between retrieval and generation.

## What You'll Learn

- Retrieve a broad set of candidates from a vector store
- Use an LLM as a reranker to score document relevance
- Answer questions using only the highest-ranked documents

## Run

```bash
export OPENAI_API_KEY="your-key"
python main.py
```
