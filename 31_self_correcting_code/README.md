# 31 — Self-Correcting Code Agent

An agent that writes code, executes it, reads errors, and fixes itself in a loop.

## What You'll Learn

- Generate code from natural language with an LLM
- Execute code safely in a subprocess sandbox
- Build a generate → execute → fix loop with LangGraph
- Handle timeouts and maximum retry limits

## Run

```bash
export OPENAI_API_KEY="your-key"
python main.py
```
