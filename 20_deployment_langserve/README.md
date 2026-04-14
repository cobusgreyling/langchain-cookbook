# 20 — Deployment with FastAPI

Serve LangChain chains as REST API endpoints with FastAPI.

## What You'll Learn

- Wrap chains in FastAPI endpoints with Pydantic request/response models
- Use async chain invocation for concurrent requests
- Structure a production-ready API with health checks

## Run

```bash
export OPENAI_API_KEY="your-key"

# Demo mode (simulates API calls):
python main.py

# Production server:
pip install fastapi uvicorn
# Copy the generated server.py code, then:
uvicorn server:app --reload
```
