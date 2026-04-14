# 34 — Web Research Agent

An agent that searches the web, reads pages, and synthesises findings into a report.

## What You'll Learn

- Build a research agent with search, read, and note-taking tools
- Implement a multi-step research workflow with LangGraph
- Synthesise findings from multiple sources
- Adapt to real web search APIs (Tavily, SerpAPI) for production use

## Run

```bash
export OPENAI_API_KEY="your-key"
python main.py
```

## Production Tips

Replace the simulated web data with real search APIs:

```python
# pip install tavily-python
from langchain_community.tools.tavily_search import TavilySearchResults
search_tool = TavilySearchResults(max_results=3)
```
