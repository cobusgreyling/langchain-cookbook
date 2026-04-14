# 35 — MCP Tool Server Integration

Connect LangChain agents to external tools via Model Context Protocol (MCP).

## What You'll Learn

- Create an MCP-compatible tool server
- Convert MCP tools into LangChain StructuredTools
- Combine MCP tools with native LangChain tools in a single agent
- Understand the MCP pattern for tool interoperability

## Run

```bash
export OPENAI_API_KEY="your-key"
python main.py
```

## Production Tips

For real MCP server connections, use `langchain-mcp-adapters`:

```python
# pip install langchain-mcp-adapters
from langchain_mcp_adapters.client import MultiServerMCPClient

async with MultiServerMCPClient(
    {"weather": {"command": "python", "args": ["weather_server.py"]}}
) as client:
    tools = client.get_tools()
```
