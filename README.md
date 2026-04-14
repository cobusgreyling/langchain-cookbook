# LangChain Cookbook

A collection of practical, self-contained recipes for building LLM applications with [LangChain](https://github.com/langchain-ai/langchain).

Each recipe focuses on one concept, includes working code, and can be run independently.

**Author:** [Cobus Greyling](https://github.com/cobusgreyling)

## Recipes

### Fundamentals

| # | Recipe | Description |
|---|--------|-------------|
| 01 | [Basic RAG with ChromaDB](01_basic_rag/) | Retrieval-Augmented Generation pipeline |
| 02 | [Conversational Agent](02_conversational_agent/) | Agent with tools and multi-turn memory |
| 03 | [Structured Output](03_structured_output/) | Extract typed data with Pydantic models |
| 04 | [Multi-Chain Routing](04_multi_chain_routing/) | Route queries to specialised chains |
| 05 | [Streaming](05_streaming/) | Stream tokens, chains, and events |
| 06 | [Custom Tools](06_custom_tools/) | Build custom tools for agents |
| 07 | [Memory Patterns](07_memory_patterns/) | Buffer, window, and summary memory |
| 08 | [Document Loaders](08_document_loaders/) | Load text, CSV, and web documents |
| 09 | [Evaluation](09_evaluation/) | Test and score your chains |
| 10 | [Local LLM (Ollama)](10_local_llm_ollama/) | Run everything locally — no API keys |

### Intermediate

| # | Recipe | Description |
|---|--------|-------------|
| 11 | [LangGraph Agent](11_langgraph_agent/) | Stateful graph-based agent with checkpointing |
| 12 | [Multi-Agent Collaboration](12_multi_agent/) | Supervisor + worker agent architecture |
| 13 | [Modern Tool Calling](13_tool_calling/) | Native function/tool calling with LLMs |
| 14 | [RAG with Reranking](14_rag_reranking/) | LLM-based reranking for better retrieval |
| 15 | [Hybrid Search](15_hybrid_search/) | Vector + keyword search with Reciprocal Rank Fusion |
| 16 | [Callbacks & Observability](16_callbacks_observability/) | Custom logging, token counting, cost tracking |
| 17 | [Output Guardrails](17_output_guardrails/) | PII detection, moderation, output validation |
| 18 | [PDF & Multimodal RAG](18_multimodal_rag/) | RAG over PDFs and vision prompting |
| 19 | [Caching & Optimization](19_caching_optimization/) | Response caching, token budgeting, prompt compression |
| 20 | [Deployment (FastAPI)](20_deployment_langserve/) | Serve chains as REST API endpoints |

### Advanced Patterns

| # | Recipe | Description |
|---|--------|-------------|
| 21 | [Persistent Memory](21_persistent_memory/) | SQLite-backed chat history across sessions |
| 22 | [Few-Shot Prompting](22_few_shot_prompting/) | Static and dynamic example selection |
| 23 | [LCEL Chains](23_lcel_chains/) | Pipes, parallel, branching, and lambda runnables |
| 24 | [Error Handling & Fallbacks](24_error_handling_fallbacks/) | Retries, model fallbacks, graceful degradation |
| 25 | [Async Execution](25_async_execution/) | Concurrent chains for better throughput |

### Agentic & Cutting-Edge

| # | Recipe | Description |
|---|--------|-------------|
| 26 | [Graph RAG](26_graph_rag/) | Knowledge graph + vector retrieval for richer context |
| 27 | [Agentic RAG](27_agentic_rag/) | Self-correcting retrieval with query rewriting and grading |
| 28 | [Multi-Modal Agents](28_multimodal_agents/) | Vision-capable agents that reason over images and call tools |
| 29 | [Human-in-the-Loop](29_human_in_the_loop/) | Agent pauses for human approval on sensitive actions |
| 30 | [Long-Term Memory](30_long_term_memory/) | Semantic memory that persists across conversations |
| 31 | [Self-Correcting Code](31_self_correcting_code/) | Agent writes, executes, and fixes code in a loop |
| 32 | [RAG Evaluation](32_rag_evaluation/) | RAGAS-style faithfulness, relevance, and correctness metrics |
| 33 | [Text-to-SQL](33_text_to_sql/) | Natural language → SQL → results → natural language |
| 34 | [Web Research Agent](34_web_research_agent/) | Search, read, and synthesise web information |
| 35 | [MCP Tool Server](35_mcp_tool_server/) | Connect agents to tools via Model Context Protocol |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/cobusgreyling/langchain-cookbook.git
cd langchain-cookbook

# Install dependencies
pip install -r requirements.txt
# Or with uv:
uv pip install -r requirements.txt

# Set your API key (recipes 01-09, 11-35)
export OPENAI_API_KEY="your-key"

# Optional: for Anthropic as an alternative LLM provider
export ANTHROPIC_API_KEY="your-key"

# Run any recipe
cd 01_basic_rag
python main.py
```

## Development

```bash
# Install dependencies
make install
# Or: make uv-install

# Check all recipes compile
make lint

# Verify structure (main.py + README.md in each folder)
make check

# Run a specific recipe
make run-01_basic_rag
```

## Requirements

- Python 3.10+
- OpenAI API key (for recipes 01-09, 11-35)
- [Ollama](https://ollama.com) (for recipe 10 only)
- Anthropic API key (optional — for using Claude as an alternative LLM)

## Using Anthropic/Claude Instead of OpenAI

Any recipe can be adapted to use Claude. Replace the LLM initialization:

```python
# Before (OpenAI)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# After (Anthropic)
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new recipes.

## License

MIT — [Cobus Greyling](https://github.com/cobusgreyling)
