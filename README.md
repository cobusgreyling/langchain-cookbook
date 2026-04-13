# LangChain Cookbook

A collection of practical, self-contained recipes for building LLM applications with [LangChain](https://github.com/langchain-ai/langchain).

Each recipe focuses on one concept, includes working code, and can be run independently.

## Recipes

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

## Quick Start

```bash
# Clone the repo
git clone https://github.com/cobusgreyling/langchain-cookbook.git
cd langchain-cookbook

# Install dependencies
pip install -r requirements.txt

# Set your API key (recipes 01-09)
export OPENAI_API_KEY="your-key"

# Run any recipe
cd 01_basic_rag
python main.py
```

## Requirements

- Python 3.10+
- OpenAI API key (for recipes 01-09)
- [Ollama](https://ollama.com) (for recipe 10 only)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new recipes.

## License

MIT
