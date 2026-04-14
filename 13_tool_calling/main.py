"""
Modern Tool Calling
====================
Demonstrates native tool/function calling with OpenAI and Anthropic models:
1. Bind tools to a model
2. Parse tool call results
3. Multi-tool invocation in a single turn
"""

import os
import sys
import json

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Tools ----------

class StockQuery(BaseModel):
    """Input for stock price lookup."""
    symbol: str = Field(description="Stock ticker symbol, e.g. AAPL")


@tool(args_schema=StockQuery)
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a ticker symbol."""
    prices = {
        "AAPL": 178.52, "GOOGL": 141.80, "MSFT": 378.91,
        "AMZN": 178.25, "TSLA": 248.42, "NVDA": 495.22,
    }
    symbol = symbol.upper()
    if symbol in prices:
        return json.dumps({"symbol": symbol, "price": prices[symbol], "currency": "USD"})
    return json.dumps({"error": f"Unknown symbol: {symbol}"})


@tool
def get_company_info(name: str) -> str:
    """Get basic information about a company."""
    companies = {
        "apple": {"name": "Apple Inc.", "sector": "Technology", "employees": "164,000", "founded": 1976},
        "google": {"name": "Alphabet Inc.", "sector": "Technology", "employees": "182,000", "founded": 1998},
        "tesla": {"name": "Tesla Inc.", "sector": "Automotive/Energy", "employees": "128,000", "founded": 2003},
    }
    info = companies.get(name.lower())
    if info:
        return json.dumps(info)
    return json.dumps({"error": f"No info for '{name}'"})


@tool
def calculate_portfolio_value(holdings: str) -> str:
    """Calculate total portfolio value. Holdings format: 'AAPL:10,GOOGL:5' (symbol:shares)."""
    prices = {"AAPL": 178.52, "GOOGL": 141.80, "MSFT": 378.91, "AMZN": 178.25, "TSLA": 248.42, "NVDA": 495.22}
    total = 0.0
    breakdown = []
    for item in holdings.split(","):
        sym, shares = item.strip().split(":")
        sym = sym.upper()
        shares = int(shares)
        price = prices.get(sym, 0)
        value = price * shares
        total += value
        breakdown.append({"symbol": sym, "shares": shares, "value": round(value, 2)})
    return json.dumps({"total": round(total, 2), "breakdown": breakdown})


def main():
    check_api_key()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_stock_price, get_company_info, calculate_portfolio_value]
    llm_with_tools = llm.bind_tools(tools)

    # Map tool names to functions for execution
    tool_map = {t.name: t for t in tools}

    queries = [
        "What's the stock price of AAPL and NVDA?",
        "Tell me about Tesla as a company and its current stock price.",
        "Calculate the value of my portfolio: AAPL:50,MSFT:20,NVDA:10",
    ]

    print("=== Modern Tool Calling ===\n")

    for query in queries:
        print(f"Q: {query}")
        messages = [HumanMessage(content=query)]

        # First LLM call — may produce tool calls
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Execute any tool calls
        if response.tool_calls:
            print(f"  [Tools called: {', '.join(tc['name'] for tc in response.tool_calls)}]")
            for tc in response.tool_calls:
                result = tool_map[tc["name"]].invoke(tc["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

            # Second LLM call — synthesise tool results
            final = llm_with_tools.invoke(messages)
            print(f"A: {final.content}\n")
        else:
            print(f"A: {response.content}\n")

    print("Done!")


if __name__ == "__main__":
    main()
