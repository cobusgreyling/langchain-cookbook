"""
Custom Tool Creation
=====================
Shows how to create custom tools for LangChain agents:
1. Simple @tool decorator
2. Multi-argument tools with Pydantic schemas
3. Async tools
"""

import os
import sys
import json
from datetime import datetime, timedelta

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Tool 1: Simple tool ----------

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------- Tool 2: Tool with complex input ----------

class WeatherInput(BaseModel):
    """Input for the weather tool."""
    city: str = Field(description="City name")
    units: str = Field(default="celsius", description="Temperature units: 'celsius' or 'fahrenheit'")


@tool(args_schema=WeatherInput)
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city. Returns temperature and conditions."""
    # Simulated weather data
    weather_data = {
        "london": {"temp_c": 12, "condition": "Cloudy"},
        "new york": {"temp_c": 22, "condition": "Sunny"},
        "tokyo": {"temp_c": 28, "condition": "Humid"},
        "sydney": {"temp_c": 18, "condition": "Partly cloudy"},
        "paris": {"temp_c": 15, "condition": "Rainy"},
    }

    city_lower = city.lower()
    data = weather_data.get(city_lower, {"temp_c": 20, "condition": "Unknown"})
    temp = data["temp_c"]

    if units == "fahrenheit":
        temp = round(temp * 9 / 5 + 32)
        unit_str = "°F"
    else:
        unit_str = "°C"

    return f"{city}: {temp}{unit_str}, {data['condition']}"


# ---------- Tool 3: Data lookup tool ----------

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another.
    Supported currencies: USD, EUR, GBP, JPY, AUD.
    """
    # Simulated exchange rates (relative to USD)
    rates = {
        "USD": 1.0,
        "EUR": 0.92,
        "GBP": 0.79,
        "JPY": 149.50,
        "AUD": 1.53,
    }

    from_curr = from_currency.upper()
    to_curr = to_currency.upper()

    if from_curr not in rates or to_curr not in rates:
        return f"Unsupported currency. Supported: {', '.join(rates.keys())}"

    # Convert to USD first, then to target
    usd_amount = amount / rates[from_curr]
    result = usd_amount * rates[to_curr]

    return f"{amount} {from_curr} = {result:.2f} {to_curr}"


# ---------- Tool 4: Date calculation ----------

@tool
def date_calculator(operation: str, days: int) -> str:
    """Calculate dates relative to today.
    operation: 'add' or 'subtract'
    days: number of days to add or subtract
    """
    today = datetime.now()
    if operation == "subtract":
        result = today - timedelta(days=days)
    else:
        result = today + timedelta(days=days)
    return f"Result: {result.strftime('%A, %B %d, %Y')}"


def main():
    check_api_key()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_current_time, get_weather, convert_currency, date_calculator]
    agent = create_react_agent(llm, tools)

    queries = [
        "What time is it right now?",
        "What's the weather like in Tokyo and London?",
        "Convert 100 USD to EUR and JPY",
        "What date will it be 45 days from now?",
    ]

    print("=== Custom Tools Agent ===\n")

    for q in queries:
        print(f"Q: {q}")
        response = agent.invoke({"messages": [HumanMessage(content=q)]})
        answer = response["messages"][-1].content
        print(f"A: {answer}\n")

    print("Done!")


if __name__ == "__main__":
    main()
