"""
MCP Tool Server Integration
=============================
Connect LangChain agents to tools via Model Context Protocol (MCP):
1. Create a simple MCP-compatible tool server
2. Connect to it from a LangChain agent
3. Use MCP tools alongside native LangChain tools

Note: This recipe demonstrates the MCP pattern with a simulated server
for portability. For production use, install langchain-mcp-adapters.
"""

import os
import sys
import json
from typing import Annotated, Any

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Simulated MCP Server ----------
# In production, this would be a separate process communicating via stdio or SSE.

class MCPToolServer:
    """Simulated MCP tool server that exposes tools via a protocol-like interface."""

    def __init__(self, name: str):
        self.name = name
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, Any] = {}

    def register_tool(self, name: str, description: str, parameters: dict, handler):
        """Register a tool with the server."""
        self._tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": parameters,
            },
        }
        self._handlers[name] = handler

    def list_tools(self) -> list[dict]:
        """MCP list_tools — returns available tool definitions."""
        return list(self._tools.values())

    def call_tool(self, name: str, arguments: dict) -> dict:
        """MCP call_tool — execute a tool and return results."""
        if name not in self._handlers:
            return {"error": f"Unknown tool: {name}"}
        try:
            result = self._handlers[name](**arguments)
            return {"content": [{"type": "text", "text": str(result)}]}
        except Exception as e:
            return {"error": str(e)}


# ---------- Set up MCP server with tools ----------

def create_weather_server() -> MCPToolServer:
    """Create a simulated weather MCP server."""
    server = MCPToolServer("weather-server")

    WEATHER_DATA = {
        "new york": {"temp": 72, "condition": "Partly cloudy", "humidity": 65},
        "london": {"temp": 59, "condition": "Rainy", "humidity": 80},
        "tokyo": {"temp": 78, "condition": "Sunny", "humidity": 55},
        "paris": {"temp": 64, "condition": "Overcast", "humidity": 70},
        "sydney": {"temp": 68, "condition": "Clear", "humidity": 50},
    }

    def get_weather(city: str) -> str:
        city_lower = city.lower()
        if city_lower in WEATHER_DATA:
            w = WEATHER_DATA[city_lower]
            return f"Weather in {city}: {w['temp']}°F, {w['condition']}, {w['humidity']}% humidity"
        return f"Weather data not available for {city}."

    def get_forecast(city: str, days: int = 3) -> str:
        city_lower = city.lower()
        if city_lower not in WEATHER_DATA:
            return f"Forecast not available for {city}."
        base = WEATHER_DATA[city_lower]
        lines = [f"Forecast for {city} ({days} days):"]
        for i in range(days):
            temp = base["temp"] + (i * 2 - 1)
            lines.append(f"  Day {i+1}: {temp}°F, {base['condition']}")
        return "\n".join(lines)

    server.register_tool(
        "get_weather",
        "Get current weather for a city",
        {"city": {"type": "string", "description": "City name"}},
        get_weather,
    )

    server.register_tool(
        "get_forecast",
        "Get weather forecast for a city",
        {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer", "description": "Number of days (1-7)"},
        },
        get_forecast,
    )

    return server


# ---------- MCP → LangChain adapter ----------

def mcp_to_langchain_tools(server: MCPToolServer) -> list[StructuredTool]:
    """Convert MCP server tools into LangChain StructuredTools."""
    lc_tools = []

    for tool_def in server.list_tools():
        name = tool_def["name"]
        description = tool_def["description"]

        # Create a closure that captures the server and tool name
        def make_handler(srv, tool_name):
            def handler(**kwargs) -> str:
                result = srv.call_tool(tool_name, kwargs)
                if "error" in result:
                    return f"Error: {result['error']}"
                return result["content"][0]["text"]
            return handler

        lc_tool = StructuredTool.from_function(
            func=make_handler(server, name),
            name=name,
            description=description,
        )
        lc_tools.append(lc_tool)

    return lc_tools


# ---------- Native LangChain tool ----------

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between temperature units (fahrenheit/celsius)."""
    if from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        result = (value - 32) * 5 / 9
        return f"{value}°F = {result:.1f}°C"
    elif from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
        result = value * 9 / 5 + 32
        return f"{value}°C = {result:.1f}°F"
    return f"Unsupported conversion: {from_unit} to {to_unit}"


# ---------- Agent state ----------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------- Build agent ----------

def build_agent(mcp_tools: list, native_tools: list):
    all_tools = mcp_tools + native_tools

    def agent_node(state: AgentState):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        llm_with_tools = llm.bind_tools(all_tools)

        system = SystemMessage(content=(
            "You are a helpful weather assistant. You can check weather and forecasts "
            "using your tools, and convert temperatures between units. "
            "Be helpful and provide clear answers."
        ))

        messages = [system] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_use_tools(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(all_tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_use_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def main():
    check_api_key()

    # Create MCP server and convert tools
    print("Starting MCP weather server...")
    weather_server = create_weather_server()
    mcp_tools = mcp_to_langchain_tools(weather_server)
    print(f"Loaded {len(mcp_tools)} tools from MCP server: {[t.name for t in mcp_tools]}")

    # Combine with native tools
    native_tools = [unit_converter]
    print(f"Native tools: {[t.name for t in native_tools]}\n")

    agent = build_agent(mcp_tools, native_tools)

    questions = [
        "What's the weather like in Tokyo right now?",
        "Give me a 5-day forecast for London.",
        "What's the weather in New York? Also, convert the temperature to Celsius.",
    ]

    print("=== MCP Tool Server Integration ===\n")

    for q in questions:
        print(f"User: {q}")
        result = agent.invoke({"messages": [HumanMessage(content=q)]})
        ai_msg = result["messages"][-1]
        print(f"Agent: {ai_msg.content}\n")
        print("-" * 60 + "\n")

    print("Done!")


if __name__ == "__main__":
    main()
