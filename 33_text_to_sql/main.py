"""
Text-to-SQL Agent
==================
Natural language → SQL → results → natural language answer:
1. Parse the user's question
2. Generate a SQL query
3. Execute against an in-memory SQLite database
4. Summarise the results in natural language
"""

import os
import sys
import sqlite3
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Database setup ----------

def create_database() -> sqlite3.Connection:
    """Create an in-memory database with sample data."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary REAL NOT NULL,
            hire_date TEXT NOT NULL
        );

        CREATE TABLE departments (
            name TEXT PRIMARY KEY,
            manager TEXT NOT NULL,
            budget REAL NOT NULL
        );

        INSERT INTO departments VALUES ('Engineering', 'Alice Chen', 500000);
        INSERT INTO departments VALUES ('Marketing', 'Bob Smith', 300000);
        INSERT INTO departments VALUES ('Sales', 'Carol Davis', 400000);
        INSERT INTO departments VALUES ('HR', 'Dave Wilson', 200000);

        INSERT INTO employees VALUES (1, 'Alice Chen', 'Engineering', 150000, '2020-01-15');
        INSERT INTO employees VALUES (2, 'Bob Smith', 'Marketing', 120000, '2019-06-01');
        INSERT INTO employees VALUES (3, 'Carol Davis', 'Sales', 130000, '2018-03-20');
        INSERT INTO employees VALUES (4, 'Dave Wilson', 'HR', 110000, '2021-09-10');
        INSERT INTO employees VALUES (5, 'Eve Brown', 'Engineering', 140000, '2020-07-22');
        INSERT INTO employees VALUES (6, 'Frank Lee', 'Engineering', 135000, '2022-01-05');
        INSERT INTO employees VALUES (7, 'Grace Kim', 'Marketing', 105000, '2023-02-14');
        INSERT INTO employees VALUES (8, 'Hank Moore', 'Sales', 115000, '2021-11-30');
        INSERT INTO employees VALUES (9, 'Iris Wang', 'Sales', 125000, '2020-04-18');
        INSERT INTO employees VALUES (10, 'Jack Taylor', 'HR', 95000, '2023-06-01');
    """)

    conn.commit()
    return conn


# Global database connection
db_conn: sqlite3.Connection = None

DATABASE_SCHEMA = """
Tables:
  employees (id INTEGER PK, name TEXT, department TEXT, salary REAL, hire_date TEXT)
  departments (name TEXT PK, manager TEXT, budget REAL)
"""


# ---------- Tools ----------

@tool
def get_schema() -> str:
    """Get the database schema to understand available tables and columns."""
    return DATABASE_SCHEMA


@tool
def run_sql(query: str) -> str:
    """Execute a read-only SQL query and return results. Only SELECT queries are allowed."""
    query = query.strip()
    if not query.upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for safety."

    try:
        cursor = db_conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        if not rows:
            return "Query returned no results."

        # Format as a simple table
        header = " | ".join(columns)
        separator = "-" * len(header)
        data_rows = [" | ".join(str(v) for v in row) for row in rows]

        return f"{header}\n{separator}\n" + "\n".join(data_rows)

    except sqlite3.Error as e:
        return f"SQL Error: {e}"


# ---------- State ----------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------- Nodes ----------

TOOLS = [get_schema, run_sql]


def agent_node(state: AgentState):
    """Text-to-SQL agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    system = SystemMessage(content=(
        "You are a database assistant. Convert natural language questions into SQL queries.\n"
        "Steps:\n"
        "1. First call get_schema() to understand the database structure\n"
        "2. Write a SELECT query and call run_sql() to execute it\n"
        "3. Summarise the results in natural language\n\n"
        "Rules:\n"
        "- Only use SELECT queries (read-only)\n"
        "- Be precise with column and table names\n"
        "- If the query fails, read the error and try a corrected query"
    ))

    messages = [system] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_use_tools(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ---------- Build graph ----------

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_use_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def main():
    global db_conn
    check_api_key()

    print("Setting up database...")
    db_conn = create_database()

    agent = build_agent()

    questions = [
        "How many employees are in each department?",
        "Who are the top 3 highest-paid employees?",
        "What is the average salary in the Engineering department?",
        "Which departments are over budget if you sum up all employee salaries?",
    ]

    print("=== Text-to-SQL Agent ===\n")

    for q in questions:
        print(f"Q: {q}")
        result = agent.invoke({"messages": [HumanMessage(content=q)]})
        ai_msg = result["messages"][-1]
        print(f"A: {ai_msg.content}\n")
        print("-" * 60 + "\n")

    db_conn.close()
    print("Done!")


if __name__ == "__main__":
    main()
