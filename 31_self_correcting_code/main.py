"""
Self-Correcting Code Agent
===========================
An agent that writes code, executes it, reads errors, and fixes itself:
1. Generate code from a natural language description
2. Execute the code in a subprocess
3. If errors occur, analyze and fix them
4. Loop until the code runs successfully or max attempts reached
"""

import os
import sys
import subprocess
import tempfile
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- State ----------

class CodeAgentState(TypedDict):
    task: str
    code: str
    output: str
    error: str
    attempts: int
    success: bool


# ---------- Nodes ----------

def generate_code(state: CodeAgentState) -> dict:
    """Generate Python code from the task description."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if state.get("error"):
        # Fix mode: include the error
        prompt = ChatPromptTemplate.from_template(
            "Fix this Python code based on the error.\n\n"
            "Task: {task}\n\n"
            "Previous code:\n```python\n{code}\n```\n\n"
            "Error:\n{error}\n\n"
            "Return ONLY the corrected Python code, no markdown fences or explanation."
        )
        chain = prompt | llm | StrOutputParser()
        code = chain.invoke({
            "task": state["task"],
            "code": state["code"],
            "error": state["error"],
        })
    else:
        # Initial generation
        prompt = ChatPromptTemplate.from_template(
            "Write Python code to accomplish this task. The code should be self-contained, "
            "use only the standard library, and print its results.\n\n"
            "Task: {task}\n\n"
            "Return ONLY Python code, no markdown fences or explanation."
        )
        chain = prompt | llm | StrOutputParser()
        code = chain.invoke({"task": state["task"]})

    # Clean up markdown fences if present
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()

    attempts = state.get("attempts", 0) + 1
    print(f"  [Generate] Attempt {attempts}")
    print(f"  Code:\n{_indent(code)}")

    return {"code": code, "attempts": attempts}


def execute_code(state: CodeAgentState) -> dict:
    """Execute the generated code in a sandboxed subprocess."""
    code = state["code"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print(f"  [Execute] Success!")
            print(f"  Output:\n{_indent(result.stdout)}")
            return {"output": result.stdout, "error": "", "success": True}
        else:
            error = result.stderr.strip()
            print(f"  [Execute] Error:\n{_indent(error)}")
            return {"output": "", "error": error, "success": False}

    except subprocess.TimeoutExpired:
        error = "Execution timed out (10s limit)"
        print(f"  [Execute] {error}")
        return {"output": "", "error": error, "success": False}
    finally:
        os.unlink(temp_path)


def route_after_execution(state: CodeAgentState) -> str:
    """Decide whether to fix or finish."""
    if state["success"]:
        return END
    if state["attempts"] >= 3:
        return END
    return "generate"


# ---------- Helpers ----------

def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.strip().split("\n"))


# ---------- Build graph ----------

def build_agent():
    graph = StateGraph(CodeAgentState)

    graph.add_node("generate", generate_code)
    graph.add_node("execute", execute_code)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "execute")
    graph.add_conditional_edges("execute", route_after_execution, {
        "generate": "generate",
        END: END,
    })

    return graph.compile()


def main():
    check_api_key()

    agent = build_agent()

    tasks = [
        "Write a function that checks if a number is prime, then test it with the numbers 2, 17, 25, and 97.",
        "Create a function that converts a Roman numeral string to an integer. Test with 'XIV', 'MCMXCIV', and 'IX'.",
        "Write a function that finds the longest palindromic substring in a given string. Test with 'babad' and 'racecar'.",
    ]

    print("=== Self-Correcting Code Agent ===\n")

    for task in tasks:
        print(f"Task: {task}\n")
        result = agent.invoke({
            "task": task,
            "code": "",
            "output": "",
            "error": "",
            "attempts": 0,
            "success": False,
        })

        if result["success"]:
            print(f"  ✓ Completed in {result['attempts']} attempt(s)\n")
        else:
            print(f"  ✗ Failed after {result['attempts']} attempts\n")

        print("=" * 60 + "\n")

    print("Done!")


if __name__ == "__main__":
    main()
