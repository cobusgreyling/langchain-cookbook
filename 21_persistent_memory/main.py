"""
Persistent Chat Memory
=======================
Store conversation history in SQLite for persistence across sessions:
1. SQLite-backed message storage
2. Session management (multiple conversations)
3. Retrieval of past conversations
"""

import os
import sys
import sqlite3
import json
import tempfile
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- SQLite message store ----------

class SQLiteMessageStore:
    """Persistent message store using SQLite."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def add_message(self, session_id: str, role: str, content: str):
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now().isoformat()),
        )
        self.conn.commit()

    def get_messages(self, session_id: str, limit: int = 50) -> list:
        cursor = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cursor.fetchall()
        rows.reverse()  # Oldest first
        messages = []
        for role, content in rows:
            if role == "human":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        return messages

    def list_sessions(self) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT session_id, COUNT(*) as msg_count, MIN(timestamp) as started "
            "FROM messages GROUP BY session_id ORDER BY started"
        )
        return [{"session_id": r[0], "messages": r[1], "started": r[2]} for r in cursor.fetchall()]

    def close(self):
        self.conn.close()


# ---------- Chat with persistent memory ----------

def chat(store: SQLiteMessageStore, session_id: str, user_input: str, llm: ChatOpenAI) -> str:
    """Send a message and get a response, persisting to SQLite."""
    # Load history
    history = store.get_messages(session_id)

    # Build chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()

    # Invoke
    response = chain.invoke({"input": user_input, "history": history})

    # Persist
    store.add_message(session_id, "human", user_input)
    store.add_message(session_id, "ai", response)

    return response


def main():
    check_api_key()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    db_path = os.path.join(tempfile.gettempdir(), "langchain_memory_demo.db")
    store = SQLiteMessageStore(db_path)

    print("=== Persistent Chat Memory (SQLite) ===\n")
    print(f"  Database: {db_path}\n")

    # Session 1
    print("--- Session: alice-001 ---")
    session1 = "alice-001"
    for msg in [
        "Hi, I'm Alice. I work on machine learning.",
        "My current project is a recommendation engine for e-commerce.",
        "What do you remember about me?",
    ]:
        print(f"  User: {msg}")
        response = chat(store, session1, msg, llm)
        print(f"  AI: {response}\n")

    # Session 2 (different user)
    print("--- Session: bob-001 ---")
    session2 = "bob-001"
    for msg in [
        "Hey, I'm Bob. I'm learning Python.",
        "Can you suggest a project for beginners?",
    ]:
        print(f"  User: {msg}")
        response = chat(store, session2, msg, llm)
        print(f"  AI: {response}\n")

    # Show all sessions
    print("--- All Sessions ---")
    for session in store.list_sessions():
        print(f"  {session['session_id']}: {session['messages']} messages (started {session['started'][:19]})")

    # Resume session 1 (proves persistence)
    print(f"\n--- Resuming Session: alice-001 ---")
    response = chat(store, session1, "Remind me what project I told you about?", llm)
    print(f"  User: Remind me what project I told you about?")
    print(f"  AI: {response}\n")

    store.close()
    # Clean up temp DB
    os.unlink(db_path)

    print("Done!")


if __name__ == "__main__":
    main()
