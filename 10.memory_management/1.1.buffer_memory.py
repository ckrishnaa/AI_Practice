"""
1. BUFFER MEMORY (short-term memory via a checkpointer)

The checkpointer saves the full message history of every thread_id.
Same thread_id  -> the agent sees the whole earlier conversation.
Different id    -> a fresh, empty conversation.
checkpointer    -> choose between in-memory or persistent storage.
"""
import sqlite3

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import (
    InMemorySaver,  # For temporary, in-memory testing
)
from langgraph.checkpoint.sqlite import (
    SqliteSaver,  # For persistent, local database storage
)

load_dotenv()

#llm = ChatGroq(model="llama-3.1-8b-instant") # model retired from Groq
llm = ChatGroq(model="openai/gpt-oss-120b")


def buffer_memory(thread_id: str, checkpointer: str, content: str):
    cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    if checkpointer == 'In-memory':
        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt="You are a helpful assistant.",
            checkpointer=InMemorySaver(),  # lives in RAM, gone when the script exits
        )
    elif checkpointer == 'Persistent':
        conn = sqlite3.connect("memory.db", check_same_thread=False)
        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt="You are a helpful assistant.",
            checkpointer=SqliteSaver(conn),  # persistent storage
        )
    else:
        raise ValueError(f"Unknown checkpointer: {checkpointer}")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        cfg,
    )
    return f"{thread_id}: {result["messages"][-1].content}"

if __name__ == "__main__":
    print(f"In Memory Turn 1 for User1: {buffer_memory('user1', 'In-memory', 'Hi, I\'m Krishna from Chennai.')} \n")
    print(f"In Memory Turn 2 for User1: {buffer_memory('user1', 'In-memory', 'Where am I from?')} \n")
    print(f"In Memory Turn 1 for User2: {buffer_memory('user2', 'In-memory', 'Where am I from?')} \n")
    print(f"Persistent Memory Turn 1 for User1: {buffer_memory('user1', 'Persistent', 'Hi, I\'m Krishna from Chennai.')} \n")
    print(f"Persistent Memory Turn 2 for User1: {buffer_memory('user1', 'Persistent', 'Where am I from?')} \n")
    print(f"Persistent Memory Turn 1 for User2: {buffer_memory('user2', 'Persistent', 'Where am I from?')} \n")
