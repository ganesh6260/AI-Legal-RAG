# Place this file at: rag/chat_graph.py
#
# Modern (LangChain 1.x) chat memory for the RAG app, built on LangGraph.
# Each uploaded PDF gets its own `thread_id`, so conversations don't bleed
# into each other, and switching back to a PDF within the same session
# resumes its own multi-turn history.
#
# This reuses your existing llm.llm.generate_answer() (and its 429/503
# error handling) rather than creating a second, separate LLM client —
# LangGraph here is only responsible for storing/trimming/formatting the
# conversation memory per PDF.

from typing import List

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, trim_messages

from rag.retriever import search_documents
from llm.llm import generate_answer

# Keep only the last N messages when building the chat_history text sent to
# the model, so a long conversation doesn't grow the prompt unbounded.
trimmer = trim_messages(
    max_tokens=12,          # counts messages here (token_counter=len)
    strategy="last",
    token_counter=len,
    include_system=False,
    allow_partial=False,
    start_on="human",
)


# -------------------------------
# Graph state
# -------------------------------
class ChatState(MessagesState):
    context: str
    source_chunks: List[str]


# -------------------------------
# Nodes
# -------------------------------
def retrieve_node(state: ChatState):
    question = state["messages"][-1].content
    results = search_documents(question)
    source_chunks = results["documents"][0] if results.get("documents") else []
    context = "\n\n".join(source_chunks)
    return {"context": context, "source_chunks": source_chunks}


def _format_history(messages) -> str:
    """
    Turn prior turns (everything except the current question, which is the
    last message) into the plain-text chat_history string expected by
    rag/prompt_template.py.
    """
    lines = []
    for msg in messages[:-1]:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def generate_node(state: ChatState):
    trimmed_history = trimmer.invoke(state["messages"])
    question = state["messages"][-1].content
    context = state.get("context", "")
    chat_history = _format_history(trimmed_history)

    answer = generate_answer(question, context, chat_history=chat_history)
    return {"messages": [AIMessage(content=answer)]}


# -------------------------------
# Build graph
# -------------------------------
_builder = StateGraph(ChatState)
_builder.add_node("retrieve", retrieve_node)
_builder.add_node("generate", generate_node)
_builder.add_edge(START, "retrieve")
_builder.add_edge("retrieve", "generate")

# In-memory checkpointer: memory lives for the life of the process (fine for
# a single Streamlit session/server). Swap for a persistent checkpointer
# (e.g. langgraph.checkpoint.sqlite.SqliteSaver) if you need memory to
# survive app restarts.
_checkpointer = MemorySaver()
chat_graph = _builder.compile(checkpointer=_checkpointer)


# -------------------------------
# Public API
# -------------------------------
def ask(question: str, thread_id: str):
    """
    Ask a question within the conversation identified by thread_id
    (one thread_id per uploaded PDF). Returns (answer, source_chunks).
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = chat_graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    answer = result["messages"][-1].content
    source_chunks = result.get("source_chunks", [])
    return answer, source_chunks


def get_history(thread_id: str):
    """Return the LangChain message objects stored for this PDF's thread."""
    config = {"configurable": {"thread_id": thread_id}}
    state = chat_graph.get_state(config)
    return state.values.get("messages", []) if state and state.values else []