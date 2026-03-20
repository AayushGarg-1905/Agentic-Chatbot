"""
LangGraph Chatbot Backend with MCP Integration.

This backend uses MultiServerMCPClient from langchain_mcp_adapters
to dynamically load tools from the FastMCP server (mcp_server.py).

Usage:
    1. The MCP server is started automatically via stdio transport.
    2. Run the frontend: streamlit run streamlit_frontend.py
"""

from typing import Dict, Optional, TypedDict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEndpointEmbeddings,
)
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from pathlib import Path
import asyncio
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

load_dotenv()

# ============================================================================
# LLM Setup
# ============================================================================

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)
embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",  # Available embedding model on HF Inference API
    provider="hf-inference",  # Specify the inference provider
)

THREAD_RETRIEVERS: Dict[str, any] = {}
THREAD_METADATA: Dict[str, dict] = {}


def get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in THREAD_RETRIEVERS:
        return THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(
    file_bytes: bytes, thread_id: str, filename: Optional[str] = None
) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embedding_model)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        THREAD_RETRIEVERS[str(thread_id)] = retriever
        THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ============================================================================
# MCP Client - Load tools from MCP Server
# ============================================================================

# Path to the MCP server script (same directory as this file)
MCP_SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")

# Create the MultiServerMCPClient with stdio transport
# The MCP server will be started automatically as a subprocess
mcp_client = MultiServerMCPClient(
    {
        "tools-server": {
            "command": "python",
            "args": [MCP_SERVER_PATH],
            "transport": "stdio",
        }
    }
)


def _patch_tool_sync(tool):
    """
    Add sync invocation support to async-only MCP tools.
    langchain_mcp_adapters returns StructuredTools with only 'coroutine' set.
    ToolNode calls .invoke() (sync), so we need to provide a sync 'func' as well.
    """
    if tool.coroutine is not None and tool.func is None:
        original_coro = tool.coroutine

        def _sync_invoke(**kwargs):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(original_coro(**kwargs))
            finally:
                loop.close()

        tool.func = _sync_invoke
    return tool


async def _load_tools():
    """Fetch all tools from the MCP server."""
    return await mcp_client.get_tools()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.

    Always include the thread_id when calling this tool.
    """
    retriever = get_retriever(thread_id)

    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


# Load tools at module level using asyncio, then patch for sync support
tools = [_patch_tool_sync(t) for t in asyncio.run(_load_tools())]
tools.append(rag_tool)
llm_with_tools = model.bind_tools(tools)

# ============================================================================
# LangGraph State and Nodes
# ============================================================================


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState, config=None):

    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "If the user asks questions about an uploaded PDF, "
            "you MUST call the `rag_tool` and include the thread_id "
            f"`{thread_id}`. "
            "Use other tools like calculator, stock price, or search when appropriate."
        )
    )

    messages = [system_message, *state["messages"]]

    response = llm_with_tools.invoke(messages, config=config)

    return {"messages": [response]}


tool_node = ToolNode(tools)

# ============================================================================
# Graph Construction
# ============================================================================

conn = sqlite3.connect(
    database=str(Path(__file__).parent / "chatbot.db"),
    check_same_thread=False,
)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

Chatbot = graph.compile(checkpointer=checkpointer)

# ============================================================================
# Utility Functions
# ============================================================================


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return THREAD_METADATA.get(str(thread_id), {})
