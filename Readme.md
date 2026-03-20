## Agentic Chatbot [https://agentic-chatbot-1dm5.onrender.com/]

# 🤖 MCP Agent Chatbot (LangGraph + RAG + Tool Calling)

An advanced **Agentic Chatbot** built using **LangGraph, MCP (Model Context Protocol), and HuggingFace LLMs**.  
This system supports **tool calling, RAG-based document understanding, multi-threaded conversations, and real-time streaming responses**.

---

## ✨ Features

### 🧠 Agentic AI System
- Built using **LangGraph** for structured multi-step reasoning
- Supports **tool calling** with dynamic decision-making
- Uses **HuggingFace LLM (MiniMax)** for inference

---

### 🔌 MCP Tool Integration
- Custom **FastMCP server** exposing tools:
  - 🧮 Calculator (arithmetic operations)
  - 📈 Stock Price Fetcher (Alpha Vantage API)
  - 🔍 Web Search (DuckDuckGo)
- Tools are dynamically loaded using **MultiServerMCPClient**
- Fully decoupled architecture (LLM ↔ Tool Server)

---

### 📄 RAG (Retrieval-Augmented Generation)
- Upload PDFs and chat with them
- Pipeline:
  - PDF → Chunking → Embeddings → FAISS
- Thread-specific document context
- Uses **HuggingFace Embeddings (MiniLM)**

---

### 💬 Multi-Threaded Conversations
- Each chat has a unique **thread_id**
- Conversations persisted using **SQLite (LangGraph Checkpointer)**
- Switch between chats seamlessly
- Chat naming based on first user query

---

### ⚡ Streaming Responses
- Real-time token streaming in UI
- Tool execution status shown live
- Smooth conversational UX

---

### 🔧 Tool Execution Transparency
- View:
  - Tool name
  - Input arguments
  - Output results
- Debug-friendly UI using Streamlit expanders

---

### 🧱 Scalable Architecture
- Modular separation:
  - `langgraph_backend.py` → Agent + Graph
  - `mcp_server.py` → Tool server
  - `streamlit_frontend.py` → UI
- Designed for:
  - Future microservices
  - Docker + cloud deployment

---

## 🏗️ Architecture Overview

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/agentic-chatbot.git
cd agentic-chatbot
```

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Setup environment variables**
   Create a `.env` file in the root directory:

```env
HUGGINGFACEHUB_API_TOKEN=your_default_huggingface_token
```

5. **Run the Streamlit app**

```bash
streamlit run streamlit_frontend.py
```

---

