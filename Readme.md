# Agentic Chatbot [https://agentic-chatbot-hfdjfm2ua8gcbcu8xoemqb.streamlit.app/]

**Agentic Chatbot** is a customizable AI-powered chatbot built using **LangGraph**, **LangChain**, and **HuggingFace LLMs**. It allows users to interact with AI, perform tool-assisted tasks like web search, calculator operations, and fetching stock prices. The chatbot supports **persistent conversation threads**, **user-specific API keys**, and a clean **Streamlit UI**.

---

## Features

* **User Authentication**
  Simple username/password login stored in session state.

* **Persistent Conversations**
  Conversations are saved per user and can be revisited anytime.

* **Multiple Conversation Threads**
  Users can maintain multiple independent chat threads.

* **Streaming AI Responses**
  Responses are streamed character-by-character for a real-time chat experience.

* **Tool-Enhanced LLM**

  * Web search via DuckDuckGo
  * Arithmetic calculator
  * Stock price lookup via Alpha Vantage API

* **Custom API Key Support**
  Users can use their own HuggingFace API key or default key.

* **SQLite Database**
  Stores chat states and enables retrieval of conversation history.

---

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

* Optional: Users can enter their own API key from the Streamlit sidebar.

5. **Run the Streamlit app**

```bash
streamlit run frontend.py
```

---

## Usage

1. **Login**
   Enter your username and password to log in. Passwords cannot contain `_` symbol.

2. **Start a Chat**
   Click “New Chat” in the sidebar to start a new conversation thread.

3. **View Conversation History**
   Select a conversation thread from the sidebar to load past messages.

4. **Enter Your API Key (Optional)**
   In the sidebar, enable “Use my own API key” and input your HuggingFace token.
   If no key is entered, the default key from `.env` will be used.

5. **Interact with AI**
   Type your message in the chat input box. Responses may include tool outputs like search results, calculation answers, or stock prices.

---

## Backend Architecture

* **StateGraph (LangGraph)** handles the chatbot logic with nodes:

  * `chat_node`: Generates AI responses using HuggingFace LLM
  * `tools`: Executes tool-specific functions like calculator, search, and stock price

* **Checkpoints (SQLite)**
  Conversations are saved in `chatbot.db` using `SqliteSaver` for persistence.

* **Tools**

  * `DuckDuckGoSearchRun()` for web search
  * `calculator()` for arithmetic
  * `get_stock_price()` for live stock information

* **LLM Configuration**
  HuggingFace endpoint can be dynamically configured per user session with their API key.

---

## Notes

* **Security**: Passwords are currently stored in session state in plain text. It’s recommended to implement **hash + salt** for production use.
* **API Key Limits**: If using the default HuggingFace API key, usage may be limited. Users can provide their own keys.
* **Deployment**: The app can be deployed on **Streamlit Cloud**, **Heroku**, or any server that supports Python 3.10+. SQLite is lightweight and works for small-scale deployments.

---

