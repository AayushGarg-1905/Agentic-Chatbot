from typing import TypedDict
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
import os

load_dotenv()

# tools
search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}  

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol using Alpha Vantage API."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=O0D68F5T2H0RWUIV"
    r = requests.get(url)
    return r.json()

tools = [search_tool, get_stock_price, calculator]


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState, config):
    
    messages = state['messages']

    api_key = None
    if config and "configurable" in config:
        api_key = config["configurable"].get("api_key")

    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    model = ChatHuggingFace(llm=llm)
    llm_with_tools = model.bind_tools(tools)

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", lambda state, config=None: chat_node(state, config))
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

Chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads(user):
    all_threads = set()
    username = user['username']
    password = user['password']
    key = str(username) + str(password)
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if key == thread_id.split('_')[0]:
            all_threads.add(thread_id)
    return list(all_threads)
