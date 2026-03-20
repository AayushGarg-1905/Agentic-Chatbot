"""
FastMCP Server exposing tools for the LangGraph Chatbot.

Run this server before starting the Streamlit frontend:
    python mcp_server.py

Tools:
    - calculator: Perform basic arithmetic (add, sub, mul, div)
    - get_stock_price: Fetch stock prices from Alpha Vantage
    - search_web: Search the web via DuckDuckGo
"""

from mcp.server.fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from dotenv import load_dotenv
import os
load_dotenv()

mcp = FastMCP("tools-server")


@mcp.tool()
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
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

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using the Alpha Vantage API.
    """
    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={symbol}"
            f"&apikey={os.getenv('STOCK_API')}"
        )
        r = requests.get(url, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def search_web(query: str) -> dict:
    """
    Search the web using DuckDuckGo and return the results.
    """
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return {"query": query, "result": result}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
