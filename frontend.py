import streamlit as st
from langgraph_backend import Chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import time
from streamlit_local_storage import LocalStorage
import json
import os

locals = LocalStorage()

def generate_thread_id():
    user = st.session_state["user"]
    thread_id = str(uuid.uuid4())
    return user['username'] + user['password'] + "_" + thread_id

def show_message(message, role):
    with st.chat_message(role):
        st.text(message)

def new_chat_window():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_chat_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_chat_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    try:
        return Chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']
    except:
        return []

if "user" not in st.session_state:
    st.session_state["user"] = None
    
if st.session_state["user"] is None:
    st.title("Login to Agentic Chatbot")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password and '_' not in password:
            st.session_state["user"] = {"username": username, "password": password}  # TODO: store hash+salt
            st.rerun()
        else:
            if password and '_' in password:
                st.error("Password cannot have _ symbol")
            else:
                st.error("Please enter both username and password")
            
else:

    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []

    if 'thread_id' not in st.session_state:
        st.session_state['thread_id'] = generate_thread_id()

    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = retrieve_all_threads(st.session_state['user'])

    add_chat_thread(st.session_state['thread_id'])

    st.sidebar.title('LangGraph Chatbot')

    if st.sidebar.button('New Chat'):
        new_chat_window()

    st.sidebar.header('MY Conversations')
    for thread_id in st.session_state['chat_threads'][::-1]:
        if st.sidebar.button(str(thread_id)):
            st.session_state['thread_id'] = thread_id 
            messages = load_conversation(thread_id)

            temp_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = 'user'
                else:
                    role = 'assistant'
                temp_messages.append({'role': role, 'content': msg.content})

            st.session_state['message_history'] = temp_messages

    st.sidebar.header("API Key Settings")
    DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")

    use_custom_key = st.sidebar.checkbox("Use my own API key")
    if use_custom_key:
        user_api_key = st.sidebar.text_input("Enter OpenAI API key", type="password")
        if user_api_key:
            st.session_state["api_key"] = user_api_key
            st.sidebar.success("Using your custom API key ")
        else:
            st.session_state["api_key"] = DEFAULT_API_KEY
            st.sidebar.warning("No key entered, using default ")
    else:
        st.session_state["api_key"] = DEFAULT_API_KEY
        st.sidebar.info("Using default API key")

    active_api_key = st.session_state["api_key"]


    for message in st.session_state['message_history']:
        show_message(message['content'], message['role'])

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
            "api_key": active_api_key,   
        },
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn"
    }

    user_input = st.chat_input('Type Here')

    if user_input:
        st.session_state['message_history'].append({'role': 'user', 'content': user_input})
        show_message(user_input, 'user')

        with st.chat_message('assistant'):
            def get_full_response():
                for event in Chatbot.stream(
                    {'messages': [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="updates"
                ):
                    if 'chat_node' in event:
                        msg = event['chat_node']['messages'][0]
                        if isinstance(msg, AIMessage):
                            text = msg.content if isinstance(msg.content, str) else msg.content[0].get("text", "")
                            text = text.replace("<tool_call>", "").strip()
                            for char in text:
                                yield char
                                time.sleep(0.02)

                    elif 'tools' in event:
                        tool_msg = event['tools']['messages'][0].content
                        yield f"**Tool Call:** `{tool_msg}`"
                    else:
                        ai_message = "Unhandled message"
                        yield ai_message
                    
            ai_message = st.write_stream(get_full_response())   

        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
