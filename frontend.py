import streamlit as st
from langgraph_backend import Chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import time

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

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


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()
    print(st.session_state['chat_threads'])

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

for message in st.session_state['message_history']:
    show_message(message['content'], message['role'])

CONFIG = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {
        "thread_id": st.session_state["thread_id"]
    },
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
                    if isinstance(msg,AIMessage):
                        text = msg.content if isinstance(msg.content, str) else msg.content[0].get("text", "")
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

