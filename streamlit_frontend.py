import streamlit as st
from langgraph_backend import Chatbot, retrieve_all_threads,ingest_pdf, thread_document_metadata
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import time
import json


def generate_thread_id():
    return str(uuid.uuid4())


def show_message(message, role, tool_calls=None):
    """Display a chat message with optional tool call dropdown."""
    with st.chat_message(role):
        st.markdown(message)
        if tool_calls:
            with st.expander(f"🔧 Tool Calls ({len(tool_calls)})", expanded=False):
                for tc in tool_calls:
                    st.markdown(f"**Tool:** `{tc.get('name', 'tool')}`")
                    st.markdown("**Input:**")
                    st.json(tc.get('args', {}))
                    if tc.get('result') is not None:
                        st.markdown("**Result:**")
                        st.code(str(tc['result'])[:500], language="json")
                    st.divider()


def new_chat_window():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_chat_thread(thread_id)
    st.session_state['message_history'] = []


def add_chat_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    """Load conversation from LangGraph state and extract tool call data."""
    try:
        msgs = Chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']
    except:
        msgs = []

    history = []
    pending_tool_calls = {}  # tool_call_id -> {name, args, result}

    for msg in msgs:
        if isinstance(msg, HumanMessage):
            history.append({'role': 'user', 'content': msg.content, 'tool_calls': []})

        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get('id', '')
                    pending_tool_calls[tc_id] = {
                        'name': tc.get('name', 'tool'),
                        'args': tc.get('args', {}),
                        'result': None,
                    }
            elif msg.content:
                resolved = list(pending_tool_calls.values())
                pending_tool_calls.clear()
                history.append({
                    'role': 'assistant',
                    'content': msg.content,
                    'tool_calls': resolved,
                })

        elif isinstance(msg, ToolMessage):
            tc_id = getattr(msg, 'tool_call_id', '')
            if tc_id in pending_tool_calls:
                pending_tool_calls[tc_id]['result'] = msg.content

    return history


def get_thread_name(thread_id):
    """Get a display name for a thread. Uses the first user message, or falls back to truncated ID."""

    if thread_id in st.session_state.get('thread_names', {}):
        return st.session_state['thread_names'][thread_id]

    # Try to peek at the first message from LangGraph state
    try:
        msgs = Chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values.get('messages', [])
        for msg in msgs:
            if isinstance(msg, HumanMessage) and msg.content:
                name = msg.content[:30].strip()
                if len(msg.content) > 30:
                    name += "..."
                st.session_state['thread_names'][thread_id] = name
                return name
    except:
        pass

    return f"New Chat ({str(thread_id)[:8]})"


# ============================================================================
# Session State
# ============================================================================

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'thread_names' not in st.session_state:
    st.session_state['thread_names'] = {}

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_chat_thread(st.session_state['thread_id'])
thread_docs = st.session_state["ingested_docs"].setdefault(
    st.session_state["thread_id"], {}
)
# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.title('🤖 MCP Agent Chatbot')

if st.sidebar.button('➕ New Chat', use_container_width=True):
    new_chat_window()
    st.rerun()

st.sidebar.header('💬 Conversations')

# Sort: active thread first, then the rest in reverse order (newest first)
active_tid = st.session_state.get('thread_id')
other_threads = [t for t in st.session_state['chat_threads'] if str(t) != str(active_tid)]
sorted_threads = ([active_tid] if active_tid else []) + other_threads[::-1]

for thread_id in sorted_threads:
    label = get_thread_name(thread_id)
    is_active = str(thread_id) == str(active_tid)
    if st.sidebar.button(f"{'▶ ' if is_active else '💬 '}{label}", key=f"t_{thread_id}", use_container_width=True):
        st.session_state['thread_id'] = thread_id
        st.session_state['message_history'] = load_conversation(thread_id)
        st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=st.session_state.get("thread_id"),
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.divider()
st.sidebar.caption("**MCP Tools:** 🧮 Calculator · 📈 Stock Price · 🔍 Web Search")

# ============================================================================
# Main Chat Area
# ============================================================================

st.title("MCP Agent Chatbot")
st.caption(f"🧵 Thread: `{str(st.session_state['thread_id'])[:18]}...`")

# Display chat history
for message in st.session_state['message_history']:
    show_message(message['content'], message['role'], message.get('tool_calls', []))

# Config for LangGraph
CONFIG = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {"thread_id": st.session_state["thread_id"]},
    "run_name": "chat_turn",
}

# ============================================================================
# Chat Input & Streaming Response
# ============================================================================

user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input, 'tool_calls': []})
    show_message(user_input, 'user')

    # Set thread name from first message if not already set
    tid = st.session_state['thread_id']
    if tid not in st.session_state['thread_names']:
        name = user_input[:30].strip()
        if len(user_input) > 30:
            name += "..."
        st.session_state['thread_names'][tid] = name

    with st.chat_message('assistant'):
        # Use st.status for the live thinking/tool-calling indicator
        status = st.status("Thinking...", expanded=True)
        response_placeholder = st.empty()

        full_response = ""
        collected_tool_calls = []
        pending_tc = {}

        for event in Chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="updates",
        ):
            if 'chat_node' in event:
                msg = event['chat_node']['messages'][0]

                # AI decided to call tools
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        tool_args = tc.get('args', {})
                        tc_id = tc.get('id', '')

                        entry = {'name': tool_name, 'args': tool_args, 'result': None}
                        collected_tool_calls.append(entry)
                        pending_tc[tc_id] = len(collected_tool_calls) - 1

                        status.update(label=f"🔧 Calling tool: **{tool_name}**...", expanded=True)
                        status.write(f"**Tool:** `{tool_name}`")
                        status.json(tool_args)

                # Regular AI text response → stream it
                elif isinstance(msg, AIMessage) and msg.content:
                    status.update(label="✅ Done", state="complete", expanded=False)
                    text = msg.content if isinstance(msg.content, str) else msg.content[0].get("text", "")
                    if text:
                        for char in text:
                            full_response += char
                            response_placeholder.markdown(full_response + "▌")
                            time.sleep(0.015)
                        response_placeholder.markdown(full_response)

            # Tool result event
            elif 'tools' in event:
                tool_msg = event['tools']['messages'][0]
                tool_result = tool_msg.content if hasattr(tool_msg, 'content') else str(tool_msg)
                tc_id = getattr(tool_msg, 'tool_call_id', '')

                if tc_id in pending_tc:
                    collected_tool_calls[pending_tc[tc_id]]['result'] = tool_result

                status.update(label="⚙️ Processing tool results...", expanded=True)
                status.write(f"**Result:** `{str(tool_result)[:200]}`")

        # Finalize the status indicator
        status.update(label="✅ Done", state="complete", expanded=False)

        # Show tool calls in a collapsible expander (persists on rerun)
        if collected_tool_calls:
            with st.expander(f"🔧 Tool Calls ({len(collected_tool_calls)})", expanded=False):
                for tc in collected_tool_calls:
                    st.markdown(f"**Tool:** `{tc['name']}`")
                    st.json(tc['args'])
                    if tc.get('result') is not None:
                        st.markdown("**Result:**")
                        st.code(str(tc['result'])[:500], language="json")
                    st.divider()

        if not full_response:
            full_response = "I processed your request. Check the tool results above."
            response_placeholder.markdown(full_response)

    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': full_response,
        'tool_calls': collected_tool_calls,
    })
