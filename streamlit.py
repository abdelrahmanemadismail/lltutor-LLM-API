import streamlit as st
import requests
import os

# Constants
SESSION_ID = "t7"
API_URL = "http://127.0.0.1:8000/chat/SW2/"
TIMEOUT = 600  # Timeout for HTTP requests in seconds

# Path to the favicon
favicon_path = os.path.join(os.path.dirname(__file__), "favicon.ico")

# Function to fetch chat history
def fetch_chat_history():
    try:
        response = requests.get(f"{API_URL}?session_id={SESSION_ID}", timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        history = data.get("history", [])
        return [
            {
                "type": chat["type"],
                "content": chat["content"].replace("AIMessage(content='", "").replace("')", ""),
            }
            for chat in history
        ]
    except requests.RequestException:
        st.error("Failed to fetch chat history. Please try again later.")
        return []

# Function to send a message
def send_message(message):
    data = {
        "session_id": SESSION_ID,
        "question": message,
    }
    try:
        response = requests.post(API_URL, json=data, timeout=TIMEOUT)
        response.raise_for_status()
        reply = response.json()
        return reply.get("reply", "").replace("AIMessage(content='", "").replace("')", "")
    except requests.RequestException:
        st.error("Failed to send message. Please try again later.")
        return "No response from AI"

# Streamlit app layout
st.set_page_config(page_title="LLTutor", page_icon=favicon_path)

# Navbar (simulated using sidebar)
st.sidebar.title("Course Overview")
st.sidebar.header("Network")
st.sidebar.header("Software Engineering")
st.sidebar.header("System Design")

# Chat title
st.title("Chat with LLTutor")

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = fetch_chat_history()

# Chat history container
chat_history = st.container()

# Display chat history
with chat_history:
    for chat in st.session_state.chat_history:
        if chat["type"] == "human":
            st.write(f"**You:** {chat['content']}")
        else:
            st.write("**AI Tutor:** {}".format(chat['content'].replace("\\n", "\n")))

# Chat input container
message = st.text_area("Enter Message", key="message_input", height=100)

# Send button
if st.button("Send"):
    if message.strip():
        st.session_state.chat_history.append({"type": "human", "content": message})
        ai_reply = send_message(message)
        st.session_state.chat_history.append({"type": "ai", "content": ai_reply})
        st.session_state.message_input = ""
        st.experimental_rerun()

# Function to refresh chat history
def refresh_chat_history():
    st.session_state.chat_history = fetch_chat_history()
    st.experimental_rerun()

# Refresh chat history periodically
if 'chat_refresher' not in st.session_state:
    st.session_state.chat_refresher = st.empty()
    refresh_chat_history()

# Display error messages if any
if "error_message" in st.session_state:
    st.error(st.session_state.error_message)