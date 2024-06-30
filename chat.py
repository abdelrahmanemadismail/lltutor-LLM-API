import streamlit as st
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
import time

# Define a custom callback handler to capture the streaming output
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.output = ""

    def on_new_token(self, token: str, **kwargs):
        self.output += token
        st.session_state.output = self.output
        time.sleep(0.05)  # Add a slight delay to simulate streaming

# Initialize the custom callback handler
callback_handler = StreamlitCallbackHandler()

# Define the prompt template
template = """
<s>[INST] <<SYS>>
You are an AI virtual tutor specialized in Software Engineering, Your responses must be clear, concise, and contextually appropriate. Only address inquiries directly related to software topics.
<</SYS>>
{chat_history}

Student: {text}
Tutor: [/INST]
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "text"],
    template=template,
)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([callback_handler])

model_path = "Llama-2-7b-chat-hf-SW2-test-fine-tuned-cpu/Q4_K_M.gguf"  # Adjust the path as needed
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.5,
    n_ctx=4096,
    max_tokens=4096,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

chain = prompt | llm

password = "Pass123456789"
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: RedisChatMessageHistory(
        session_id, url=f"redis://:{password}@192.168.1.201:6379/0"),
    input_messages_key="text",
    history_messages_key="chat_history",
)

config = {"configurable": {"session_id": "t50"}}

# Streamlit app
st.set_page_config(page_title="LLTutor", page_icon="🧑‍🏫")

# CSS to style the chat interface
st.markdown("""
    <style>
    .chat-box {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .student-msg {
        color: blue;
    }
    .tutor-msg {
        color: green;
    }
    .input-box {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #fff;
        padding: 10px;
        box-shadow: 0 -1px 3px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("Course Overview")
st.sidebar.header("Network")
st.sidebar.header("Software Engineering")
st.sidebar.header("System Design")
st.sidebar.header("My Profile")
st.sidebar.header("Settings")
st.sidebar.header("Info")

st.title("AI Tutor")

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'output' not in st.session_state:
    st.session_state.output = ""

def generate_response(user_input):
    st.session_state.output = ""  # Clear previous output
    response = chain_with_history.invoke({"text": user_input}, config=config)
    return response

# Display chat messages
for message in st.session_state.messages:
    if message['role'] == 'Student':
        st.markdown(f"<div class='chat-box student-msg'><strong>Student:</strong> {message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-box tutor-msg'><strong>Tutor:</strong> {message['content']}</div>", unsafe_allow_html=True)

# Real-time streaming output
if st.session_state.output:
    st.markdown(f"<div class='chat-box tutor-msg'><strong>Tutor:</strong> {st.session_state.output}</div>", unsafe_allow_html=True)

# Message input box at the bottom
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("Enter your message:", key="input")
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    st.session_state.messages.append({"role": "Student", "content": user_input})
    response = generate_response(user_input)
    st.session_state.messages.append({"role": "Tutor", "content": response})
    st.experimental_rerun()

# Ensure input box stays at the bottom
st.markdown("<div class='input-box'></div>", unsafe_allow_html=True)
