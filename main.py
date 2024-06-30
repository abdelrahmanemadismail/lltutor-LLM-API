import time
import copy
import asyncio
import requests

from fastapi import FastAPI, Request
from llama_cpp import Llama
from sse_starlette import EventSourceResponse

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory

print("Loading model...")
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

# text = "enumerate five software evaluation techniques"
# print(prompt.format(text=text))

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model_path = "Llama-2-7b-chat-hf-SW2-test-fine-tuned-cpu\Q4_K_M.gguf"
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.5,
    n_ctx=4096,
    max_tokens=4096,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

# output = llm.invoke(prompt.format(text=text))
# print(output)

# memory = ConversationBufferMemory(memory_key="chat_history", human_prefix="Student", ai_prefix="Tutor")
# chain = LLMChain(
#     llm=llm,
#     prompt=prompt,
#     memory=memory,
#     verbose=True,
#     )

# while True:
#     text = input("Student: ")
#     if text == "exit":
#         break
#     chain.predict(text=text)
#     print(memory.json(indent=2))

chain = prompt | llm
password = "Pass123456789"
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: RedisChatMessageHistory(
    session_id, url = f"redis://:{password}@192.168.1.201:6379/0"),
    input_messages_key="text",
    history_messages_key="chat_history",
)
config = {"configurable": {"session_id": "t5"}}

print("Model loaded!")

while True:
    text = input("Student: ")
    if text == "exit":
        break
    chain_with_history.invoke({"text": text}, config=config)
