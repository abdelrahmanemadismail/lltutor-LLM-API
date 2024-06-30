import time
import copy
import asyncio
import requests

from fastapi import FastAPI, Request
from llama_cpp import Llama
from sse_starlette import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.llms import LlamaCpp
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import wolframalpha

app_id = "XRP95P-AT5K72TQGP"
wolfram_client = wolframalpha.Client(app_id)

async def query_wolfram_alpha(query):
    """Query Wolfram Alpha and return the first result."""
    res = await wolfram_client.aquery(query)
    try:
        return next(res.results).text
    except StopIteration:
        return None
    
print("Loading model...")

template = """
<s>[INST] <<SYS>>
You are an AI virtual tutor specialized in Math. Your responses must be clear, concise, and contextually appropriate. Only address inquiries directly related to math topics.
<</SYS>>
{chat_history}

Problem and Solution: {text}
Tutor: [/INST]
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "text"],
    template=template,
)

model_path = "Llama-2-7b-chat-hf-SW2-test-fine-tuned-cpu/Q4_K_M.gguf"
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.5,
    n_ctx=4096,
    max_tokens=4096,
    top_p=1,
)

chain = prompt | llm
password = "Pass123456789"

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: RedisChatMessageHistory(
        session_id, url=f"redis://:{password}@192.168.1.201:6379/0"
    ),
    input_messages_key="text",
    history_messages_key="chat_history",
)

print("Model loaded!")
app = FastAPI()

# Enable CORS
origins = [
    "http://localhost",
    "http://localhost:3000",  # React dev server
    "http://localhost:5173",  # React dev server
    "https://3584-156-210-15-233.ngrok-free.app"  # Adjust with your frontend domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_prefix = "Wolfram-" 

@app.post("/chat/wolfram/")
async def stream(request: Request):
    body = await request.json()
    session_id = body.get("session_id")
    question = body.get("question")
    text = await query_wolfram_alpha(question)

    async def event_publisher():
        config = {"configurable": {"session_id": session_prefix + session_id}}
        await asyncio.sleep(1)
        output = chain_with_history.invoke({"text": text}, config=config)
        yield {"data": output}

    return EventSourceResponse(event_publisher())

@app.get("/chat/wolfram/")
async def get_history(session_id: str):
    session_history = chain_with_history.get_session_history(session_prefix + session_id)
    history_data = session_history.messages
    return {"history": history_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)