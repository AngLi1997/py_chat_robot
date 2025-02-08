import os

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "liang-lang-smith"
os.environ["LANGSMITH_API_KEY"]="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"


model = OllamaLLM(model="llama3.1")

message = [
    HumanMessage(content="我叫什么名字")
]

store = {}

config={
    "configurable":{"session_id":"liang"}
}

def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


client = RunnableWithMessageHistory(model, get_session_history)

resp = client.invoke(message, config=config)

print(resp)