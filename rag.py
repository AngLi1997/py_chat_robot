import os

from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "liang-lang-smith"
os.environ["LANGSMITH_API_KEY"]="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"


model = OllamaLLM(model="llama3.1")
parser = StrOutputParser()
message = [
    SystemMessage(content="请将下列的句子翻译成中文"),
    HumanMessage(content="lately I've been losing sleep"),
]
result = model.invoke(message)
print(parser.invoke(result))
