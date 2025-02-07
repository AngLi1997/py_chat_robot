from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

model = OllamaLLM(model="llama3.1")
parser = StrOutputParser()
message = [
    SystemMessage(content="请将下列的句子翻译成中文"),
    HumanMessage(content="lately I've been losing sleep"),
]
result = model.invoke(message)
print(parser.invoke(result))
