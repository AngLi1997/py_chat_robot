import os

from gradio import ChatInterface, ChatMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# 使用smith进行调试
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "liang-lang-smith"
os.environ["LANGSMITH_API_KEY"] = "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"

# 基本模型
model = OllamaLLM(model="deepseek-r1:latest")
# 向量库
vectorstore = Chroma(persist_directory="./chroma_data", embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
retriever = vectorstore.as_retriever()

# 提示词

template = """

你是一个公司规章制度的问答机器人,请以检索到的上下文的说明进行分析和回答
回答尽可能低贴近原文档
只针对用户问题相关进行相关回答
如果有你不知道的问题，直接回答“我不知道”

上下文:{context}

"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)


def chat(message, history):
    result = ""
    for chunk in rag_chain.stream({"input": message}):
        result += chunk.get("answer", "")
        yield result


if __name__ == '__main__':
    gr = ChatInterface(fn=chat, type="messages")
    gr.launch()
