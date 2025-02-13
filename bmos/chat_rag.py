from gradio import ChatInterface
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "liang-lang-smith"
os.environ["LANGSMITH_API_KEY"] = "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"


vectorstore = Chroma(persist_directory="document_loader/chroma_data",
                     collection_name="bmos",
                     embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="deepseek-r1:latest")

template = """
    上下文信息如下
    ---------------------
    {context}
    ---------------------
    请根据以上提供的上下文信息，回答以下问题
    回答尽量从上下文中总结
    问题: {input}
    答案:
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

chain = create_retrieval_chain(retriever, question_answer_chain)



def chat(message, history):
    result = ""
    for chunk in chain.stream({"input": message, "history":[]}):
        result += chunk.get("answer", "")
        if result != "":
            yield result

if __name__ == '__main__':
    chain.get_graph().print_ascii()
    gr = ChatInterface(fn=chat, type="messages")
    gr.launch()