from gradio import ChatInterface
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama import OllamaEmbeddings, ChatOllama
import os

# 接入LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "liang-lang-smith"
os.environ["LANGSMITH_API_KEY"] = "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"

# 知识库
vectorstore = Chroma(persist_directory="document_loader/chroma_data",
                     collection_name="bmos",
                     embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"))
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 主对话模型
llm = ChatOllama(model="llama3.1")

# 提示词模版
template = """
    上下文信息如下
    ---------------------
    {context}
    ---------------------
    如果用户的输入不是问题，则直接根据自己的思考回答；
    如果用户问了问题，则根据上下文信息进行回答
"""

# 问题
question_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 历史消息
history_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("human", "在给定上述对话的情况下，生成一个要查找的搜索查询，以获取与对话相关的信息")
])

# 历史消息链
history_chain = create_history_aware_retriever(llm, retriever, history_prompt)

# 提问链
question_answer_chain = create_stuff_documents_chain(llm, question_prompt)

# 创建检索链
chain = create_retrieval_chain(history_chain, question_answer_chain)

# gradio的对话函数
def chat(message, history):
    result = ""
    # 流式消息
    for chunk in chain.stream({"input":message, "chat_history": history}, config={"configurable": {"session_id": "1"}}):
        result += chunk.get("answer", "")
        if result != "":
            yield result

if __name__ == '__main__':
    # 打印调用链
    chain.get_graph().print_ascii()
    # 启动对话
    gr = ChatInterface(fn=chat, type="messages")
    gr.launch()