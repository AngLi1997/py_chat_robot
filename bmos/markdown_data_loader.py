from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("开始加载知识库...")

# 加载文档
loader = DirectoryLoader("./md", glob="**/*.md", silent_errors=True)
docs = loader.load()
for doc in docs:
    print(f'加载文件:{doc.metadata["source"]}')


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

# 嵌入模型
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
Chroma.from_documents(texts, collection_name="bmos", embedding=embeddings, persist_directory="./chroma_data")

print("知识库加载完成!")