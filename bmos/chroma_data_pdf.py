from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, PDFMinerLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 加载PDF
loader = PyPDFLoader("./pdfs/test.pdf")
doc = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_data = text_splitter.split_documents(doc)

Chroma.from_documents(documents=split_data, embedding=OllamaEmbeddings(model="nomic-embed-text"), persist_directory="./chroma_data")