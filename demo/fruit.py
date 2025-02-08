from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA


llm = OllamaLLM(model="llama3.1", temperature=0.1)
embeddings = OllamaEmbeddings(model="llama3.1")
source_document = "txt/fruit_data.txt"
chroma_path = "../chroma_db"


def load_knowledge():
    loader = TextLoader(source_document)
    document = loader.load()
    cts = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = cts.split_documents(document)
    Chroma.from_documents(texts, embedding=embeddings, persist_directory=chroma_path, collection_name="fruit")


def chat():
    vectordb = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever())
    print(qa.invoke({"query": "世界上有哪些水果"}))


if __name__ == '__main__':
    # load_knowledge()
    chat()