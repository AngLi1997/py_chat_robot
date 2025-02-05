import chromadb

if __name__ == '__main__':
    chroma_client = chromadb.PersistentClient("./chroma_db")
    collection = chroma_client.get_or_create_collection("my_collection")
    collection.add(documents=["This is a document about engineer", "This is a document about steak"],
                   metadatas=[{"source": "doc1"}, {"source": "doc2"}],
                   ids=["id1", "id2"])
    results = collection.query(query_texts=["Which food is the best?"], n_results=2)
    print(results)
