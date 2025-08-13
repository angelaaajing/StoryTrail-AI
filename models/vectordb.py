import chromadb

client = chromadb.PersistentClient(path="./data/chroma_db")


def add_item(collection_name, id, embedding, metadata, document=""):
    col = client.get_or_create_collection(collection_name)
    col.add(ids=[id], embeddings=[embedding.tolist()],
            metadatas=[metadata], documents=[document])
    print(f"Added item {id} to collection '{collection_name}'")


def search_collection(collection_name, query_embedding, n_results=5):
    collection = client.get_or_create_collection(collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results
