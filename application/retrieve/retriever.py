from config import TOP_K

def retrieve(vector_store, query_embedding, top_k=TOP_K):
    """
    Fetch most relevant chunks from vector database.
    """
    return vector_store.search(query_embedding, top_k)