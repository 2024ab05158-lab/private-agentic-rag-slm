def retrieve(vector_store, query_embedding, top_k=2):
    """
    Fetch most relevant chunks from vector database.
    """
    return vector_store.search(query_embedding, top_k)