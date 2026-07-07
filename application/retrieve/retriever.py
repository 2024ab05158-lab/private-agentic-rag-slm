"""
retriever.py
---------------------------------------
Core retrieval component.

Supports:
1. Fixed Top-K retrieval for Core RAG
2. Dynamic Top-K retrieval for Agentic RAG
"""


from config import TOP_K


def retrieve(
        vector_store,
        query_embedding,
        top_k=None
):

    """
    Fetch most relevant chunks from vector database.
    """

    if top_k is None:

        top_k = TOP_K


    results = vector_store.search(
        query_embedding,
        top_k
    )


    return results