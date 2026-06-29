"""
main.py
------------------------------------
Entry point for the Private Agentic RAG System

Author : Kathula Deepak
Version : Phase2
"""

from config import (
    FAISS_INDEX,
    METADATA_FILE,
    SLM_MODEL,
    EMBEDDING_MODEL,
    TOP_K
)

from application.vectordb.faiss_store import VectorStore
from application.retrieve.retriever import retrieve
from application.rag_pipeline.rag import (
    build_prompt,
    get_query_embedding
)
from application.slm.slm import generate_response


INDEX_PATH = FAISS_INDEX
METADATA_PATH = METADATA_FILE


def load_vector_database():
    """
    Load the persisted FAISS vector database.
    """

    store = VectorStore()

    store.load(
        INDEX_PATH,
        METADATA_PATH
    )

    return store


def display_sources(results):
    """
    Display unique source documents used to answer the query.
    """

    unique_sources = []

    for item in results:

        source = item["source"]

        if source not in unique_sources:

            unique_sources.append(source)

    print("\nSources Used:")

    for source in unique_sources:

        print(f"✓ {source}")


def print_system_info():
    """
    Display current system configuration.
    """

    print("=" * 60)
    print("Private Agentic RAG System")
    print("=" * 60)

    print(f"SLM Model        : {SLM_MODEL}")
    print(f"Embedding Model  : {EMBEDDING_MODEL}")
    print(f"Top-K Retrieval  : {TOP_K}")

    print("=" * 60)


def start_chat():

    print_system_info()

    print("\nLoading Vector Database...")

    store = load_vector_database()

    print("Vector Database Loaded Successfully!")

    while True:

        query = input("\nEnter your question (or type 'exit'): ")

        if query.lower() == "exit":

            print("\nGoodbye!")

            break

        query_embedding = get_query_embedding(query)

        context_chunks = retrieve(
            store,
            query_embedding
        )

        prompt = build_prompt(
            context_chunks,
            query
        )

        answer = generate_response(
            prompt
        )

        print("\n")

        print("=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(answer)

        display_sources(
            context_chunks
        )

        print("\n")


if __name__ == "__main__":

    start_chat()