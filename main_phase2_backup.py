"""
main.py
------------------------------------
Entry point for the Private Agentic RAG System

Author : Kathula Deepak
Version : Phase2
"""
from unittest import result

from application.logger.experiment_logger import ExperimentLogger

from config import (
    FAISS_INDEX,
    METADATA_FILE,

    SLM_MODEL,
    EMBEDDING_MODEL,

    CHUNK_SIZE,
    CHUNK_OVERLAP,

    TOP_K,

    PROJECT_NAME,
    VERSION
)

from application.vectordb.faiss_store import VectorStore
from application.retrieve.retriever import retrieve
from application.logger.experiment import Experiment
from application.rag_pipeline.rag import (
    run_query_pipeline
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

    database_info = store.get_database_summary()

    document_count = database_info["documents"]

    total_chunks = database_info["chunks"]

    logger = ExperimentLogger()

    print("Vector Database Loaded Successfully!")

    while True:

        query = input("\nEnter your question (or type 'exit'): ")

        if query.lower() == "exit":

            print("\nGoodbye!")

            break

        result = run_query_pipeline(
            store,
            query
        )


        answer = result["answer"]

        context_chunks = result["context_chunks"]

        metrics = result["metrics"]


        logger.log(
            config={

                "project": PROJECT_NAME,

                "version": VERSION,

                "model": SLM_MODEL,

                "embedding": EMBEDDING_MODEL,

                "chunk_size": CHUNK_SIZE,

                "overlap": CHUNK_OVERLAP,

                "top_k": TOP_K,

                "documents": document_count,

                "chunks": total_chunks

            },

            metrics=metrics,

            question=query,

            answer=answer
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