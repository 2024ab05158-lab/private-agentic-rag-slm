from application.embedd.embedder import model

from application.vectordb.faiss_store import VectorStore

from application.retrieve.retriever import retrieve

from application.rag_pipeline.rag import (
    build_prompt,
    get_query_embedding
)

from application.slm.slm import generate_response


INDEX_PATH = "data/vector_db/faiss.index"

METADATA_PATH = "data/vector_db/metadata.pkl"


def load_vector_database():

    store = VectorStore()

    store.load(
        INDEX_PATH,
        METADATA_PATH
    )

    return store


def display_sources(results):

    unique_sources = []

    for item in results:

        source = item["source"]

        if source not in unique_sources:

            unique_sources.append(source)

    print("\nSources Used:")

    for source in unique_sources:

        print(f"✓ {source}")


def start_chat():

    print("=" * 60)

    print("Private Agentic RAG System")

    print("=" * 60)

    print("\nLoading vector database...")

    store = load_vector_database()

    print("Ready!")

    while True:

        query = input("\nEnter your question (or type exit): ")

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