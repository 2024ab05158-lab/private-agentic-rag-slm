# main.py
# Entry point for Private Agentic RAG System (Phase 2 MVP)

from application.ingest.pdf_loader import load_pdf
from application.chunk.text_splitter import split_text

from application.embedd.embedder import get_embeddings, model
from application.vectordb.faiss_store import VectorStore

from application.retrieve.retriever import retrieve
from application.rag_pipeline.rag import build_prompt, get_query_embedding

from application.slm.slm import generate_response


def build_rag_pipeline(pdf_path, query):
    """
    End-to-end RAG pipeline:
    1. Load document
    2. Chunk text
    3. Create embeddings
    4. Store in vector DB
    5. Retrieve relevant context
    6. Build prompt
    7. Send to SLM
    """

    # Step 1: Load document
    text = load_pdf(pdf_path)

    # Step 2: Chunk text
    chunks = split_text(text)

    # Step 3: Create embeddings for chunks
    embeddings = get_embeddings(chunks)

    # Step 4: Initialize vector store
    dimension = len(embeddings[0])
    store = VectorStore(dimension)
    store.add(embeddings, chunks)

    # Step 5: Query embedding
    query_embedding = get_query_embedding(query)

    # Step 6: Retrieve relevant chunks
    context_chunks = retrieve(store, query_embedding)

    # Step 7: Build prompt
    prompt = build_prompt(context_chunks, query)

    # Step 8: Generate final answer using SLM (IMPORTANT FIX)
    final_answer = generate_response(prompt)

    return prompt, final_answer


if __name__ == "__main__":

    pdf_path = "data/sample.pdf"

    query = input("Enter your question: ")

    prompt, answer = build_rag_pipeline(pdf_path, query)

    print("\n================ FINAL PROMPT ================\n")
    print(prompt)

    print("\n================ FINAL ANSWER (SLM RESPONSE) ================\n")
    print(answer)