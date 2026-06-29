"""
build_index.py
------------------------------------
Creates the FAISS Vector Database
from all PDF documents.

Author : Kathula Deepak
Version : Phase2
"""

import os

from config import (
    DOCUMENT_FOLDER,
    FAISS_INDEX,
    METADATA_FILE
)

from application.ingest.pdf_loader import load_pdf
from application.chunk.text_splitter import split_text
from application.embedd.embedder import get_embeddings
from application.vectordb.faiss_store import VectorStore


INDEX_PATH = FAISS_INDEX
METADATA_PATH = METADATA_FILE


def build_vector_database():

    all_metadata = []
    all_embeddings = []

    print("=" * 60)
    print("Scanning Documents")
    print("=" * 60)

    pdf_count = 0

    for file_name in os.listdir(DOCUMENT_FOLDER):

        if file_name.lower().endswith(".pdf"):

            pdf_count += 1

            file_path = os.path.join(
                DOCUMENT_FOLDER,
                file_name
            )

            print(f"\nReading: {file_name}")

            text = load_pdf(file_path)

            chunks = split_text(text)

            embeddings = get_embeddings(chunks)

            print(f"Chunks Created : {len(chunks)}")

            for i, chunk in enumerate(chunks):

                chunk_info = {

                    "text": chunk,

                    "source": file_name,

                    "chunk_id": i + 1,

                    "total_chunks": len(chunks)

                }

                all_metadata.append(chunk_info)

            all_embeddings.extend(embeddings)

    print("\n")
    print("=" * 60)
    print("Creating FAISS Index")
    print("=" * 60)

    dimension = len(all_embeddings[0])

    store = VectorStore(dimension)

    store.add(
        all_embeddings,
        all_metadata
    )

    print("\nSaving Index...")

    store.save(
        INDEX_PATH,
        METADATA_PATH
    )

    print("\n")
    print("=" * 60)
    print("Index Created Successfully")
    print("=" * 60)

    print(f"PDF Documents : {pdf_count}")
    print(f"Total Chunks  : {len(all_metadata)}")

    print(f"\nIndex File    : {INDEX_PATH}")
    print(f"Metadata File : {METADATA_PATH}")


if __name__ == "__main__":

    build_vector_database()