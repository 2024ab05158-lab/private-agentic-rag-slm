import os

from application.ingest.pdf_loader import load_pdf
from application.chunk.text_splitter import split_text

from application.embedd.embedder import get_embeddings
from application.vectordb.faiss_store import VectorStore


DOCUMENT_FOLDER = "data/documents"

INDEX_PATH = "data/vector_db/faiss.index"
METADATA_PATH = "data/vector_db/metadata.pkl"


all_metadata = []
all_embeddings = []

print("Scanning documents...\n")

for file_name in os.listdir(DOCUMENT_FOLDER):

    if file_name.lower().endswith(".pdf"):

        file_path = os.path.join(
            DOCUMENT_FOLDER,
            file_name
        )

        print(f"Reading: {file_name}")

        text = load_pdf(file_path)

        chunks = split_text(text)

        embeddings = get_embeddings(chunks)

        for i, chunk in enumerate(chunks):

            chunk_info = {

                "text": chunk,

                "source": file_name,

                "chunk_id": i,

                "total_chunks": len(chunks)

            }

            all_metadata.append(chunk_info)

        all_embeddings.extend(embeddings)


print("\nCreating FAISS index...")

dimension = len(all_embeddings[0])

store = VectorStore(dimension)

store.add(
    all_embeddings,
    all_metadata
)

print("Saving index...")

store.save(
    INDEX_PATH,
    METADATA_PATH
)

print("\nIndex created successfully!")