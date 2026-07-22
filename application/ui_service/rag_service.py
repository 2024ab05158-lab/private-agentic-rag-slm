"""
rag_service.py
-------------------------------------------------

Streamlit Service Layer.

Connects UI with existing RAG backend.

Supports:

1. Build FAISS index from uploaded PDF
2. Core RAG execution
3. Agentic RAG execution

"""


import os
from unittest import result

import faiss


from application.ingest.pdf_loader import load_pdf


from application.chunk.text_splitter import split_text


from application.embedd.embedder import (

    get_embeddings

)


from application.vectordb.faiss_store import VectorStore


from application.pipeline.pipeline_manager import PipelineManager

from application.ui_service.index_manager import IndexManager


from config import (

    FAISS_INDEX,

    METADATA_FILE,

    VECTOR_DB_FOLDER

)

DOCUMENT_FOLDER = "data/documents"

index_manager = IndexManager()

os.makedirs(
    DOCUMENT_FOLDER,
    exist_ok=True
)



# -------------------------------------------------
# Build Vector Index
# -------------------------------------------------


def build_single_pdf(

        pdf_path

):


    # ---------------------------------------------
    # Load PDF (Text + OCR supported)
    # ---------------------------------------------


    text = load_pdf(

        pdf_path

    )



    # ---------------------------------------------
    # Create text chunks
    # ---------------------------------------------


    raw_chunks = split_text(

        text

    )



    # ---------------------------------------------
    # Convert chunks into metadata format
    #
    # Required by:
    # - ReRanker
    # - Retrieval Agent
    # - Agentic RAG
    # ---------------------------------------------


    chunks = []



    for index, chunk_text in enumerate(

            raw_chunks

    ):


        chunks.append(

            {

                "text": chunk_text,


                "source": os.path.basename(

                    pdf_path

                ),


                "chunk_id": index

            }

        )



    # ---------------------------------------------
    # Generate embeddings
    #
    # Only send text to embedding model
    # ---------------------------------------------


    embeddings = get_embeddings(

        [

            chunk["text"]

            for chunk in chunks

        ]

    )



    # ---------------------------------------------
    # Initialize FAISS
    # ---------------------------------------------


    store = VectorStore()



    dimension = len(

        embeddings[0]

    )



    store.index = faiss.IndexFlatL2(

        dimension

    )



    # ---------------------------------------------
    # Add vectors + metadata
    # ---------------------------------------------


    store.add(

        embeddings,

        chunks

    )



    # ---------------------------------------------
    # Save FAISS database
    # ---------------------------------------------


    os.makedirs(

        VECTOR_DB_FOLDER,

        exist_ok=True

    )



    store.save(

        FAISS_INDEX,

        METADATA_FILE

    )



    return {


        "status": "success",


        "chunks_created": len(

            chunks

        )


    }


# -------------------------------------------------
# List Documents
# -------------------------------------------------

def list_documents():

    documents = []

    indexed_files = {}

    for doc in index_manager.list_documents():

        indexed_files[doc["filename"]] = doc


    for file in os.listdir(DOCUMENT_FOLDER):

        if file.lower().endswith(".pdf"):

            documents.append(

                {

                    "filename": file,

                    "indexed": file in indexed_files

                }

            )

    return sorted(
        documents,
        key=lambda x: x["filename"]
    )

# -------------------------------------------------
# Upload Document
# -------------------------------------------------

def upload_document(uploaded_file):

    destination = os.path.join(

        DOCUMENT_FOLDER,

        uploaded_file.name

    )

    with open(
        destination,
        "wb"
    ) as file:

        file.write(

            uploaded_file.getbuffer()

        )

    return destination


# -------------------------------------------------
# Delete Selected Documents
# -------------------------------------------------

# -------------------------------------------------
# Delete Selected Documents
# -------------------------------------------------

def delete_documents(files):

    deleted = 0

    for file in files:

        pdf_path = os.path.join(
            DOCUMENT_FOLDER,
            file
        )

        if os.path.exists(pdf_path):

            os.remove(pdf_path)

            index_manager.remove_document(
                file
            )

            deleted += 1

    # -----------------------------------------
    # Synchronize Knowledge Base
    # -----------------------------------------

    remaining_documents = [

        file

        for file in os.listdir(
            DOCUMENT_FOLDER
        )

        if file.lower().endswith(".pdf")

    ]


    if len(remaining_documents) == 0:

        # Remove FAISS database

        if os.path.exists(FAISS_INDEX):

            os.remove(
                FAISS_INDEX
            )

        if os.path.exists(METADATA_FILE):

            os.remove(
                METADATA_FILE
            )

        index_manager.clear_metadata()

        return {

            "deleted": deleted,

            "rebuilt": False,

            "status": "Knowledge Base is now empty."

        }

    else:

        rebuild_result = build_knowledge_base(
            force_rebuild=True
        )

        return {

            "deleted": deleted,

            "rebuilt": True,

            "status": rebuild_result["status"]

        }


# -------------------------------------------------
# Delete All Documents
# -------------------------------------------------

# -------------------------------------------------
# Delete All Documents
# -------------------------------------------------

def delete_all_documents():

    deleted = 0

    for file in os.listdir(DOCUMENT_FOLDER):

        if file.lower().endswith(".pdf"):

            os.remove(

                os.path.join(

                    DOCUMENT_FOLDER,

                    file

                )

            )

            deleted += 1

    index_manager.clear_metadata()

    # Remove FAISS database

    if os.path.exists(FAISS_INDEX):

        os.remove(
            FAISS_INDEX
        )

    if os.path.exists(METADATA_FILE):

        os.remove(
            METADATA_FILE
        )

    return {

        "deleted": deleted,

        "rebuilt": False,

        "status": "Knowledge Base cleared."

    }

# -------------------------------------------------
# Knowledge Base Status
# -------------------------------------------------

def knowledge_base_changed():

    documents = list_documents()

    if len(documents) == 0:

        return False

    for doc in documents:

        if not doc["indexed"]:

            return True

    if len(index_manager.list_documents()) != len(documents):

        return True

    return False

# -------------------------------------------------
# Build Knowledge Base
# -------------------------------------------------

def build_knowledge_base(force_rebuild=False):

    # -------------------------------------------------
    # Skip rebuild only for manual requests
    # -------------------------------------------------

    if not force_rebuild:

        if not knowledge_base_changed():

            return {

                "status": "Knowledge Base already up-to-date.",

                "documents": len(list_documents()),

                "chunks": 0,

                "rebuilt": False

            }


    pdf_files = [

        file

        for file in os.listdir(DOCUMENT_FOLDER)

        if file.lower().endswith(".pdf")

    ]


    if len(pdf_files) == 0:

        return {

            "status": "No PDF documents found.",

            "documents": 0,

            "chunks": 0,

            "rebuilt": False

        }


    all_chunks = []


    chunk_id = 0


    for pdf in pdf_files:


        pdf_path = os.path.join(

            DOCUMENT_FOLDER,

            pdf

        )


        text = load_pdf(

            pdf_path

        )


        raw_chunks = split_text(

            text

        )


        for chunk in raw_chunks:


            all_chunks.append(

                {

                    "text": chunk,

                    "source": pdf,

                    "chunk_id": chunk_id

                }

            )


            chunk_id += 1


    embeddings = get_embeddings(

        [

            chunk["text"]

            for chunk in all_chunks

        ]

    )


    store = VectorStore()


    dimension = len(

        embeddings[0]

    )


    store.index = faiss.IndexFlatL2(

        dimension

    )


    store.add(

        embeddings,

        all_chunks

    )


    store.save(

        FAISS_INDEX,

        METADATA_FILE

    )


    for pdf in pdf_files:


        pdf_path = os.path.join(

            DOCUMENT_FOLDER,

            pdf

        )


        pdf_chunks = sum(

            1

            for chunk in all_chunks

            if chunk["source"] == pdf

        )


        index_manager.update_document(

            filename=pdf,

            last_modified=os.path.getmtime(

                pdf_path

            ),

            chunks=pdf_chunks

        )


    return {

        "status": "Knowledge Base Built Successfully",

        "documents": len(pdf_files),

        "chunks": len(all_chunks),

        "rebuilt": True

    }


# -------------------------------------------------
# Load Existing Vector DB
# -------------------------------------------------


def load_vector_database():


    store = VectorStore()



    store.load(

        FAISS_INDEX,

        METADATA_FILE

    )



    return store





# -------------------------------------------------
# Execute Selected Pipeline
# -------------------------------------------------


def run_rag(

        query,

        mode

):


    store = load_vector_database()



    pipeline = PipelineManager(

        store

    )



    result = pipeline.execute(query, mode)

    return result





# -------------------------------------------------
# Core RAG Wrapper
# -------------------------------------------------


def run_core_rag(

        query

):


    return run_rag(

        query,

        "core"

    )





# -------------------------------------------------
# Agentic RAG Wrapper
# -------------------------------------------------


def run_agentic_rag(

        query

):


    return run_rag(

        query,

        "agentic"

    )