"""
embedder.py
---------------------------------------
Local embedding model loader.

Supports privacy-preserving offline RAG.
"""


import os

from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "all-MiniLM-L6-v2"
)


model = SentenceTransformer(
    MODEL_PATH
)


def get_embeddings(texts):

    embeddings = model.encode(
        texts
    )

    return embeddings