import faiss
import numpy as np
import pickle
import os


class VectorStore:

    def __init__(self, dimension=None):

        self.dimension = dimension

        self.chunk_metadata = []

        if dimension is not None:

            self.index = faiss.IndexFlatL2(dimension)

        else:

            self.index = None

    def add(self, embeddings, chunk_metadata):

        self.index.add(
        np.array(embeddings).astype("float32")
        )

        self.chunk_metadata.extend(chunk_metadata)

    def search(self, query_embedding, top_k=3):

        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"),
            top_k
        )

        results = []

        for idx in indices[0]:

            if idx != -1:
                results.append(self.chunk_metadata[idx])

        return results

    def save(self, index_path, metadata_path):

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        faiss.write_index(
            self.index,
            index_path
        )

        with open(metadata_path, "wb") as f:
            pickle.dump(
                self.chunk_metadata,
                f
            )

    def load(self, index_path, metadata_path):

        self.index = faiss.read_index(
            index_path
        )

        with open(metadata_path, "rb") as f:
            self.chunk_metadata = pickle.load(f)