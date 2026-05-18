from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(text_list):
    """
    Convert list of text chunks into embeddings.
    """
    embeddings = model.encode(text_list)
    return embeddings