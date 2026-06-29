from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Split text into overlapping chunks.

    Args:
        text (str): Input document text.
        chunk_size (int): Size of each chunk.
        overlap (int): Number of overlapping characters.

    Returns:
        list: List of text chunks.
    """

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks