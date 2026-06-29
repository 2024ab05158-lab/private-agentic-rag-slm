def split_text(text, chunk_size=700, overlap=150):
    """
    Split long text into smaller overlapping chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks