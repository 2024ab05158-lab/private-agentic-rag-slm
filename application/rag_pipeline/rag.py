from application.embedd.embedder import model

def build_prompt(context_chunks, query):
    """
    Create prompt using retrieved context.
    """
    context = "\n".join(context_chunks)

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer in a clear and simple way:
"""
    return prompt


def get_query_embedding(query):
    return model.encode([query])[0]