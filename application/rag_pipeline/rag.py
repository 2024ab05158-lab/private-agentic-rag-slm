from application.embedd.embedder import model


def build_prompt(context_chunks, query):

    context = ""

    for item in context_chunks:

        context += f"""
Source Document:
{item['source']}

Content:
{item['text']}

"""

    prompt = f"""
Use the following context to answer the question.

Context:

{context}

Question:

{query}

Answer in a clear and simple way.
If multiple documents are used, combine the information naturally.
"""

    return prompt


def get_query_embedding(query):

    return model.encode([query])[0]