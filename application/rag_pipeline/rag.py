from application.embedd.embedder import model
from application.retrieve.retriever import retrieve
from application.slm.slm import generate_response

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


def run_query_pipeline(store, query):
    """
    Executes the complete Core RAG pipeline.

    Returns a dictionary containing all intermediate
    and final outputs for logging and evaluation.
    """

    # Step 1 - Query Embedding
    query_embedding = get_query_embedding(query)

    # Step 2 - Retrieve Context
    context_chunks = retrieve(
        store,
        query_embedding
    )

    # Step 3 - Build Prompt
    prompt = build_prompt(
        context_chunks,
        query
    )

    # Step 4 - Generate Response
    answer = generate_response(
        prompt
    )

    return {

        "query": query,

        "query_embedding": query_embedding,

        "context_chunks": context_chunks,

        "prompt": prompt,

        "answer": answer

    }