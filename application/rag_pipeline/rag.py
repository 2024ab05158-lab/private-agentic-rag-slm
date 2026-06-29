import time
from tracemalloc import start

from application.embedd.embedder import model
from application.retrieve.retriever import retrieve
from application.slm.slm import generate_response
from application.rag_pipeline.pipeline_metrics import PipelineMetrics

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

    metrics = PipelineMetrics()

    total_start = time.perf_counter()

    # Step 1 - Query Embedding

    start = time.perf_counter()

    query_embedding = get_query_embedding(query)

    metrics.embedding_time = (
        time.perf_counter() - start
    )

    # Step 2 - Retrieve Context
    start = time.perf_counter()

    context_chunks = retrieve(
    store,
    query_embedding
    )

    metrics.retrieval_time = (
        time.perf_counter() - start
    )

    # Step 3 - Build Prompt
    start = time.perf_counter()

    prompt = build_prompt(
    context_chunks,
    query
    )

    metrics.prompt_time = (
        time.perf_counter() - start
    )

    # Step 4 - Generate Response
    start = time.perf_counter()

    answer = generate_response(
    prompt
    )

    metrics.generation_time = (
        time.perf_counter() - start
    )

    metrics.total_time = (
        time.perf_counter() - total_start
    )
    return {

        "query": query,

        "query_embedding": query_embedding,

        "context_chunks": context_chunks,

        "prompt": prompt,

        "answer": answer,

         "metrics": metrics

    }