"""
rag.py
----------------------------------------------------
Core RAG Pipeline

Responsible for:
1. Query Embedding
2. Context Retrieval
3. Prompt Construction
4. SLM Response Generation
5. Pipeline Performance Metrics
6. System Metrics
7. Retrieval Metrics
8. Response Metrics
"""

import time

import psutil

from application.embedd.embedder import model
from application.retrieve.retriever import retrieve
from application.slm.slm import generate_response
from application.rag_pipeline.pipeline_metrics import PipelineMetrics
from application.utils.process_monitor import ProcessMonitor


def build_prompt(context_chunks, query):
    """
    Build the final prompt sent to the Small Language Model.
    """

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
    """
    Convert the user query into an embedding vector.
    """

    return model.encode([query])[0]


def run_query_pipeline(store, query):
    """
    Executes the complete Core RAG pipeline.

    Returns:
        Dictionary containing:
        - query
        - query_embedding
        - retrieved chunks
        - prompt
        - answer
        - metrics
    """

    metrics = PipelineMetrics()

    total_start = time.perf_counter()

    # ==========================================================
    # STEP 1 - QUERY EMBEDDING
    # ==========================================================

    start = time.perf_counter()

    query_embedding = get_query_embedding(query)

    metrics.embedding_time = (
        time.perf_counter() - start
    )

    # ==========================================================
    # STEP 2 - CONTEXT RETRIEVAL
    # ==========================================================

    start = time.perf_counter()

    context_chunks = retrieve(
        store,
        query_embedding
    )

    metrics.retrieval_time = (
        time.perf_counter() - start
    )

    # ==========================================================
    # STEP 3 - PROMPT CONSTRUCTION
    # ==========================================================

    start = time.perf_counter()

    prompt = build_prompt(
        context_chunks,
        query
    )

    metrics.prompt_time = (
        time.perf_counter() - start
    )

    # ==========================================================
    # STEP 4 - SLM GENERATION
    # ==========================================================

    start = time.perf_counter()

    answer = generate_response(
        prompt
    )

    metrics.generation_time = (
        time.perf_counter() - start
    )

    # ==========================================================
    # TOTAL EXECUTION TIME
    # ==========================================================

    metrics.total_time = (
        time.perf_counter() - total_start
    )

    # ==========================================================
    # CPU METRICS
    # ==========================================================

    metrics.cpu_percent = ProcessMonitor.get_peak_cpu()


    # ==========================================================
    # MEMORY METRICS
    # ==========================================================

    memory = psutil.virtual_memory()

    metrics.ram_percent = memory.percent

    metrics.ram_used_gb = round(
        memory.used / (1024 ** 3),
        2
    )

    metrics.ram_available_gb = round(
        memory.available / (1024 ** 3),
        2
    )

    metrics.logical_cpu_count = psutil.cpu_count()

    metrics.physical_cpu_count = psutil.cpu_count(
        logical=False
    )

    # ==========================================================
    # RETRIEVAL METRICS
    # ==========================================================

    metrics.retrieved_chunk_count = len(
        context_chunks
    )

    documents = []
    chunk_ids = []

    for chunk in context_chunks:

        documents.append(
            chunk["source"]
        )

        chunk_ids.append(
            chunk["chunk_id"]
        )

    metrics.retrieved_documents = sorted(
        list(set(documents))
    )

    metrics.retrieved_document_count = len(
        metrics.retrieved_documents
    )

    metrics.retrieved_chunk_ids = chunk_ids

    # ==========================================================
    # RESPONSE METRICS
    # ==========================================================

    metrics.answer_length = len(answer)

    metrics.word_count = len(
        answer.split()
    )

    metrics.character_count = len(answer)

    # ==========================================================
    # RETURN PIPELINE RESULT
    # ==========================================================

    return {

        "mode": "Core RAG",

        "query": query,

        "query_embedding": query_embedding,

        "context_chunks": context_chunks,

        "prompt": prompt,

        "answer": answer,

        "metrics": metrics

    }