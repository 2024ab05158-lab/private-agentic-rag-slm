"""
pipeline_metrics.py

Stores all metrics collected during one
RAG pipeline execution.
"""

from dataclasses import dataclass, field


@dataclass
class PipelineMetrics:


    # ===================================================
    # Pipeline Timing
    # ===================================================


    embedding_time: float = 0.0


    retrieval_time: float = 0.0


    prompt_time: float = 0.0


    generation_time: float = 0.0


    total_time: float = 0.0



    # ===================================================
    # Re-Ranking Metrics
    # ===================================================


    reranking_enabled: bool = False


    reranking_time: float = 0.0


    average_rerank_score: float = 0.0

    # ===================================================
    # Relevance Guard Metrics
    # ===================================================


    knowledge_available: bool = True


    relevance_score: float = 0.0


    generation_skipped: bool = False

    # ===================================================
    # CPU / Memory
    # ===================================================


    cpu_percent: float = 0.0


    ram_percent: float = 0.0


    ram_used_gb: float = 0.0


    ram_available_gb: float = 0.0


    logical_cpu_count: int = 0


    physical_cpu_count: int = 0



    # ===================================================
    # Retrieval
    # ===================================================


    retrieved_document_count: int = 0


    retrieved_chunk_count: int = 0


    retrieved_documents: list = field(
        default_factory=list
    )


    retrieved_chunk_ids: list = field(
        default_factory=list
    )



    # ===================================================
    # Response
    # ===================================================


    answer_length: int = 0


    word_count: int = 0


    character_count: int = 0