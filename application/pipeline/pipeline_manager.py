"""
pipeline_manager.py
----------------------------------------

Controls execution between:

1. Core RAG
2. Agentic RAG
"""

from application.rag_pipeline.rag import run_query_pipeline
from application.agentic_pipeline.agentic_rag import AgenticRAG
from application.logger.experiment_logger import ExperimentLogger

from config import (
    PROJECT_NAME,
    VERSION,
    SLM_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K
)


class PipelineManager:

    def __init__(self, vector_store):

        self.vector_store = vector_store

        self.agentic_pipeline = AgenticRAG(
            vector_store
        )

        self.logger = ExperimentLogger()

    def _build_config(self, result):

        context = result.get("context_chunks", [])

        return {
            "project": PROJECT_NAME,
            "version": VERSION,
            "model": SLM_MODEL,
            "embedding": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "documents": len(
                set(
                    chunk["source"]
                    for chunk in context
                )
            ),
            "chunks": len(context)
        }

    def execute(
            self,
            query,
            mode
    ):

        if mode.lower() == "core":

            result = run_query_pipeline(
                self.vector_store,
                query
            )

            self.logger.log(
                config=self._build_config(result),
                metrics=result["metrics"],
                question=query,
                answer=result["answer"]
            )

            return result

        elif mode.lower() == "agentic":

            result = self.agentic_pipeline.run(
                query
            )

            self.logger.log(
                config=self._build_config(result),
                metrics=result["metrics"],
                question=query,
                answer=result["answer"],
                agentic_data=result["agentic_metrics"]
            )

            return result

        else:

            raise ValueError(
                "Invalid pipeline mode"
            )