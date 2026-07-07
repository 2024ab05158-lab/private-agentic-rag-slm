"""
pipeline_manager.py
----------------------------------------

Controls execution between:

1. Core RAG
2. Agentic RAG
"""


from application.rag_pipeline.rag import run_query_pipeline

from application.agentic_pipeline.agentic_rag import AgenticRAG


class PipelineManager:


    def __init__(
            self,
            vector_store
    ):

        self.vector_store = vector_store


        self.agentic_pipeline = AgenticRAG(
            vector_store
        )


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


            return result



        elif mode.lower() == "agentic":


            result = self.agentic_pipeline.run(
                query
            )


            return result



        else:


            raise ValueError(
                "Invalid pipeline mode"
            )