"""
agentic_rag.py
-------------------------------------------------

Self-Correcting Agentic RAG Pipeline.

Components:
1. Planner Agent
2. Adaptive Retrieval Agent
3. Local SLM Generator
4. Reflection Agent
5. Self Correction Loop
"""


import time
import psutil


from application.agents.planner_agent import PlannerAgent
from application.agents.retrieval_agent import RetrievalAgent
from application.agents.reflection_agent import ReflectionAgent
from application.agents.relevance_agent import RelevanceAgent
from application.utils.process_monitor import ProcessMonitor


from application.rag_pipeline.pipeline_metrics import PipelineMetrics


from application.rag_pipeline.rag import (
    get_query_embedding,
    build_prompt
)


from application.slm.slm import generate_response



class AgenticRAG:


    def __init__(
            self,
            vector_store
    ):


        self.planner = PlannerAgent()


        self.retriever = RetrievalAgent(
            vector_store
        )


        self.reflector = ReflectionAgent()

        self.relevance_agent = RelevanceAgent()



    def run(
            self,
            query
    ):


        retry_count = 0


        metrics = PipelineMetrics()


        total_start = time.perf_counter()



        # ----------------------------------
        # Step 1 - Planner Agent
        # ----------------------------------


        plan = self.planner.analyze_query(
            query
        )



        # ----------------------------------
        # Step 2 - Query Embedding
        # ----------------------------------


        start = time.perf_counter()


        query_embedding = get_query_embedding(
            query
        )


        metrics.embedding_time = (
            time.perf_counter()
            -
            start
        )



        # ----------------------------------
        # Step 3 - Adaptive Retrieval
        # ----------------------------------


        start = time.perf_counter()


        retrieval_result = self.retriever.retrieve_context(
            query_embedding,
            plan,
            query

        )


        metrics.retrieval_time = (
            time.perf_counter()
            -
            start
        )


        context_chunks = retrieval_result[
            "context"
        ]

        # ----------------------------------
        # ReRanking Metrics
        # ----------------------------------


        metrics.reranking_enabled = retrieval_result[
            "reranking_enabled"
        ]


        metrics.reranking_time = retrieval_result[
            "reranking_time"
        ]


        metrics.average_rerank_score = retrieval_result[
            "average_rerank_score"
        ]

        # ----------------------------------
        # Step 4 - Relevance Validation
        # ----------------------------------


        relevance = self.relevance_agent.validate(

            context_chunks,

            plan

        )


        metrics.knowledge_available = relevance[

            "knowledge_available"

        ]


        metrics.relevance_score = relevance[

            "relevance_score"

        ]



        if not relevance[

            "knowledge_available"

        ]:


            metrics.generation_skipped = True


            return {


                "mode": "Agentic RAG",


                "query": query,


                "answer":

                    "The uploaded knowledge base does not contain sufficient information to answer this question.",


                "context_chunks": context_chunks,


                "plan": plan,


                "reflection": {


                    "confidence_score": 0,


                    "similarity_score": 0,


                    "retrieval_score": 0,


                    "completeness_score": 0,


                    "uncertainty_score": 1


                },


                "retry_count": 0,


                "self_corrected": False,


                "metrics": metrics,


                "agentic_metrics": {


                    "pipeline_mode": "Agentic RAG",


                    "query_type": plan[
                        "query_type"
                    ],


                    "retrieval_strategy": plan[
                        "retrieval_strategy"
                    ],


                    "dynamic_top_k": plan[
                        "recommended_top_k"
                    ],


                    "confidence_score": 0,


                    "similarity_score": 0,


                    "retrieval_score": 0,


                    "completeness_score": 0,


                    "uncertainty_score": 1,


                    "retry_count": 0,


                    "self_corrected": False


                }

            }

        # ----------------------------------
        # Step 4 - Prompt Creation
        # ----------------------------------


        start = time.perf_counter()


        prompt = build_prompt(
            context_chunks,
            query
        )


        metrics.prompt_time = (
            time.perf_counter()
            -
            start
        )



        # ----------------------------------
        # Step 5 - Generate Response
        # ----------------------------------


        start = time.perf_counter()


        answer = generate_response(
            prompt
        )


        metrics.generation_time = (
            time.perf_counter()
            -
            start
        )



        # ----------------------------------
        # Step 6 - Reflection Agent
        # ----------------------------------


        reflection = self.reflector.evaluate(
            query,
            answer,
            context_chunks
        )



        # ----------------------------------
        # Step 7 - Self Correction
        # ----------------------------------


        if reflection[
            "retry_required"
        ]:


            retry_count += 1



            plan[
                "recommended_top_k"
            ] += 2



            retrieval_result = self.retriever.retrieve_context(
                query_embedding,
                plan,
                query
            )


            context_chunks = retrieval_result[
                "context"
            ]

            metrics.reranking_enabled = retrieval_result[
                "reranking_enabled"
            ]


            metrics.reranking_time += retrieval_result[
                "reranking_time"
            ]


            metrics.average_rerank_score = retrieval_result[
                "average_rerank_score"
            ]


            prompt = build_prompt(
                context_chunks,
                query
            )



            start = time.perf_counter()


            answer = generate_response(
                prompt
            )


            metrics.generation_time += (
                time.perf_counter()
                -
                start
            )



            reflection = self.reflector.evaluate(
                query,
                answer,
                context_chunks
            )



        # ----------------------------------
        # Final Performance Metrics
        # ----------------------------------


        metrics.total_time = (
            time.perf_counter()
            -
            total_start
        )



        memory = psutil.virtual_memory()



        #metrics.cpu_percent = psutil.cpu_percent(interval=0.1)

        # ----------------------------------
        # CPU Metrics
        # ----------------------------------

        metrics.cpu_percent = ProcessMonitor.get_peak_cpu()



        metrics.ram_percent = memory.percent



        metrics.ram_used_gb = round(
            memory.used / (1024 ** 3),
            2
        )



        metrics.ram_available_gb = round(
            memory.available / (1024 ** 3),
            2
        )



        metrics.retrieved_documents = list(
            set(
                [
                    item["source"]
                    for item in context_chunks
                ]
            )
        )



        metrics.retrieved_chunk_count = len(
            context_chunks
        )



        metrics.retrieved_chunk_ids = [

            item["chunk_id"]

            for item in context_chunks

            if "chunk_id" in item

        ]



        metrics.answer_length = len(
            answer
        )



        metrics.word_count = len(
            answer.split()
        )



        metrics.character_count = len(
            answer
        )



        # ----------------------------------
        # Final Agentic Output
        # ----------------------------------


        return {


            "mode": "Agentic RAG",


            "query": query,


            "answer": answer,


            "context_chunks": context_chunks,


            "plan": plan,


            "reflection": reflection,


            "retry_count": retry_count,


            "self_corrected": retry_count > 0,


            "metrics": metrics,



            "agentic_metrics": {


                "pipeline_mode": "Agentic RAG",


                "query_type": plan[
                    "query_type"
                ],


                "retrieval_strategy": plan[
                    "retrieval_strategy"
                ],


                "dynamic_top_k": plan[
                    "recommended_top_k"
                ],



                "confidence_score": reflection[
                    "confidence_score"
                ],


                "similarity_score": reflection[
                    "similarity_score"
                ],


                "retrieval_score": reflection[
                    "retrieval_score"
                ],


                "completeness_score": reflection[
                    "completeness_score"
                ],


                "uncertainty_score": reflection[
                    "uncertainty_score"
                ],



                "retry_count": retry_count,


                "self_corrected": retry_count > 0


            }


        }