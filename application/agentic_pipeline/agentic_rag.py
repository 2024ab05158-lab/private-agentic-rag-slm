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


from application.agents.planner_agent import PlannerAgent
from application.agents.retrieval_agent import RetrievalAgent
from application.agents.reflection_agent import ReflectionAgent

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


    def run(
            self,
            query
    ):


        retry_count = 0


        # -------------------------------
        # Step 1: Planning
        # -------------------------------

        plan = self.planner.analyze_query(
            query
        )


        # -------------------------------
        # Step 2: Embedding
        # -------------------------------

        query_embedding = get_query_embedding(
            query
        )


        # -------------------------------
        # Step 3: Adaptive Retrieval
        # -------------------------------

        retrieval_result = self.retriever.retrieve_context(
            query_embedding,
            plan
        )


        context_chunks = retrieval_result["context"]


        # -------------------------------
        # Step 4: Prompt Creation
        # -------------------------------

        prompt = build_prompt(
            context_chunks,
            query
        )


        # -------------------------------
        # Step 5: Generate Answer
        # -------------------------------

        answer = generate_response(
            prompt
        )


        # -------------------------------
        # Step 6: Reflection
        # -------------------------------

        reflection = self.reflector.evaluate(
            query,
            answer,
            context_chunks
        )


        # -------------------------------
        # Step 7: Self Correction
        # -------------------------------

        if reflection["retry_required"]:


            retry_count += 1


            plan["recommended_top_k"] += 2


            retrieval_result = self.retriever.retrieve_context(
                query_embedding,
                plan
            )


            context_chunks = retrieval_result["context"]


            prompt = build_prompt(
                context_chunks,
                query
            )


            answer = generate_response(
                prompt
            )


            reflection = self.reflector.evaluate(
                query,
                answer,
                context_chunks
            )


        return {

            "query": query,

            "answer": answer,

            "plan": plan,

            "context": context_chunks,

            "reflection": reflection,

            "retry_count": retry_count,

            "self_corrected": retry_count > 0

        }