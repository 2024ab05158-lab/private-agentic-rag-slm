"""
retrieval_agent.py
-------------------------------------------------
Retrieval Agent for Agentic RAG.

Responsible for:
1. Reading Planner decisions
2. Applying adaptive retrieval strategy
3. Fetching context from FAISS
"""


from application.retrieve.retriever import retrieve


class RetrievalAgent:


    def __init__(
            self,
            vector_store
    ):

        self.vector_store = vector_store


    def retrieve_context(
            self,
            query_embedding,
            plan
    ):

        """
        Retrieves context based on Planner recommendation.
        """

        top_k = plan.get(
            "recommended_top_k",
            2
        )


        strategy = plan.get(
            "retrieval_strategy",
            "standard"
        )


        context_chunks = retrieve(
            self.vector_store,
            query_embedding,
            top_k
        )


        return {

            "context": context_chunks,

            "top_k_used": top_k,

            "strategy": strategy,

            "retrieved_chunks": len(context_chunks)

        }