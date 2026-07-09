"""
retrieval_agent.py
-------------------------------------------------

Retrieval Agent for Agentic RAG.

Responsible for:

1. Reading Planner decisions
2. Adaptive retrieval
3. FAISS candidate retrieval
4. Cross Encoder Re-Ranking
5. Re-ranking metrics collection
"""


import time


from application.retrieve.retriever import retrieve

from application.reranker.reranker import ReRanker



class RetrievalAgent:


    def __init__(
            self,
            vector_store
    ):


        self.vector_store = vector_store


        self.reranker = ReRanker()



    def retrieve_context(
            self,
            query_embedding,
            plan,
            query=None
    ):


        final_top_k = plan.get(
            "recommended_top_k",
            2
        )


        strategy = plan.get(
            "retrieval_strategy",
            "standard"
        )



        # Fetch more candidates for re-ranking

        candidate_k = final_top_k * 3



        candidate_chunks = retrieve(

            self.vector_store,

            query_embedding,

            candidate_k

        )



        reranking_time = 0.0

        average_score = 0.0



        if query is not None:


            start = time.perf_counter()


            context_chunks = self.reranker.rerank(

                query,

                candidate_chunks,

                final_top_k

            )


            reranking_time = (

                time.perf_counter()

                -

                start

            )


            reranking_enabled = True



            scores = [

                chunk.get(
                    "rerank_score",
                    0
                )

                for chunk in context_chunks

            ]



            if scores:


                average_score = (

                    sum(scores)

                    /

                    len(scores)

                )



        else:


            context_chunks = candidate_chunks[

                :final_top_k

            ]


            reranking_enabled = False




        return {


            "context": context_chunks,


            "top_k_used": final_top_k,


            "candidate_chunks": len(
                candidate_chunks
            ),


            "retrieved_chunks": len(
                context_chunks
            ),


            "strategy": strategy,


            "reranking_enabled": reranking_enabled,


            "reranking_time": reranking_time,


            "average_rerank_score": average_score


        }