"""
reranker.py
------------------------------------------------

Cross Encoder based re-ranking.

Improves retrieval quality by comparing:

(Query, Retrieved Chunk)

instead of only vector similarity.
"""


from sentence_transformers import CrossEncoder


class ReRanker:


    def __init__(self):


        self.model = CrossEncoder(
            "models/ms-marco-MiniLM-L-6-v2"
        )



    def rerank(
            self,
            query,
            retrieved_chunks,
            top_k
    ):


        if not retrieved_chunks:

            return []



        pairs = []


        for chunk in retrieved_chunks:


            pairs.append(
                (
                    query,
                    chunk["text"]
                )
            )



        scores = self.model.predict(
            pairs
        )



        ranked_results = []


        for chunk, score in zip(
            retrieved_chunks,
            scores
        ):


            chunk[
                "rerank_score"
            ] = float(
                score
            )


            ranked_results.append(
                chunk
            )



        ranked_results.sort(

            key=lambda x:
                x["rerank_score"],

            reverse=True

        )



        return ranked_results[
            :top_k
        ]