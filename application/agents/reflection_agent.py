"""
reflection_agent.py
-------------------------------------------------

Advanced Reflection Agent for Agentic RAG.

Responsibilities:
1. Evaluate generated answer quality
2. Compare answer with retrieved context
3. Calculate confidence score
4. Decide if self-correction is required

Confidence Formula:

35% - Answer Context Similarity
25% - Retrieval Quality
25% - Answer Completeness
15% - Uncertainty Detection
"""


from sklearn.metrics.pairwise import cosine_similarity

from application.embedd.embedder import get_embeddings


class ReflectionAgent:


    def __init__(
            self,
            confidence_threshold=0.75
    ):

        self.confidence_threshold = confidence_threshold


    def calculate_similarity(
            self,
            answer,
            context_chunks
    ):

        if len(context_chunks) == 0:

            return 0


        context_text = ""


        for chunk in context_chunks:

            context_text += chunk["text"] + " "


        embeddings = get_embeddings(
            [
                answer,
                context_text
            ]
        )


        similarity = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]


        return float(similarity)


    def calculate_retrieval_score(
            self,
            context_chunks
    ):

        retrieved_count = len(
            context_chunks
        )


        if retrieved_count >= 3:

            return 1.0


        elif retrieved_count == 2:

            return 0.8


        elif retrieved_count == 1:

            return 0.5


        else:

            return 0


    def calculate_completeness(
            self,
            answer
    ):

        word_count = len(
            answer.split()
        )


        if word_count >= 100:

            return 1.0


        elif word_count >= 50:

            return 0.8


        elif word_count >= 20:

            return 0.5


        else:

            return 0.2


    def calculate_uncertainty(
            self,
            answer
    ):

        uncertainty_terms = [

            "i don't know",

            "not sure",

            "cannot determine",

            "no information",

            "not available"

        ]


        answer_lower = answer.lower()


        for term in uncertainty_terms:

            if term in answer_lower:

                return 0


        return 1.0


    def evaluate(
            self,
            query,
            answer,
            context_chunks
    ):


        similarity_score = self.calculate_similarity(
            answer,
            context_chunks
        )


        retrieval_score = self.calculate_retrieval_score(
            context_chunks
        )


        completeness_score = self.calculate_completeness(
            answer
        )


        uncertainty_score = self.calculate_uncertainty(
            answer
        )


        confidence_score = (

            (0.35 * similarity_score)

            +

            (0.25 * retrieval_score)

            +

            (0.25 * completeness_score)

            +

            (0.15 * uncertainty_score)

        )


        confidence_score = round(
            confidence_score,
            2
        )


        retry_required = (
            confidence_score <
            self.confidence_threshold
        )


        return {

            "confidence_score": confidence_score,

            "similarity_score": round(
                similarity_score,
                2
            ),

            "retrieval_score": retrieval_score,

            "completeness_score": completeness_score,

            "uncertainty_score": uncertainty_score,

            "retry_required": retry_required

        }