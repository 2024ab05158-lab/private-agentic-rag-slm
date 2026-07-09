"""
relevance_agent.py
-------------------------------------------------

Planner-Aware Relevance Guard Agent.

Responsibilities:

1. Validate retrieved knowledge
2. Prevent hallucination
3. Use Planner intent + ReRanker confidence
4. Allow document exploration queries

"""


class RelevanceAgent:


    def __init__(self):


        # Cross Encoder thresholds

        self.strict_best_score = -1.5


        self.strict_average_score = -3.0



    def validate(
            self,
            retrieved_chunks,
            plan
    ):


        query_type = plan.get(

            "query_type",

            "general"

        )



        # ----------------------------------
        # No retrieved knowledge
        # ----------------------------------


        if not retrieved_chunks:


            return {


                "knowledge_available": False,


                "relevance_score": 0,


                "reason": "No retrieved context"


            }



        scores = []


        for chunk in retrieved_chunks:


            scores.append(

                chunk.get(

                    "rerank_score",

                    0

                )

            )



        best_score = max(

            scores

        )



        average_score = (

            sum(scores)

            /

            len(scores)

        )



        # ----------------------------------
        # Summary Queries
        #
        # Example:
        # describe document
        # summarize file
        # ----------------------------------


        if query_type == "summary":


            return {


                "knowledge_available": True,


                "relevance_score": average_score,


                "reason":

                    "Summary query - retrieved document context available"


            }



        # ----------------------------------
        # Definition Queries
        #
        # Example:
        # what is SET?
        # what is Azure Stack HCI?
        # ----------------------------------


        if query_type == "definition":


            if best_score >= self.strict_average_score:


                return {


                    "knowledge_available": True,


                    "relevance_score": average_score,


                    "reason":

                        "Definition query matched retrieved knowledge"


                }



        # ----------------------------------
        # Procedural / Factual Queries
        #
        # Require stronger evidence
        # ----------------------------------


        if query_type in [

            "procedural",

            "factual"

        ]:


            if (

                best_score >= self.strict_best_score

                or

                average_score >= self.strict_average_score

            ):


                return {


                    "knowledge_available": True,


                    "relevance_score": average_score,


                    "reason":

                        "High confidence retrieved evidence"


                }



        # ----------------------------------
        # General fallback
        # Strict hallucination protection
        # ----------------------------------


        if (

            best_score >= self.strict_best_score

            and

            average_score >= self.strict_average_score

        ):


            return {


                "knowledge_available": True,


                "relevance_score": average_score,


                "reason":

                    "General query matched knowledge base"


            }



        return {


            "knowledge_available": False,


            "relevance_score": average_score,


            "reason":

                "Insufficient document evidence"


        }