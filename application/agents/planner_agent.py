"""
planner_agent.py
-------------------------------------------------

Planner Agent for Agentic RAG Pipeline.

Responsibilities:

1. Understand user query intent
2. Select retrieval strategy
3. Decide dynamic Top-K value

Used by:
- Retrieval Agent
- Relevance Agent
"""


class PlannerAgent:


    def __init__(self):

        pass



    def analyze_query(
            self,
            query
    ):


        query_lower = query.lower()



        plan = {


            "query_type": "general",


            "retrieval_strategy": "standard",


            "recommended_top_k": 2


        }



        # --------------------------------------
        # Document Summary / Exploration Queries
        # --------------------------------------


        if any(

            phrase in query_lower

            for phrase in [

                "summarize",

                "summary",

                "overview",

                "describe",

                "document about",

                "file about",

                "uploaded file",

                "uploaded document",

                "explain the document",

                "explain this document"

            ]

        ):


            plan[

                "query_type"

            ] = "summary"



            plan[

                "retrieval_strategy"

            ] = "broad_context"



            plan[

                "recommended_top_k"

            ] = 5



        # --------------------------------------
        # Definition / Concept Queries
        # --------------------------------------


        elif any(

            phrase in query_lower

            for phrase in [

                "what is",

                "what are",

                "define",

                "meaning of",

                "tell me about",

                "explain"

            ]

        ):


            plan[

                "query_type"

            ] = "definition"



            plan[

                "retrieval_strategy"

            ] = "concept_lookup"



            plan[

                "recommended_top_k"

            ] = 4



        # --------------------------------------
        # Procedural Queries
        # --------------------------------------


        elif any(

            word in query_lower

            for word in [

                "how",

                "steps",

                "procedure",

                "process",

                "setup",

                "configure",

                "install",

                "upgrade",

                "deploy"

            ]

        ):


            plan[

                "query_type"

            ] = "procedural"



            plan[

                "retrieval_strategy"

            ] = "multi_step"



            plan[

                "recommended_top_k"

            ] = 3



        # --------------------------------------
        # Exact Lookup Queries
        # --------------------------------------


        elif any(

            word in query_lower

            for word in [

                "command",

                "parameter",

                "value",

                "version",

                "number",

                "date"

            ]

        ):


            plan[

                "query_type"

            ] = "factual"



            plan[

                "retrieval_strategy"

            ] = "precision"



            plan[

                "recommended_top_k"

            ] = 2



        return plan