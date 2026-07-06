"""
planner_agent.py
-------------------------------------------------
Planner Agent for Agentic RAG Pipeline.

Analyzes the user query and decides retrieval
strategy before execution.
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
        # Procedural Questions
        # --------------------------------------

        if any(
            word in query_lower
            for word in [
                "how",
                "steps",
                "procedure",
                "process",
                "setup",
                "configure"
            ]
        ):

            plan["query_type"] = "procedural"

            plan["retrieval_strategy"] = "multi_step"

            plan["recommended_top_k"] = 3


        # --------------------------------------
        # Summary Questions
        # --------------------------------------

        elif any(
            word in query_lower
            for word in [
                "summarize",
                "summary",
                "explain",
                "overview"
            ]
        ):

            plan["query_type"] = "summary"

            plan["retrieval_strategy"] = "broad_context"

            plan["recommended_top_k"] = 4


        # --------------------------------------
        # Exact Lookup Questions
        # --------------------------------------

        elif any(
            word in query_lower
            for word in [
                "command",
                "parameter",
                "value",
                "version"
            ]
        ):

            plan["query_type"] = "factual"

            plan["retrieval_strategy"] = "precision"

            plan["recommended_top_k"] = 2


        return plan