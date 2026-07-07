from application.agents.planner_agent import PlannerAgent
from application.agents.retrieval_agent import RetrievalAgent

from application.rag_pipeline.rag import get_query_embedding

from application.vectordb.faiss_store import VectorStore

from config import (
    FAISS_INDEX,
    METADATA_FILE
)


store = VectorStore()


store.load(
    FAISS_INDEX,
    METADATA_FILE
)


planner = PlannerAgent()


retrieval_agent = RetrievalAgent(
    store
)


query = "Explain the Azure Stack HCI upgrade process"


print("\nQuestion:")
print(query)


plan = planner.analyze_query(
    query
)


print("\nPlanner Decision:")
print(plan)


embedding = get_query_embedding(
    query
)


result = retrieval_agent.retrieve_context(
    embedding,
    plan
)


print("\nRetrieval Result:")

print(
    "Strategy:",
    result["strategy"]
)


print(
    "Top-K Used:",
    result["top_k_used"]
)


print(
    "Chunks Retrieved:",
    result["retrieved_chunks"]
)


print("\nSources:")


for item in result["context"]:

    print(
        item["source"],
        "- Chunk",
        item["chunk_id"]
    )