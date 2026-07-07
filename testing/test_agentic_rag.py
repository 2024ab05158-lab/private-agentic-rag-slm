from application.agentic_pipeline.agentic_rag import AgenticRAG

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


agentic_rag = AgenticRAG(
    store
)


question = (
    "Explain Azure Stack HCI upgrade process"
)


result = agentic_rag.run(
    question
)


print("\nQUESTION")
print("=" * 50)

print(
    result["query"]
)


print("\nANSWER")
print("=" * 50)

print(
    result["answer"]
)


print("\nPLANNER")
print("=" * 50)

print(
    result["plan"]
)


print("\nREFLECTION")
print("=" * 50)

print(
    result["reflection"]
)


print("\nSELF CORRECTION")

print("=" * 50)

print(
    "Retry Count:",
    result["retry_count"]
)

print(
    "Self Corrected:",
    result["self_corrected"]
)