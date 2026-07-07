from application.pipeline.pipeline_manager import PipelineManager

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


manager = PipelineManager(
    store
)


question = (
    "Explain Azure Stack HCI upgrade process"
)


mode = "agentic"


result = manager.execute(
    question,
    mode
)


print("\nMODE")
print("=" * 50)

print(
    result["mode"]
)


print("\nANSWER")
print("=" * 50)

print(
    result["answer"]
)


if mode == "agentic":

    print("\nREFLECTION")

    print(
        result["reflection"]
    )