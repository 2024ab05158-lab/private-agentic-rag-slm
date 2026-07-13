from application.ui_service.rag_service import run_agentic_rag


result = run_agentic_rag(
    "what is this document about?"
)


print(result["answer"])


print(result["mode"])