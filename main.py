"""
main.py
----------------------------------------------------

Private Self-Correcting Agentic RAG System

Supports:

1. Core RAG
2. Agentic RAG

"""


from application.pipeline.pipeline_manager import PipelineManager

from application.vectordb.faiss_store import VectorStore

from application.logger.experiment_logger import ExperimentLogger


from config import (

    FAISS_INDEX,
    METADATA_FILE,

    PROJECT_NAME,
    VERSION,

    SLM_MODEL,
    EMBEDDING_MODEL,

    CHUNK_SIZE,
    CHUNK_OVERLAP,

    TOP_K

)



def load_vector_database():


    store = VectorStore()


    store.load(

        FAISS_INDEX,

        METADATA_FILE

    )


    return store




def print_system_info():


    print("=" * 60)

    print(
        "Private Self-Correcting Agentic RAG System"
    )

    print("=" * 60)



    print(
        f"SLM Model        : {SLM_MODEL}"
    )


    print(
        f"Embedding Model  : {EMBEDDING_MODEL}"
    )


    print("=" * 60)




def select_pipeline():


    print("\nSelect Pipeline Mode")

    print("=" * 40)


    print(
        "1. Core RAG"
    )


    print(
        "2. Agentic RAG"
    )


    choice = input(
        "\nEnter option: "
    )



    if choice == "1":

        return "core"



    elif choice == "2":

        return "agentic"



    else:


        print(
            "Invalid option. Defaulting to Core RAG"
        )


        return "core"




def display_output(
        result
):


    print("\n")

    print("=" * 60)

    print("ANSWER")

    print("=" * 60)


    print(
        result["answer"]
    )



    if result["mode"] == "Agentic RAG":


        print("\n")

        print("=" * 60)

        print("AGENTIC DETAILS")

        print("=" * 60)



        print(
            "Confidence Score:",
            result["reflection"]["confidence_score"]
        )


        print(
            "Similarity Score:",
            result["reflection"]["similarity_score"]
        )


        print(
            "Retry Count:",
            result["retry_count"]
        )


        print(
            "Self Corrected:",
            result["self_corrected"]
        )




def start_chat():


    print_system_info()



    print(
        "\nLoading Vector Database..."
    )


    store = load_vector_database()



    print(
        "Vector Database Loaded Successfully!"
    )



    mode = select_pipeline()



    pipeline = PipelineManager(
        store
    )



    logger = ExperimentLogger()



    database_info = store.get_database_summary()



    while True:


        query = input(
            "\nEnter your question (or type 'exit'): "
        )



        if query.lower() == "exit":


            print(
                "\nGoodbye!"
            )


            break



        result = pipeline.execute(

            query,

            mode

        )



        display_output(
            result
        )



        agentic_data = None



        if result["mode"] == "Agentic RAG":


            agentic_data = result[
                "agentic_metrics"
            ]



        logger.log(


            config={


                "project": PROJECT_NAME,


                "version": VERSION,


                "model": SLM_MODEL,


                "embedding": EMBEDDING_MODEL,


                "chunk_size": CHUNK_SIZE,


                "overlap": CHUNK_OVERLAP,


                "top_k": TOP_K,


                "documents": database_info[
                    "documents"
                ],


                "chunks": database_info[
                    "chunks"
                ]

            },


            metrics=result[
                "metrics"
            ],


            question=query,


            answer=result[
                "answer"
            ],


            agentic_data=agentic_data

        )




if __name__ == "__main__":


    start_chat()