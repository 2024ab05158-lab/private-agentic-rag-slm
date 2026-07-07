import csv
import os

from datetime import datetime


from config import (
    EXPERIMENT_LOG,
    ENABLE_LOGGER,
    PROJECT_NAME,
    VERSION
)


class ExperimentLogger:


    def __init__(self):


        if not ENABLE_LOGGER:

            return


        os.makedirs(
            "logs",
            exist_ok=True
        )


        if not os.path.exists(
            EXPERIMENT_LOG
        ):


            with open(
                EXPERIMENT_LOG,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:


                writer = csv.writer(
                    file
                )


                writer.writerow([


                    "ExperimentID",
                    "Timestamp",


                    "Project",
                    "Version",


                    "Model",
                    "EmbeddingModel",


                    "ChunkSize",
                    "ChunkOverlap",
                    "TopK",


                    "DocumentCount",
                    "TotalChunks",


                    "EmbeddingTime",
                    "RetrievalTime",
                    "PromptTime",
                    "GenerationTime",
                    "TotalTime",


                    "CPUPercent",
                    "RAMPercent",
                    "RAMUsedGB",
                    "RAMAvailableGB",


                    "RetrievedDocuments",
                    "RetrievedChunkCount",
                    "RetrievedChunkIDs",


                    "AnswerLength",
                    "WordCount",
                    "CharacterCount",


                    "Question",
                    "Answer",


                    # Agentic RAG Metrics

                    "PipelineMode",


                    "QueryType",
                    "RetrievalStrategy",
                    "DynamicTopK",


                    "ConfidenceScore",
                    "SimilarityScore",
                    "RetrievalScore",
                    "CompletenessScore",
                    "UncertaintyScore",


                    "RetryCount",
                    "SelfCorrected",


                    "Status"

                ])


    def log(
            self,
            config,
            metrics,
            question,
            answer,
            agentic_data=None,
            status="Success"
    ):


        if not ENABLE_LOGGER:

            return



        if agentic_data is None:


            agentic_data = {


                "pipeline_mode": "Core RAG",


                "query_type": "NA",


                "retrieval_strategy": "NA",


                "dynamic_top_k": "NA",


                "confidence_score": "NA",


                "similarity_score": "NA",


                "retrieval_score": "NA",


                "completeness_score": "NA",


                "uncertainty_score": "NA",


                "retry_count": 0,


                "self_corrected": False

            }



        experiment_id = datetime.now().strftime(
            "EXP%Y%m%d%H%M%S"
        )


        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )



        with open(
            EXPERIMENT_LOG,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:


            writer = csv.writer(
                file
            )


            writer.writerow([


                experiment_id,

                timestamp,


                config["project"],

                config["version"],


                config["model"],

                config["embedding"],


                config["chunk_size"],

                config["overlap"],

                config["top_k"],


                config["documents"],

                config["chunks"],



                round(
                    metrics.embedding_time,
                    4
                ),


                round(
                    metrics.retrieval_time,
                    4
                ),


                round(
                    metrics.prompt_time,
                    4
                ),


                round(
                    metrics.generation_time,
                    4
                ),


                round(
                    metrics.total_time,
                    4
                ),



                metrics.cpu_percent,

                metrics.ram_percent,

                metrics.ram_used_gb,

                metrics.ram_available_gb,



                ", ".join(
                    metrics.retrieved_documents
                ),


                metrics.retrieved_chunk_count,


                ", ".join(
                    map(
                        str,
                        metrics.retrieved_chunk_ids
                    )
                ),



                metrics.answer_length,

                metrics.word_count,

                metrics.character_count,



                question,

                answer,



                # Agentic Values


                agentic_data[
                    "pipeline_mode"
                ],


                agentic_data[
                    "query_type"
                ],


                agentic_data[
                    "retrieval_strategy"
                ],


                agentic_data[
                    "dynamic_top_k"
                ],



                agentic_data[
                    "confidence_score"
                ],


                agentic_data[
                    "similarity_score"
                ],


                agentic_data[
                    "retrieval_score"
                ],


                agentic_data[
                    "completeness_score"
                ],


                agentic_data[
                    "uncertainty_score"
                ],



                agentic_data[
                    "retry_count"
                ],


                agentic_data[
                    "self_corrected"
                ],



                status

            ])