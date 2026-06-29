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

        os.makedirs("logs", exist_ok=True)

        if not os.path.exists(EXPERIMENT_LOG):

            with open(
                EXPERIMENT_LOG,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

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
                    "Question",
                    "Answer",
                    "AnswerLength",
                    "Status"
                ])

    def log(
        self,
        config,
        question,
        answer,
        status="Success"
    ):

        if not ENABLE_LOGGER:
            return

        experiment_id = datetime.now().strftime("EXP%Y%m%d%H%M%S")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(
            EXPERIMENT_LOG,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

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

                question,
                answer,

                len(answer),

                status
            ])