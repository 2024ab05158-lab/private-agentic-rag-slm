import csv
import os
from datetime import datetime

from config import (
    EXPERIMENT_LOG,
    ENABLE_LOGGER
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
                    "Model",
                    "Question",
                    "Answer",
                    "AnswerLength",
                    "Status"
                ])

    def log(
        self,
        model,
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
                model,
                question,
                answer,
                len(answer),
                status
            ])