"""
experiment_analyzer.py
-----------------------------------------------------
Loads experiment_log.csv and provides reusable
analysis methods for reports and graphs.

Author : Kathula Deepak
"""

from pathlib import Path

import pandas as pd


class ExperimentAnalyzer:

    def __init__(self):

        self.log_file = Path("logs/experiment_log.csv")

        if not self.log_file.exists():
            raise FileNotFoundError(
                "Experiment log not found."
            )

        self.df = pd.read_csv(
            self.log_file
        )

    # =====================================================
    # Basic Information
    # =====================================================

    def total_experiments(self):

        return len(self.df)

    def latest_model(self):

        return self.df.iloc[-1]["Model"]

    def models_tested(self):

        return self.df["Model"].unique()

    # =====================================================
    # Timing
    # =====================================================

    def average_embedding_time(self):

        return self.df["EmbeddingTime"].mean()

    def average_retrieval_time(self):

        return self.df["RetrievalTime"].mean()

    def average_prompt_time(self):

        return self.df["PromptTime"].mean()

    def average_generation_time(self):

        return self.df["GenerationTime"].mean()

    def average_total_time(self):

        return self.df["TotalTime"].mean()

    # =====================================================
    # CPU / RAM
    # =====================================================

    def average_cpu(self):

        return self.df["CPUPercent"].mean()

    def average_ram(self):

        return self.df["RAMPercent"].mean()

    def average_ram_used(self):

        return self.df["RAMUsedGB"].mean()

    # =====================================================
    # Response
    # =====================================================

    def average_words(self):

        return self.df["WordCount"].mean()

    def average_characters(self):

        return self.df["CharacterCount"].mean()

    # =====================================================
    # Fastest Run
    # =====================================================

    def fastest_run(self):

        return self.df.loc[
            self.df["TotalTime"].idxmin()
        ]

    # =====================================================
    # Slowest Run
    # =====================================================

    def slowest_run(self):

        return self.df.loc[
            self.df["TotalTime"].idxmax()
        ]