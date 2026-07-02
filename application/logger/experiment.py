"""
experiment.py

Represents one experiment execution.
"""

from dataclasses import dataclass

from application.rag_pipeline.pipeline_metrics import PipelineMetrics


@dataclass
class Experiment:

    config: dict

    metrics: PipelineMetrics

    question: str

    answer: str

    status: str = "Success"