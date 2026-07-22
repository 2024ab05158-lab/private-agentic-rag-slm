"""
===============================================================================
Dissertation Graph Generator
===============================================================================

Project:
    Design and Evaluation of a Self-Correcting,
    Privacy-Preserving Agentic Retrieval-Augmented
    Generation System using Quantized Small Language Models

Description
-----------
Reads logs/experiment_log.csv and generates dissertation-quality
comparison graphs between

    • Core RAG
    • Agentic RAG

Graphs are automatically skipped when the required columns
are unavailable.

Author  : Deepak Kathula
Version : 2.0
===============================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ==============================================================================
# Paths
# ==============================================================================

LOG_FILE = Path("logs/experiment_log.csv")

OUTPUT_DIR = Path("evaluation/graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Figure Settings
# ==============================================================================

DPI = 300

FIG_SIZE = (8,5)

TITLE_SIZE = 14

LABEL_SIZE = 11

TICK_SIZE = 10

BAR_WIDTH = 0.55


plt.rcParams["figure.dpi"] = DPI
plt.rcParams["axes.titlesize"] = TITLE_SIZE
plt.rcParams["axes.labelsize"] = LABEL_SIZE
plt.rcParams["xtick.labelsize"] = TICK_SIZE
plt.rcParams["ytick.labelsize"] = TICK_SIZE
plt.rcParams["font.size"] = 10


# ==============================================================================
# Dissertation Color Palette
# ==============================================================================

CORE_COLOR = "#4E79A7"

AGENTIC_COLOR = "#F28E2B"

GRID_COLOR = "#D9D9D9"

HIST_COLOR = "#76B7B2"

SUCCESS_COLOR = "#59A14F"

WARNING_COLOR = "#E15759"


# ==============================================================================
# Column Mapping
#
# Change ONLY here if the logger changes in future.
# ==============================================================================

COLUMNS = {

    "PIPELINE"          : "PipelineMode",

    "TOTAL_TIME"        : "TotalTime",

    "CPU"               : "CPUPercent",

    "RAM"               : "RAMUsedGB",

    "RAM_PERCENT"       : "RAMPercent",

    "EMBEDDING"         : "EmbeddingTime",

    "RETRIEVAL"         : "RetrievalTime",

    "PROMPT"            : "PromptTime",

    "GENERATION"        : "GenerationTime",

    "SIMILARITY"        : "SimilarityScore",

    "RETRIEVAL_SCORE"   : "RetrievalScore",

    "RELEVANCE_SCORE"   : "RelevanceScore",

    "COMPLETENESS"      : "CompletenessScore",

    "UNCERTAINTY"       : "UncertaintyScore",

    "CONFIDENCE"        : "ConfidenceScore",

    "WORD_COUNT"        : "WordCount",

    "CHUNKS"            : "RetrievedChunkCount",

    "TOPK"              : "DynamicTopK",

    "RERANK_TIME"       : "ReRankingTime",

    "RERANK_SCORE"      : "AverageReRankScore",

    "RETRY"             : "RetryCount",

    "SELF_CORRECTED"    : "SelfCorrected"

}


# ==============================================================================
# Load CSV
# ==============================================================================

def load_data():

    if not LOG_FILE.exists():

        raise FileNotFoundError(

            f"\nExperiment log not found:\n{LOG_FILE}"

        )

    df = pd.read_csv(LOG_FILE)

    print("-"*60)
    print(f"Loaded {len(df)} experiments")
    print(f"Detected {len(df.columns)} columns")
    print("-"*60)

    return df


# ==============================================================================
# Normalize Pipeline Names
# ==============================================================================

def normalize_pipeline(df):

    pipeline = COLUMNS["PIPELINE"]

    if pipeline not in df.columns:

        raise Exception(

            "PipelineMode column missing."

        )

    df[pipeline] = (

        df[pipeline]

        .astype(str)

        .str.strip()

        .str.lower()

    )

    mapping = {

        "core":"Core RAG",

        "corerag":"Core RAG",

        "core rag":"Core RAG",

        "agentic":"Agentic RAG",

        "agenticrag":"Agentic RAG",

        "agentic rag":"Agentic RAG"

    }

    df[pipeline] = (

        df[pipeline]

        .replace(mapping)

    )

    return df


# ==============================================================================
# Utility
# ==============================================================================

def column_exists(df,key):

    column = COLUMNS[key]

    return column in df.columns


def available(df,key):

    if not column_exists(df,key):

        return False

    return df[COLUMNS[key]].notna().sum() > 0


# ==============================================================================
# Save Figure
# ==============================================================================

def save(fig,filename):

    fig.tight_layout()

    fig.savefig(

        OUTPUT_DIR / filename,

        dpi=DPI,

        bbox_inches="tight"

    )

    plt.close(fig)

    print(f"Generated : {filename}")


# ==============================================================================
# Grouped Mean
# ==============================================================================

def grouped_mean(df,key):

    if not available(df,key):

        return None

    column = COLUMNS[key]

    pipeline = COLUMNS["PIPELINE"]

    grouped = (

        df

        .groupby(pipeline)[column]

        .mean()

        .round(3)

    )

    if len(grouped) < 2:

        return None

    return grouped


# ==============================================================================
# Common Comparison Bar Chart
# ==============================================================================

def comparison_bar(

        df,

        key,

        title,

        ylabel,

        filename

):

    grouped = grouped_mean(df,key)

    if grouped is None:

        print(f"Skipping {filename}")

        return

    labels = grouped.index.tolist()

    values = grouped.values

    colors = []

    for label in labels:

        if "core" in label.lower():

            colors.append(CORE_COLOR)

        else:

            colors.append(AGENTIC_COLOR)

    fig,ax = plt.subplots(figsize=FIG_SIZE)

    bars = ax.bar(

        labels,

        values,

        width=BAR_WIDTH,

        color=colors,

        edgecolor="black"

    )

    for bar in bars:

        value = bar.get_height()

        ax.text(

            bar.get_x()+bar.get_width()/2,

            value,

            f"{value:.2f}",

            ha="center",

            va="bottom",

            fontsize=10,

            fontweight="bold"

        )

    ax.set_title(title)

    ax.set_ylabel(ylabel)

    ax.grid(

        axis="y",

        linestyle="--",

        alpha=0.35

    )

    save(fig,filename)


# ==============================================================================
# Common Histogram
# ==============================================================================

def histogram(

        df,

        key,

        title,

        filename,

        bins=10

):

    if not available(df,key):

        print(f"Skipping {filename}")

        return

    pipeline = COLUMNS["PIPELINE"]

    column = COLUMNS[key]

    agentic = df[

        df[pipeline]=="Agentic RAG"

    ]

    if len(agentic)==0:

        print(f"Skipping {filename}")

        return

    fig,ax = plt.subplots(figsize=FIG_SIZE)

    ax.hist(

        agentic[column].dropna(),

        bins=bins,

        color=HIST_COLOR,

        edgecolor="black"

    )

    ax.set_title(title)

    ax.set_xlabel(column)

    ax.set_ylabel("Frequency")

    ax.grid(

        linestyle="--",

        alpha=0.30

    )

    save(fig,filename)

    # ==============================================================================
# Performance Comparison Graphs
# ==============================================================================

def response_time(df):

    comparison_bar(
        df,
        "TOTAL_TIME",
        "Average Response Time",
        "Time (seconds)",
        "response_time.png"
    )


def cpu_usage(df):

    comparison_bar(
        df,
        "CPU",
        "Average CPU Usage",
        "CPU (%)",
        "cpu_usage.png"
    )


def ram_usage(df):

    if available(df, "RAM"):

        comparison_bar(
            df,
            "RAM",
            "Average RAM Usage",
            "RAM (GB)",
            "ram_usage.png"
        )

    elif available(df, "RAM_PERCENT"):

        comparison_bar(
            df,
            "RAM_PERCENT",
            "Average RAM Usage",
            "RAM (%)",
            "ram_usage.png"
        )

    else:

        print("Skipping ram_usage.png")


# ==============================================================================
# Pipeline Breakdown
# ==============================================================================

def pipeline_breakdown(df):

    pipeline = COLUMNS["PIPELINE"]

    stage_keys = [

        "EMBEDDING",
        "RETRIEVAL",
        "PROMPT",
        "GENERATION"

    ]

    stage_columns = []

    labels = []

    for key in stage_keys:

        if available(df, key):

            stage_columns.append(COLUMNS[key])

            labels.append(key.replace("_", " ").title())

    if len(stage_columns) < 2:

        print("Skipping pipeline_breakdown.png")

        return

    grouped = (

        df

        .groupby(pipeline)[stage_columns]

        .mean()

    )

    fig, ax = plt.subplots(figsize=(10,5))

    grouped.T.plot(

        kind="bar",

        ax=ax,

        width=0.75,

        color=[CORE_COLOR, AGENTIC_COLOR]

    )

    ax.set_title("Average Pipeline Stage Execution Time")

    ax.set_ylabel("Time (seconds)")

    ax.set_xlabel("Pipeline Stage")

    ax.grid(

        axis="y",

        linestyle="--",

        alpha=0.30

    )

    ax.legend(title="Pipeline")

    save(fig, "pipeline_breakdown.png")


# ==============================================================================
# Retrieval Quality
# ==============================================================================

def similarity_score(df):

    comparison_bar(

        df,

        "SIMILARITY",

        "Average Similarity Score",

        "Similarity Score",

        "similarity_score.png"

    )


def retrieval_score(df):

    comparison_bar(

        df,

        "RETRIEVAL_SCORE",

        "Average Retrieval Score",

        "Retrieval Score",

        "retrieval_score.png"

    )


def relevance_score(df):

    comparison_bar(

        df,

        "RELEVANCE_SCORE",

        "Average Relevance Score",

        "Relevance Score",

        "relevance_score.png"

    )


def retrieved_chunks(df):

    comparison_bar(

        df,

        "CHUNKS",

        "Average Retrieved Chunks",

        "Chunk Count",

        "retrieved_chunks.png"

    )


def reranking_time(df):

    comparison_bar(

        df,

        "RERANK_TIME",

        "Average Re-ranking Time",

        "Time (seconds)",

        "reranking_time.png"

    )


def reranking_score(df):

    comparison_bar(

        df,

        "RERANK_SCORE",

        "Average Re-ranking Score",

        "Average Score",

        "reranking_score.png"

    )


# ==============================================================================
# Response Quality
# ==============================================================================

def confidence_score(df):

    comparison_bar(

        df,

        "CONFIDENCE",

        "Average Confidence Score",

        "Confidence",

        "confidence_score.png"

    )


def completeness_score(df):

    comparison_bar(

        df,

        "COMPLETENESS",

        "Average Completeness Score",

        "Completeness",

        "completeness_score.png"

    )


def uncertainty_score(df):

    comparison_bar(

        df,

        "UNCERTAINTY",

        "Average Uncertainty Score",

        "Uncertainty",

        "uncertainty_score.png"

    )


def word_count(df):

    comparison_bar(

        df,

        "WORD_COUNT",

        "Average Response Length",

        "Words",

        "word_count.png"

    )


# ==============================================================================
# Agentic RAG Behaviour
# ==============================================================================

def retry_distribution(df):

    histogram(

        df,

        "RETRY",

        "Retry Count Distribution",

        "retry_distribution.png",

        bins=8

    )


def confidence_distribution(df):

    histogram(

        df,

        "CONFIDENCE",

        "Confidence Score Distribution",

        "confidence_distribution.png"

    )


def dynamic_topk(df):

    if available(df, "TOPK"):

        histogram(

            df,

            "TOPK",

            "Dynamic Top-K Distribution",

            "dynamic_topk_distribution.png",

            bins=8

        )

    else:

        print("Skipping dynamic_topk_distribution.png")


# ==============================================================================
# Self Correction Pie Chart
# ==============================================================================

def self_correction(df):

    if not available(df, "SELF_CORRECTED"):

        print("Skipping self_correction.png")

        return

    column = COLUMNS["SELF_CORRECTED"]

    counts = (

        df[column]

        .dropna()

        .astype(str)

        .value_counts()

    )

    if counts.empty:

        print("Skipping self_correction.png")

        return

    fig, ax = plt.subplots(figsize=(6,6))

    ax.pie(

        counts.values,

        labels=counts.index,

        autopct="%1.1f%%",

        startangle=90,

        colors=[SUCCESS_COLOR, WARNING_COLOR]

    )

    ax.set_title("Self-Correction Rate")

    save(fig, "self_correction.png")

    # ==============================================================================
# Generate All Graphs
# ==============================================================================

def generate_all_graphs(df):

    print("\n" + "=" * 70)
    print("Generating Dissertation Evaluation Graphs")
    print("=" * 70)

    graph_functions = [

        # ---------------------------------------------------------
        # Performance
        # ---------------------------------------------------------

        response_time,
        cpu_usage,
        ram_usage,
        pipeline_breakdown,

        # ---------------------------------------------------------
        # Retrieval Quality
        # ---------------------------------------------------------

        similarity_score,
        retrieval_score,
        relevance_score,
        retrieved_chunks,
        reranking_time,
        reranking_score,

        # ---------------------------------------------------------
        # Response Quality
        # ---------------------------------------------------------

        confidence_score,
        completeness_score,
        uncertainty_score,
        word_count,

        # ---------------------------------------------------------
        # Agentic Behaviour
        # ---------------------------------------------------------

        retry_distribution,
        confidence_distribution,
        dynamic_topk,
        self_correction

    ]

    success = 0
    failed = 0

    for graph in graph_functions:

        try:

            graph(df)

            success += 1

        except Exception as ex:

            failed += 1

            print(f"[ERROR] {graph.__name__}")

            print(ex)

            print("-" * 60)

    print("\n" + "=" * 70)

    print(f"Graphs Generated : {success}")

    print(f"Graphs Failed    : {failed}")

    print(f"Output Folder    : {OUTPUT_DIR}")

    print("=" * 70)


# ==============================================================================
# Dataset Summary
# ==============================================================================

def dataset_summary(df):

    print("\nDataset Summary")
    print("-" * 60)

    pipeline = COLUMNS["PIPELINE"]

    if pipeline in df.columns:

        counts = (

            df[pipeline]

            .value_counts()

        )

        print(counts)

    print("\nColumns Available")

    print("-" * 60)

    for column in df.columns:

        print(column)

    print("-" * 60)


# ==============================================================================
# Main
# ==============================================================================

def main():

    print("\n")

    print("=" * 70)

    print("Dissertation Graph Generator")

    print("=" * 70)

    df = load_data()

    df = normalize_pipeline(df)

    dataset_summary(df)

    generate_all_graphs(df)

    print("\nFinished Successfully.")

    print(f"\nGraphs saved to:\n{OUTPUT_DIR.resolve()}")

    print("=" * 70)


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\nProcess cancelled by user.")

    except FileNotFoundError as ex:

        print(ex)

    except Exception as ex:

        print("\nUnexpected Error")

        print(ex)