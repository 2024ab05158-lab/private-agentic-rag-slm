
"""
generate_graphs.py
------------------------------------------------------------
Dissertation Graph Generator

Reads logs/experiment_log.csv and generates only the
research-relevant graphs. Future Agentic graphs are generated
automatically if the required columns exist.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

LOG_FILE = Path("logs/experiment_log.csv")
OUTPUT_DIR = Path("evaluation/graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_FILE}")
    return pd.read_csv(LOG_FILE)


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)
    print(f"Generated: {filename}")


def response_time(df):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(range(1,len(df)+1), df["TotalTime"], marker="o")
    ax.set_title("Response Time Trend")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Seconds")
    ax.grid(True)
    save(fig,"response_time.png")


def pipeline_breakdown(df):
    cols=["EmbeddingTime","RetrievalTime","PromptTime","GenerationTime"]
    vals=[df[c].mean() for c in cols]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(["Embedding","Retrieval","Prompt","Generation"], vals)
    ax.set_title("Average Pipeline Stage Time")
    ax.set_ylabel("Seconds")
    save(fig,"pipeline_breakdown.png")


def cpu(df):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(range(1,len(df)+1), df["CPUPercent"], marker="o")
    ax.set_title("CPU Usage")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("CPU %")
    ax.grid(True)
    save(fig,"cpu_usage.png")


def ram(df):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(range(1,len(df)+1), df["RAMPercent"], marker="o")
    ax.set_title("RAM Usage")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("RAM %")
    ax.grid(True)
    save(fig,"ram_usage.png")


def model_comparison(df):
    if df["Model"].nunique() < 2:
        print("Skipping model_comparison.png (only one model).")
        return
    g=df.groupby("Model")["TotalTime"].mean().sort_values()
    fig, ax=plt.subplots(figsize=(8,5))
    ax.bar(g.index,g.values)
    ax.set_title("Average Response Time by Model")
    ax.set_ylabel("Seconds")
    save(fig,"model_comparison.png")


def chunk_comparison(df):
    if df["ChunkSize"].nunique() < 2:
        print("Skipping chunk_size_comparison.png (only one chunk size).")
        return
    g=df.groupby("ChunkSize")["TotalTime"].mean()
    fig, ax=plt.subplots(figsize=(8,5))
    ax.plot(g.index.astype(str),g.values,marker="o")
    ax.set_title("Chunk Size vs Average Response Time")
    ax.set_xlabel("Chunk Size")
    ax.set_ylabel("Seconds")
    ax.grid(True)
    save(fig,"chunk_size_comparison.png")


def topk_comparison(df):
    if df["TopK"].nunique() < 2:
        print("Skipping topk_comparison.png (only one Top-K).")
        return
    g=df.groupby("TopK")["TotalTime"].mean()
    fig, ax=plt.subplots(figsize=(8,5))
    ax.plot(g.index.astype(str),g.values,marker="o")
    ax.set_title("Top-K vs Average Response Time")
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Seconds")
    ax.grid(True)
    save(fig,"topk_comparison.png")


def core_vs_agentic(df):
    if "AgentType" not in df.columns:
        print("Skipping core_vs_agentic.png (AgentType column missing).")
        return
    if df["AgentType"].nunique()<2:
        print("Skipping core_vs_agentic.png (only one AgentType).")
        return
    g=df.groupby("AgentType")["TotalTime"].mean()
    fig, ax=plt.subplots(figsize=(8,5))
    ax.bar(g.index,g.values)
    ax.set_title("Core RAG vs Agentic RAG")
    ax.set_ylabel("Seconds")
    save(fig,"core_vs_agentic.png")


def planner_reflection(df):
    needed=["PlannerTime","ReflectionTime"]
    if not all(c in df.columns for c in needed):
        print("Skipping planner_reflection.png (Planner/Reflection columns missing).")
        return
    vals=[df["PlannerTime"].mean(),df["ReflectionTime"].mean()]
    fig, ax=plt.subplots(figsize=(8,5))
    ax.bar(["Planner","Reflection"],vals)
    ax.set_title("Average Planner vs Reflection Time")
    ax.set_ylabel("Seconds")
    save(fig,"planner_reflection.png")


def main():
    df=load_data()
    response_time(df)
    pipeline_breakdown(df)
    cpu(df)
    ram(df)
    model_comparison(df)
    chunk_comparison(df)
    topk_comparison(df)
    core_vs_agentic(df)
    planner_reflection(df)
    print("\nGraph generation complete.")

if __name__=="__main__":
    main()
