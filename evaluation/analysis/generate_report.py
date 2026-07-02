"""
generate_report.py
-------------------------------------------------------
Reads experiment_log.csv and generates a summary report.

Author : Kathula Deepak
"""

import pandas as pd
from pathlib import Path


LOG_FILE = Path("logs/experiment_log.csv")


def main():

    if not LOG_FILE.exists():

        print("Experiment log not found.")

        return

    df = pd.read_csv(LOG_FILE)

    print("=" * 70)
    print("PRIVATE AGENTIC RAG - EXPERIMENT SUMMARY")
    print("=" * 70)

    print()

    print(f"Total Experiments       : {len(df)}")

    print(f"Models Tested           : {df['Model'].nunique()}")

    print(f"Latest Model            : {df.iloc[-1]['Model']}")

    print()

    print("----------- PERFORMANCE -----------")

    print(f"Average Embedding Time  : {df['EmbeddingTime'].mean():.4f} sec")

    print(f"Average Retrieval Time  : {df['RetrievalTime'].mean():.4f} sec")

    print(f"Average Prompt Time     : {df['PromptTime'].mean():.4f} sec")

    print(f"Average Generation Time : {df['GenerationTime'].mean():.4f} sec")

    print(f"Average Total Time      : {df['TotalTime'].mean():.4f} sec")

    print()

    print("----------- SYSTEM -----------")

    print(f"Average CPU Usage       : {df['CPUPercent'].mean():.2f}%")

    print(f"Average RAM Usage       : {df['RAMPercent'].mean():.2f}%")

    print(f"Average RAM Used        : {df['RAMUsedGB'].mean():.2f} GB")

    print()

    print("----------- RESPONSE -----------")

    print(f"Average Words           : {df['WordCount'].mean():.2f}")

    print(f"Average Characters      : {df['CharacterCount'].mean():.2f}")

    print()

    print("----------- DATASET -----------")

    print(f"Documents Indexed       : {df.iloc[-1]['DocumentCount']}")

    print(f"Total Chunks            : {df.iloc[-1]['TotalChunks']}")

    print()

    print("----------- FASTEST RUN -----------")

    fastest = df.loc[df["TotalTime"].idxmin()]

    print(f"Experiment ID           : {fastest['ExperimentID']}")

    print(f"Model                   : {fastest['Model']}")

    print(f"Time                    : {fastest['TotalTime']:.2f} sec")

    print()

    print("=" * 70)


if __name__ == "__main__":

    main()