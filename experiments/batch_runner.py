import os
import sys
import time
import pandas as pd

# -------------------------------------------------
# Project Path
# -------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# -------------------------------------------------
# Import Existing Pipelines
# -------------------------------------------------

from application.ui_service.rag_service import (
    run_core_rag,
    run_agentic_rag
)

# -------------------------------------------------
# Configuration
# -------------------------------------------------

QUESTION_FILE = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "questions.csv"
)

DELAY_SECONDS = 2

# -------------------------------------------------
# Load Questions
# -------------------------------------------------

questions = pd.read_csv(QUESTION_FILE)

total = len(questions)

print("=" * 60)
print("BATCH EXPERIMENT STARTED")
print("=" * 60)

# -------------------------------------------------
# CORE RAG
# -------------------------------------------------

print("\nRunning Core RAG...\n")

for index, row in questions.iterrows():

    question = row["Question"]

    print(f"[CORE] {index+1}/{total}")

    try:

        run_core_rag(question)

        print("✓ Success")

    except Exception as e:

        print(f"✗ Failed : {e}")

    time.sleep(DELAY_SECONDS)

# -------------------------------------------------
# AGENTIC RAG
# -------------------------------------------------

print("\nRunning Agentic RAG...\n")

for index, row in questions.iterrows():

    question = row["Question"]

    print(f"[AGENTIC] {index+1}/{total}")

    try:

        run_agentic_rag(question)

        print("✓ Success")

    except Exception as e:

        print(f"✗ Failed : {e}")

    time.sleep(DELAY_SECONDS)

print("\n")
print("=" * 60)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 60)
print("Experiment log updated successfully.")