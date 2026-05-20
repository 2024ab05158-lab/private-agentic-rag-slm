<<<<<<< HEAD
# Private-agentic-rag-slm
Design and Evaluation of a Self-Correcting, Privacy-Preserving Agentic RAG System using Quantized Small Language Models
=======


## Phase 1: Project Foundation



#Objective



Establish the initial project setup, environment configuration, and repository structure to ensure a clean and scalable base for developing the Private Agentic RAG System using SLMs.



1.1. Project Structure Setup



The project is organized as follows:



private-agentic-rag-slm/

│

├── application/        # Core application modules (RAG pipeline, agents, etc.)

├── data/               # Raw and processed datasets

├── models/             # Stored or downloaded models

├── notebooks/          # Experiments and prototyping

├── docus/              # Project documentation

├── testing/            # Unit tests and validation scripts

├── main.py             # Entry point of the application

├── requirements.txt    # Project dependencies

├── README.md           # Project documentation



1.2. Environment Setup



Step 1: Create Virtual Environment

python -m venv venv



Step 2: Activate Environment



Windows:



venv\\Scripts\\activate



Step 3: Install Dependencies



Install required base libraries:



pip install streamlit fastapi uvicorn faiss-cpu sentence-transformers pymupdf



Step 4: Save Dependencies

pip freeze > requirements.txt





1.3. Git Repository Initialization



Step 1: Initialize Git



git init



Step 2: Stage Files



git add .



Step 3: Commit Initial Setup



git commit -m "Initial project setup with folder structure and environment configuration"



Step 4: Push to GitHub



git branch -M main

git remote add origin <your-github-repo-url>

git push -u origin main





1.4. Development Guidelines



Maintain modular structure inside application/

Keep experiments in notebooks/

Store only lightweight artifacts in GitHub (avoid large models)

Use .gitignore to exclude:

venv/

\_\_pycache\_\_/

large model files

vector DB indexes

>>>>>>> c41e8fb (Initial clean project setup)




# Phase 2 – Private RAG System Implementation

##Overview

Phase 2 focuses on implementing the core Retrieval-Augmented Generation (RAG) pipeline using local Small Language Models (SLMs).
The system is designed to support private and secure document-based question answering without relying on external cloud APIs.

## Modules Implemented

1. Document Ingestion
Extracted text from PDF documents using PyMuPDF.
Supports local document processing.

2. Text Chunking
Implemented overlapping chunk strategy for efficient retrieval.
Improves contextual understanding.

3. Embedding Generation
Used SentenceTransformers (all-MiniLM-L6-v2) to convert text into vector embeddings.

4. Vector Database
Integrated FAISS for efficient similarity search and retrieval.

5. Semantic Retrieval
Implemented Top-K similarity retrieval mechanism.

6. Prompt Construction
Combined retrieved context with user query to create structured prompts.

7. Local SLM Integration
Integrated Ollama for fully local inference.
Used quantized Mistral model for response generation.

#Final Architecture Flow

PDF Document
      ↓
Document Ingestion
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
FAISS Vector Database
      ↓
Top-K Retrieval
      ↓
Prompt Builder
      ↓
Small Language Model (Mistral via Ollama)
      ↓
Final AI Response


# Folder Structure

private-agentic-rag-slm/
│
├── application/
│   ├── ingest/
│   ├── chunk/
│   ├── embedd/
│   ├── vectordb/
│   ├── retrieve/
│   ├── rag_pipeline/
│   └── slm/
│
├── data/
├── models/
├── notebooks/
├── docus/
├── testing/
│
├── main.py
├── requirements.txt
└── README.md

 Technologies Used

	Component				Technology

Programming Language				Python
Document Parsing				PyMuPDF
Embedding Model					SentenceTransformers
Vector Database					FAISS
Local SLM Runtime				Ollama
Small Language Model				Mistral
Environment					Python Virtual Environment

 # How to Run the Project

Step 1 – Activate Virtual Environment
venv\Scripts\activate

Step 2 – Start Local SLM

Open a separate terminal and run:

ollama run mistral

Step 3 – Run the Application
python main.py


##Features Achieved in Phase 2

Fully local RAG pipeline

Private document processing

Retrieval-based context grounding

Quantized SLM integration

Modular project architecture

Local inference without cloud APIs
