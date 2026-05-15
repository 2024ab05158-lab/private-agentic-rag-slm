<<<<<<< HEAD
# Private-agentic-rag-slm
Design and Evaluation of a Self-Correcting, Privacy-Preserving Agentic RAG System using Quantized Small Language Models
=======
\## Phase 1: Project Foundation



\#Objective



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
