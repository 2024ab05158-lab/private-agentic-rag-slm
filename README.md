# Private Agentic RAG SLM

> **Design and Evaluation of a Self-Correcting, Privacy-Preserving Agentic RAG System using Quantized Small Language Models**

---

## About this Document

This document presents the complete implementation journey of the project from the initial project foundation to the final Streamlit-based Self-Correcting Agentic RAG system. It covers architecture, design decisions, module implementation, evaluation methodology, and future-ready enhancements.

---
# Private-agentic-rag-slm

Design and Evaluation of a Self-Correcting, Privacy-Preserving Agentic RAG System using Quantized Small Language Models



## Why this project?

Large Language Models have demonstrated impressive reasoning capabilities, but they present several challenges when used for enterprise document question answering:

• Sensitive documents cannot always be sent to cloud services.
• LLMs frequently hallucinate when answering domain-specific questions.
• Most traditional RAG systems generate answers directly without validating retrieval quality.
• Existing systems rarely explain how confident they are in their responses.

To address these challenges, this project proposes a fully local, privacy-preserving Agentic Retrieval-Augmented Generation framework capable of evaluating its own retrieval quality and improving answers through self-correction.

## Development Methodology

The project followed an incremental modular development methodology.

Rather than implementing the complete Agentic RAG architecture at once, each subsystem was independently designed, tested, and integrated.

Below is the List of Phases we have implemented it in. We will see each phase in detail what work has been done.

Phase 1:
Foundation

↓

Phase 2:
Core RAG

↓

Phase 3:
Agentic RAG

↓

Phase 4:
Evaluation

↓

Phase 5:
User Interface


## Phase 1: Project Foundation



### Objective


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


## Why this Folder Structure?

Folder					Why it was created
application/				Contains all reusable project logic following modular software engineering principles.
data/					Stores uploaded documents, vector indexes, and metadata.
testing/				Allows independent validation of each module before pipeline integration.
notebooks/				Used for experimentation before production implementation.
docus/					Stores architecture diagrams, reports, and documentation.

## Why Python?

Python was selected because of its mature ecosystem for Natural Language Processing, Machine Learning, vector databases, and rapid prototyping. Libraries such as SentenceTransformers, FAISS, Streamlit, and PyMuPDF integrate naturally into Python-based workflows, making it an ideal language for building an end-to-end local RAG system.

## Why Virtual Environment?

Dependency isolation
Reproducibility
Avoiding version conflicts
Easier deployment

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



# Phase 2 – Private RAG System Implementation

## Overview

Phase 2 focuses on implementing the core Retrieval-Augmented Generation (RAG) pipeline using local Small Language Models (SLMs).
The system is designed to support private and secure document-based question answering without relying on external cloud APIs.


##Objective:

Why Core RAG was implemented before Agentic RAG?

A traditional Retrieval-Augmented Generation pipeline was implemented first to establish a reliable baseline before introducing Agentic reasoning. This allowed the retrieval quality, embedding strategy, vector search, and local language model inference to be validated independently.

## Folder Evolution

For example:

application/

↓

ingest/

↓

pdf_loader.py

Purpose

Responsibilities

Input

Output

Interaction



## Design Decisions :

##application/chunk/

Purpose

The chunk module divides large documents into smaller, semantically meaningful text segments that can be embedded and retrieved efficiently.

Responsibilities:

Split extracted document text into manageable chunks.
Preserve contextual continuity between chunks.
Improve retrieval accuracy.
Prepare text for embedding generation.
Input
Plain text extracted from PDF documents.
Output
A list of overlapping text chunks.
Interaction
pdf_loader.py
        ↓
text_splitter.py
        ↓
embedder.py

## Why Chunking?

Large Language Models and embedding models have token limitations. Processing an entire document as one block is inefficient and reduces retrieval precision.

Chunking enables:

Semantic indexing
Faster retrieval
Reduced memory usage
Better context localization

Instead of searching an entire document, the system retrieves only the relevant portions.

## Why Overlap?

Without overlap:

Chunk 1
"The Azure Stack HCI cluster"

Chunk 2
"supports Network ATC."

A query about Azure Stack HCI Network ATC might fail because the sentence is split.

With overlap:

Chunk 1
"...Azure Stack HCI cluster..."

Chunk 2
"...Azure Stack HCI cluster supports Network ATC..."

Important context is preserved.

Benefits:

Better semantic continuity
Improved retrieval quality
Reduced information loss at chunk boundaries
Why Not Fixed Paragraphs?

Paragraph lengths vary significantly.

Some paragraphs contain:

Multiple topics
Very little information
Hundreds of words

Using paragraph boundaries alone may produce:

Chunks that are too large
Chunks that are too small
Inconsistent embeddings

##Fixed-size overlapping chunks provide:

Consistent embedding quality
Predictable retrieval performance
Better semantic granularity
application/embedd/
Purpose

Convert text chunks into dense semantic vector representations.

Responsibilities
Load the embedding model.
Generate embeddings for document chunks.
Generate embeddings for user queries.
Input
List of text chunks.
User query.
Output
Numerical embedding vectors.
Interaction
text_splitter.py
        ↓
embedder.py
        ↓
faiss_store.py

## Why MiniLM?

all-MiniLM-L6-v2 offers an excellent balance between accuracy and computational efficiency.

Advantages:

Lightweight
Fast inference
Good semantic understanding
Runs locally
Suitable for CPU execution

This makes it well suited for a privacy-preserving local RAG system.

## Why SentenceTransformers?

SentenceTransformers provides:

Pre-trained embedding models
High-quality semantic embeddings
Easy integration
Active research support

It has become one of the standard libraries for dense retrieval applications.

Why 384 Dimensions?

MiniLM produces 384-dimensional vectors.

Advantages:

Lower storage requirements
Faster similarity search
Reduced memory consumption
Comparable retrieval accuracy to much larger embeddings for many document retrieval tasks

Higher-dimensional embeddings can improve some tasks but generally require more computation and storage.

##application/vectordb/
Purpose

Store embeddings efficiently and perform fast similarity search.

Responsibilities
Create the FAISS index.
Store document embeddings.
Store chunk metadata.
Retrieve nearest neighbors.
Input
Embedding vectors.
Chunk metadata.
Output
Top-K most similar document chunks.
Interaction
embedder.py
        ↓
FAISS
        ↓
retriever.py

## Why FAISS?

FAISS was selected because it:

Runs completely offline
Is optimized for high-dimensional vectors
Supports efficient similarity search
Is widely used in research and production
Why Not Chroma?

Chroma offers:

Metadata filtering
Rich persistence
Simple API

However:

Introduces additional abstraction
Requires more dependencies
Was unnecessary for the project's relatively small local knowledge base

FAISS provided a simpler and more efficient solution.

Why Not Pinecone?

Pinecone is a cloud-hosted vector database.

This project emphasizes:

Privacy
Offline execution
No external dependencies

Using Pinecone would violate the project's privacy-preserving objective.

Why a Local Vector Database?

A local vector database ensures:

No data leaves the machine.
Faster retrieval with no network latency.
Full control over sensitive enterprise documents.
Offline availability.
application/retrieve/
Purpose

Retrieve the document chunks that are most semantically similar to the user's query.

Responsibilities
Embed the query.
Search the FAISS index.
Return candidate chunks.
Input
User query.
Output
Top-K retrieved chunks.
Interaction
User Query
      ↓
Embedding
      ↓
FAISS Search
      ↓
Retrieved Chunks


## Why Semantic Retrieval?

Keyword search cannot recognize semantic similarity.

Example:

Document:

"Hyper-V cluster deployment"

User:

"How do I configure virtualization hosts?"

Keyword matching may fail.

Semantic retrieval understands conceptual similarity, retrieving relevant content even when exact words differ.

Why Top-K Retrieval?

Returning a single chunk may omit important context.

Returning all chunks introduces irrelevant information.

Top-K retrieval balances relevance and context.

Why K = 3?

K=3 was selected as the default because it:

Provides sufficient contextual information.
Reduces irrelevant content.
Maintains concise prompts.
Keeps inference latency low.

Later phases introduce dynamic Top-K, where the Planner Agent adjusts K according to query type.

#application/rag_pipeline/
Purpose

Construct the final prompt that combines retrieved evidence with the user's question.

Responsibilities
Merge retrieved chunks.
Create structured prompts.
Prepare input for the SLM.
Input
Retrieved context.
User question.
Output
Final prompt.
Interaction
Retrieved Chunks
        +
User Question
        ↓
Prompt Builder
        ↓
SLM


## Why Prompt Engineering?

The quality of generated answers depends heavily on prompt construction.

Well-designed prompts:

Reduce hallucinations.
Encourage grounded answers.
Keep responses focused on retrieved evidence.
Improve consistency.
How Is Context Constructed?

The retrieved chunks are concatenated into a structured prompt.

Typical format:

Context:
<Retrieved Chunk 1>

<Retrieved Chunk 2>

<Retrieved Chunk 3>

Question:
<User Question>

Answer only using the provided context.

This guides the SLM to rely on retrieved evidence instead of prior knowledge.

#application/slm/

Purpose

Generate answers locally using a Small Language Model.

Responsibilities
Send prompts to Ollama.
Receive generated responses.
Return answers to the pipeline.
Input
Prompt generated by the RAG pipeline.
Output
Natural language answer.
Interaction
Prompt
      ↓
Ollama
      ↓
Mistral
      ↓
Answer


## Why Ollama?

Ollama provides:

Local model execution
Simple model management
Cross-platform support
No API keys
No internet dependency

It enables fully offline inference while maintaining an easy development workflow.

## Why Mistral?

Mistral was selected because it offers:

Strong reasoning performance
Good instruction-following capability
Efficient inference on consumer hardware
Broad community adoption
High-quality responses relative to its size

These characteristics make it suitable for local RAG applications.

## Why a Quantized Model?

Quantization reduces model size by representing weights with lower-precision numerical formats.

Benefits include:

Reduced memory usage
Faster inference
Lower hardware requirements
Ability to run locally on commodity CPUs

For this project, quantization allows the system to remain fully local while delivering practical performance without requiring enterprise-grade GPUs.

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

## Final Architecture Flow

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


## Folder Structure

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



## Module-wise Implementation


1. Document Ingestion (application/ingest/pdf_loader.py)
Brief Description

The Document Ingestion module is responsible for extracting textual information from PDF documents before they enter the RAG pipeline. It supports both digitally generated PDFs and scanned image-based PDFs by combining PyMuPDF with an automatic OCR fallback mechanism. This ensures that enterprise documents, manuals, and scanned reports can all be processed without requiring manual intervention. The module produces a unified text representation that is passed to the chunking stage.

Purpose

Extract readable text from PDF documents.

Responsibilities
Load PDF documents
Extract text page-by-page
Detect low-text documents
Trigger OCR when required
Return complete extracted text
Input
PDF file path
Output
Complete document text
Interaction
PDF
      ↓
pdf_loader.py
      ↓
text_splitter.py
2. Text Chunking (application/chunk/text_splitter.py)
Brief Description

The Text Chunking module divides large documents into smaller overlapping text segments suitable for semantic embedding and retrieval. Instead of indexing an entire document as one unit, the system creates uniformly sized chunks with configurable overlap to preserve contextual continuity. This approach improves retrieval precision while ensuring important information is not lost across chunk boundaries.

Purpose

Divide documents into searchable chunks.

Responsibilities
Split extracted text
Maintain overlap
Preserve context
Generate retrieval-ready chunks
Input
Extracted document text
Output
List of text chunks
Interaction
PDF Text
      ↓
Chunking
      ↓
Embedding Generation
3. Embedding Generation (application/embedd/embedder.py)
Brief Description

The Embedding Generation module converts each text chunk into a dense semantic vector using the Sentence Transformers implementation of the all-MiniLM-L6-v2 model. The model is loaded locally to maintain complete privacy and eliminate dependence on cloud services. These embeddings form the semantic representation stored within the FAISS vector database for efficient similarity search.

Purpose

Generate semantic vector representations.

Responsibilities
Load embedding model
Generate chunk embeddings
Generate query embeddings
Support offline embedding generation
Input
Text chunks
User query
Output
Embedding vectors
Interaction
Text Chunks
      ↓
MiniLM
      ↓
Embedding Vectors
4. Vector Database (application/vectordb/faiss_store.py)
Brief Description

The Vector Database module stores semantic embeddings using Facebook AI Similarity Search (FAISS). Alongside vector storage, the module maintains metadata containing document names and chunk identifiers, enabling retrieved vectors to be mapped back to their original source. It also provides methods for persistence, loading, searching, and reporting knowledge base statistics.

Purpose

Store and retrieve embedding vectors efficiently.

Responsibilities
Store embeddings
Store metadata
Perform similarity search
Save/load vector database
Maintain database statistics
Input
Embeddings
Chunk metadata
Output
Retrieved document chunks
Interaction
Embeddings
      ↓
FAISS
      ↓
Retriever
5. Semantic Retrieval (application/retrieve/retriever.py)
Brief Description

The Retrieval module performs semantic similarity search over the FAISS vector database. After the user query is converted into an embedding, the retriever identifies the Top-K most relevant document chunks based on vector similarity. The module abstracts retrieval logic from the vector database implementation, allowing retrieval strategies to evolve independently in later phases.

Purpose

Retrieve relevant document chunks.

Responsibilities
Receive query embedding
Execute Top-K search
Return relevant chunks
Support configurable retrieval
Input
Query embedding
Output
Top-K document chunks
Interaction
Query Embedding
      ↓
Retriever
      ↓
Relevant Chunks
6. Prompt Construction (application/rag_pipeline/rag.py)
Brief Description

The Prompt Construction module combines retrieved document chunks with the user's question to create a structured prompt for the Small Language Model. Retrieved context is organized with source information before the user query is appended, encouraging the language model to produce answers grounded in the provided evidence. The module also records execution metrics for each stage of the Core RAG pipeline.

Purpose

Prepare structured prompts for the language model.

Responsibilities
Combine retrieved context
Build prompt
Execute Core RAG pipeline
Collect execution metrics
Return pipeline results
Input
Retrieved chunks
User question
Output
Final prompt
Pipeline metrics
Interaction
Retrieved Chunks
        +
User Query
        ↓
Prompt Builder
        ↓
SLM
7. Local SLM Integration (application/slm/slm.py)
Brief Description

The Local SLM Integration module communicates with Ollama through its REST API to perform fully local inference using the quantized Mistral Small Language Model. By executing inference entirely on the user's machine, the module ensures privacy, removes dependence on external APIs, and allows document-based question answering without transmitting sensitive information outside the local environment.

Purpose

Generate responses using a local Small Language Model.

Responsibilities
Connect to Ollama
Send prompts
Receive generated responses
Handle inference failures
Input
Prompt
Output
Generated answer
Interaction
Prompt
      ↓
Ollama
      ↓
Mistral
      ↓
Answer
8. Main Application (main.py)
Brief Description

The Main Application serves as the entry point of the Core RAG system. It initializes the vector database, loads system configuration, accepts user queries, invokes the complete RAG pipeline, records experimental metrics through the Experiment Logger, and displays generated answers along with the source documents used during retrieval. This module orchestrates all major components of the Core RAG architecture.

Purpose

Coordinate the execution of the complete Core RAG pipeline.

Responsibilities
Load vector database
Display system configuration
Accept user queries
Execute Core RAG pipeline
Log experiments
Display answers and sources
Input
User query
Output
AI-generated answer
Source documents
Performance metrics
Interaction
User
      ↓
main.py
      ↓
Core RAG Pipeline
      ↓
Logger
      ↓
Answer


##Features Achieved in Phase 2

Fully local RAG pipeline

Private document processing

Retrieval-based context grounding

Quantized SLM integration

Modular project architecture

Local inference without cloud APIs




# Phase 3 – Agentic RAG



## Overview

After successfully implementing the Core Retrieval-Augmented Generation (RAG) pipeline in Phase 2, the project was extended into a Self-Correcting Agentic RAG architecture. The primary motivation behind this transition was to address the inherent limitations of traditional RAG systems, such as hallucinations, fixed retrieval strategies, lack of confidence estimation, and the inability to verify or refine generated responses.

Unlike conventional RAG systems, where retrieved context is immediately passed to the language model for response generation, the Agentic RAG architecture introduces multiple intelligent software agents. Each agent performs a specialized task within the reasoning pipeline, enabling the system to plan retrieval strategies, validate retrieved evidence, evaluate answer quality, and perform self-correction whenever necessary.

The resulting architecture provides a more reliable, explainable, and adaptive question-answering system while maintaining complete privacy through local execution.


##Motivation for Agentic RAG

Although the Core RAG pipeline successfully retrieved relevant document chunks and generated responses, several practical limitations remained:

Fixed Top-K retrieval often returned either insufficient or excessive context.
The system had no mechanism to determine whether retrieved information was actually relevant to the user's question.
Responses were generated without evaluating their confidence or completeness.
Hallucinations could still occur when retrieval quality was poor.
The pipeline lacked any mechanism for retrying or refining low-quality answers.

To overcome these challenges, the project adopted an Agentic architecture, where independent agents collaborate to improve retrieval quality and answer reliability.


##Agentic RAG Architecture

The Agentic pipeline extends the traditional RAG workflow by introducing multiple reasoning stages before and after response generation.

User Query
      ↓
Planner Agent
      ↓
Retrieval Agent
      ↓
Cross Encoder ReRanker
      ↓
Relevance Agent
      ↓
Prompt Construction
      ↓
Mistral SLM
      ↓
Reflection Agent
      ↓
Confidence Evaluation
      ↓
Self-Correction (if required)
      ↓
Final Response


Components Introduced

The following intelligent agents were developed during this phase.

3.1 Planner Agent
Brief Description

The Planner Agent acts as the decision-making component of the Agentic RAG pipeline. Rather than treating every query identically, the Planner Agent analyzes the user's question to determine the most appropriate retrieval strategy. It classifies the query type, selects the retrieval approach, and dynamically adjusts retrieval parameters before any document search is performed.

This enables the system to adapt its behaviour based on the complexity and intent of the user's request.

Purpose

Analyze the incoming query before retrieval.

Responsibilities
Query classification
Dynamic Top-K selection
Retrieval strategy selection
Procedural vs factual query detection
Pipeline planning
Input
User query
Output
Query type
Retrieval strategy
Dynamic Top-K value
Interaction
User Query
      ↓
Planner Agent
      ↓
Retrieval Agent
3.2 Retrieval Agent
Brief Description

The Retrieval Agent performs semantic document retrieval based on the strategy selected by the Planner Agent. It generates the query embedding, retrieves candidate chunks from the FAISS vector database, and prepares them for semantic reranking. The agent abstracts retrieval operations from the rest of the pipeline, making retrieval configurable and extensible.

Purpose

Retrieve candidate evidence from the knowledge base.

Responsibilities
Generate query embeddings
Perform FAISS similarity search
Collect candidate chunks
Forward results for reranking
Input
User query
Retrieval strategy
Output
Candidate document chunks
Interaction
Planner Agent
      ↓
Retrieval Agent
      ↓
ReRanker
3.3 Cross Encoder ReRanker
Brief Description

Traditional embedding similarity provides approximate semantic retrieval but may still rank less relevant chunks higher than more informative ones. The Cross Encoder ReRanker addresses this limitation by evaluating each query-document pair individually and assigning a semantic relevance score. Retrieved chunks are reordered according to these scores before being passed to the language model.

This significantly improves retrieval precision and overall answer quality.

Purpose

Improve retrieval accuracy through semantic reranking.

Responsibilities
Evaluate query-document pairs
Compute semantic relevance scores
Reorder retrieved chunks
Improve retrieval precision
Input
Candidate document chunks
User query
Output
Reranked document chunks
3.4 Relevance Agent
Brief Description

The Relevance Agent determines whether the retrieved document chunks provide sufficient evidence to answer the user's question. Unlike traditional keyword filtering approaches, this agent bases its decision on semantic reranking scores, ensuring that only meaningful retrieval evidence is considered.

If the retrieved context is deemed insufficient, the pipeline can reject the query rather than generating unsupported responses.

Purpose

Prevent hallucinations caused by insufficient retrieval.

Responsibilities
Validate retrieval quality
Assess semantic relevance
Detect missing knowledge
Reject out-of-scope questions
Input
Reranked chunks
Output
Knowledge availability decision
3.5 Reflection Agent
Brief Description

The Reflection Agent evaluates the quality of the generated answer before it is returned to the user. Instead of assuming that every generated response is correct, the agent analyzes several quality indicators, including confidence, similarity, retrieval quality, completeness, and uncertainty.

This evaluation forms the basis for the self-correction mechanism introduced later in the pipeline.

Purpose

Evaluate answer quality.

Responsibilities
Confidence estimation
Similarity analysis
Completeness evaluation
Retrieval quality assessment
Uncertainty estimation
Metrics Generated
Confidence Score
Similarity Score
Retrieval Score
Completeness Score
Uncertainty Score
3.6 Self-Correction
Brief Description

One of the key innovations of the Agentic RAG architecture is its ability to perform self-correction. If the Reflection Agent determines that the generated response does not satisfy the predefined confidence threshold, the pipeline automatically attempts another retrieval and generation cycle.

This iterative refinement process improves answer reliability while reducing hallucinations.

Purpose

Improve answer quality through iterative refinement.

Responsibilities
Trigger pipeline retry
Retrieve improved context
Generate revised answer
Log retry statistics
Workflow
Reflection
      ↓
Confidence Low?
      ↓
YES
      ↓
Retrieve Again
      ↓
Generate Again
      ↓
New Reflection
      ↓
Final Answer
Features Achieved in Phase 3

The Agentic RAG implementation introduced several significant improvements over the Core RAG system:

Multi-Agent reasoning architecture
Dynamic retrieval planning
Adaptive Top-K selection
Cross Encoder semantic reranking
Semantic relevance validation
Confidence-based answer evaluation
Reflection-driven quality assessment
Automatic self-correction
Reduced hallucinations
Improved retrieval precision
Explainable decision-making process
Enhanced modularity and extensibility




After successfully building the Core RAG pipeline, the project evolved into an Agentic RAG architecture.

Instead of generating answers immediately after retrieval, the system introduced multiple intelligent agents.

New agents include:

Planner Agent

Analyzes the user query before retrieval.

Responsibilities:

Query classification
Retrieval strategy selection
Dynamic Top-K selection
Procedural vs factual query detection
Retrieval Agent

Coordinates document retrieval.

Responsibilities:

Vector similarity search
Candidate collection
Cross Encoder reranking
Context preparation
Relevance Agent

Introduced to prevent hallucinations.

Responsibilities:

Validate retrieved evidence
Determine whether sufficient knowledge exists
Reject out-of-scope questions

Unlike simple keyword matching, the Relevance Agent makes decisions using semantic retrieval evidence.

Reflection Agent

Evaluates answer quality.

Measures:

Confidence Score
Similarity Score
Retrieval Score
Completeness Score
Uncertainty Score

The Reflection Agent determines whether the generated answer is reliable enough to return to the user.

Self-Correction

If confidence falls below the predefined threshold:

Retrieval is repeated.
Context is regenerated.
A new answer is produced.
Retry statistics are logged.

This transforms the pipeline into a self-correcting Agentic RAG architecture.


Core RAG vs Agentic RAG		
Feature	Core RAG	Agentic RAG
Semantic Retrieval	✓	✓
Fixed Top-K Retrieval	✓	✗
Dynamic Top-K Selection	✗	✓
Planner Agent	✗	✓
Retrieval Agent	✗	✓
Cross Encoder ReRanking	✗	✓
Relevance Validation	✗	✓
Reflection	✗	✓
Confidence Score	✗	✓
Self-Correction	✗	✓
Retry Mechanism	✗	✓
Explainable Decisions	✗	✓
		

## Core RAG vs Self-Correcting Agentic RAG

| Capability | Core RAG | Self-Correcting Agentic RAG |
|------------|:--------:|:---------------------------:|
| PDF Document Ingestion | ✓ | ✓ |
| OCR Support | ✓ | ✓ |
| Semantic Chunking | ✓ | ✓ |
| MiniLM Embeddings | ✓ | ✓ |
| FAISS Vector Database | ✓ | ✓ |
| Semantic Retrieval | ✓ | ✓ |
| Fixed Top-K Retrieval | ✓ | ✗ |
| Dynamic Top-K Retrieval | ✗ | ✓ |
| Planner Agent | ✗ | ✓ |
| Retrieval Agent | ✗ | ✓ |
| Cross-Encoder ReRanking | ✗ | ✓ |
| Relevance Agent | ✗ | ✓ |
| Reflection Agent | ✗ | ✓ |
| Confidence Score Calculation | ✗ | ✓ |
| Similarity Score Evaluation | ✗ | ✓ |
| Retrieval Score Evaluation | ✗ | ✓ |
| Completeness Score Evaluation | ✗ | ✓ |
| Uncertainty Score Evaluation | ✗ | ✓ |
| Self-Correction | ✗ | ✓ |
| Automatic Retry Mechanism | ✗ | ✓ |
| Explainable Decision Making | ✗ | ✓ |
| Hallucination Prevention | Limited | Enhanced |
| Experiment Logger | Basic | Advanced |
| Performance Metrics | Basic | Comprehensive |
| Knowledge Base Manager | ✗ | ✓ |
| Multi-Document Support | ✗ | ✓ |
| Streamlit UI | ✗ | ✓ |
| Fully Local Execution | ✓ | ✓ |
| Privacy Preserving | ✓ | ✓ |


# Phase 4 – Retrieval Improvements


# Phase 4 – Retrieval Quality Enhancements and Knowledge Base Management


## Overview

After implementing the Self-Correcting Agentic RAG architecture in Phase 3, the next objective was to further improve the quality, reliability, and usability of the retrieval pipeline. Although semantic retrieval using FAISS provided relevant document chunks, approximate nearest-neighbor retrieval alone could still return partially relevant or incorrectly ordered results. In addition, many enterprise documents exist as scanned PDFs rather than searchable text, making OCR support essential for real-world deployments.

Phase 4 therefore focused on enhancing the retrieval process through semantic reranking, supporting OCR-based document ingestion, introducing a multi-document knowledge base, and developing a complete Knowledge Base Management system. These enhancements significantly improved retrieval precision, document coverage, and the overall user experience.

Objectives

The primary objectives of Phase 4 were:

Improve retrieval precision using semantic reranking.
Support scanned and image-based PDF documents.
Enable indexing of multiple documents within a unified knowledge base.
Introduce Knowledge Base lifecycle management.
Improve retrieval confidence before answer generation.
Simplify document management through the Streamlit interface.
Phase 4 Architecture
User Uploads Documents
          │
          ▼
PDF Ingestion
          │
          ▼
OCR (if required)
          │
          ▼
Chunk Generation
          │
          ▼
MiniLM Embeddings
          │
          ▼
Unified FAISS Knowledge Base
          │
          ▼
Semantic Retrieval
          │
          ▼
Cross Encoder ReRanking
          │
          ▼
Relevance Validation
          │
          ▼
Reflection Agent
          │
          ▼
Answer Generation
4.1 Cross Encoder ReRanking
Brief Description

Although FAISS efficiently retrieves semantically similar document chunks, embedding similarity alone cannot always determine the most informative context. Retrieved chunks may contain partially related information or may not be optimally ordered for answer generation.

To overcome this limitation, a Cross Encoder ReRanker was introduced. Instead of relying solely on embedding similarity, the Cross Encoder evaluates each retrieved query-document pair simultaneously and assigns a semantic relevance score. The retrieved chunks are then reordered according to these scores before being passed to the language model.

This additional semantic evaluation significantly improves retrieval precision and provides higher quality evidence for response generation.

Purpose

Improve the quality of retrieved context before answer generation.

Responsibilities
Evaluate semantic similarity between the query and each retrieved chunk.
Assign a reranking score.
Reorder retrieved chunks.
Improve evidence quality.
Support confidence estimation.
Input
User query
Retrieved document chunks
Output
Reranked document chunks
Interaction
FAISS Retrieval
        │
        ▼
Cross Encoder
        │
        ▼
ReRanked Chunks
Advantages
Higher retrieval precision
Improved semantic relevance
Better supporting evidence
Reduced noisy context
More reliable answer generation
Improved confidence estimation
4.2 OCR Integration
Brief Description

Many enterprise environments store technical manuals, reports, invoices, and historical documents as scanned PDF files rather than searchable text documents. Traditional PDF extraction techniques cannot process these image-based documents effectively.

To address this challenge, the ingestion pipeline was enhanced with Optical Character Recognition (OCR) support using Tesseract OCR. The system first attempts direct text extraction using PyMuPDF. If insufficient text is detected, the pipeline automatically invokes OCR to extract textual information from scanned pages.

This hybrid approach enables the system to process both digital and scanned PDFs without requiring user intervention.

Purpose

Enable support for scanned PDF documents.

Responsibilities
Detect insufficient text extraction.
Perform OCR when required.
Merge OCR output into the standard ingestion pipeline.
Support mixed document types.
Input
Digital PDF
Scanned PDF
Output
Extracted document text
Interaction
PDF
 │
 ├── Text Available
 │        │
 │        ▼
 │   Use PyMuPDF
 │
 └── Insufficient Text
          │
          ▼
     Tesseract OCR
          │
          ▼
     Extracted Text
Advantages
Supports scanned documents.
Handles image-based PDFs.
Improves enterprise document compatibility.
Automatic OCR fallback.
No manual preprocessing required.
4.3 Multi-Document Knowledge Base
Brief Description

The initial implementation of the Core RAG pipeline supported indexing a single document at a time. While suitable for early experimentation, this approach limited the practical usefulness of the system.

Phase 4 introduced a Multi-Document Knowledge Base, enabling multiple PDF documents to be indexed together within a single FAISS vector database. Each document is chunked, embedded, and stored with associated metadata, allowing the retrieval pipeline to search across the complete document collection.

This enhancement enables users to build larger knowledge repositories while maintaining efficient semantic retrieval.

Purpose

Support multiple documents within a unified retrieval system.

Responsibilities
Index multiple documents.
Store document metadata.
Build a unified FAISS index.
Retrieve evidence across document collections.
Input
Multiple PDF documents
Output
Unified knowledge base
Interaction
PDF 1
PDF 2
PDF 3
   │
   ▼
Chunking
   │
   ▼
Embeddings
   │
   ▼
Unified FAISS Index
Advantages
Supports enterprise knowledge repositories.
Enables cross-document retrieval.
Improves answer completeness.
Simplifies document management.
Easily extensible.
4.4 Knowledge Base Manager
Brief Description

To simplify document management, a dedicated Knowledge Base Manager was developed as part of the Streamlit interface. Rather than manually creating vector indexes, users can upload documents, remove obsolete files, rebuild the knowledge base, and monitor indexing status directly from the application.

The Knowledge Base Manager maintains metadata describing indexed documents and ensures that the FAISS index remains synchronized with the uploaded document collection.

Purpose

Manage the lifecycle of indexed documents.

Responsibilities
Upload documents.
Delete selected documents.
Delete all documents.
Build Knowledge Base.
Maintain metadata.
Display indexing status.
Features
Upload PDF documents
Delete selected documents
Delete all indexed documents
Build Knowledge Base
Display indexing status
Multi-document support
Metadata management
Knowledge Base Workflow
Upload PDF
      │
      ▼
Store Document
      │
      ▼
Build Knowledge Base
      │
      ▼
Chunk Documents
      │
      ▼
Generate Embeddings
      │
      ▼
Create FAISS Index
      │
      ▼
Update Metadata
Technologies Introduced in Phase 4
Component	Technology
OCR Engine	Tesseract OCR
PDF Processing	PyMuPDF
Image Processing	Pillow
OCR Wrapper	PyTesseract
Semantic ReRanking	Cross Encoder
Vector Database	FAISS
Knowledge Base	Multi-Document FAISS Index
UI Framework	Streamlit
Features Achieved in Phase 4
Cross Encoder semantic reranking
OCR support for scanned PDFs
Automatic OCR fallback
Multi-document indexing
Unified FAISS knowledge base
Knowledge Base metadata management
Knowledge Base Manager
Upload and delete document support
Build-on-demand indexing
Improved retrieval precision
Improved retrieval confidence
Enhanced enterprise document compatibility
Improved user experience
Summary

Phase 4 significantly enhanced the retrieval capabilities and usability of the Private Agentic RAG System. By introducing Cross Encoder ReRanking, OCR-based document ingestion, multi-document indexing, and a comprehensive Knowledge Base Manager, the system became more accurate, flexible, and practical for real-world document collections. These enhancements laid the foundation for Phase 5, where the focus shifted to delivering an interactive Streamlit-based user interface that exposes the complete functionality of the Agentic RAG pipeline through an intuitive and user-friendly experience.

Several enhancements were introduced to improve retrieval quality.

Cross Encoder ReRanking

Initial FAISS retrieval provides approximate semantic matches.

The Cross Encoder then reranks retrieved chunks based on query-document relevance.

Advantages:

Higher retrieval precision
Better evidence quality
Improved confidence estimation
OCR Integration

Many enterprise documents contain scanned pages.

The ingestion pipeline now supports:

Direct PDF extraction
OCR fallback using Tesseract
Mixed text and scanned PDFs
Multi-Document Knowledge Base

The system now supports:

Multiple uploaded documents
Unified FAISS index
Knowledge Base metadata
Build-on-demand indexing
Knowledge Base Manager

Introduced document lifecycle management.

Features include:

Upload PDF
Delete documents
Delete all documents
Build Knowledge Base
Document indexing status


# Phase 5 – Interactive Streamlit User Interface

## Overview

The final phase of the project focused on developing an intuitive and interactive web-based interface using Streamlit, transforming the backend implementation into a complete end-to-end application. While the previous phases concentrated on building the Core RAG pipeline, Agentic reasoning framework, retrieval improvements, and evaluation mechanisms, Phase 5 integrated all these components into a single user-friendly interface.

The Streamlit application serves as the primary interaction layer between the user and the Agentic RAG system. It enables users to upload documents, manage the knowledge base, build vector indexes, execute both Core RAG and Agentic RAG pipelines, visualize retrieved evidence, inspect agent decisions, monitor confidence metrics, and interact with the system through a conversational chat interface.

The objective of this phase was not only to improve usability but also to expose the internal reasoning process of the Agentic RAG system, making the retrieval and generation workflow transparent and explainable.

Objectives

The primary objectives of Phase 5 were:

Develop an intuitive web-based interface.
Integrate all backend modules into a single application.
Provide an interactive document management system.
Visualize retrieval and reasoning processes.
Display confidence and evaluation metrics.
Simplify experimentation with Core and Agentic pipelines.
Improve the overall user experience.
Streamlit Application Architecture
                     User

                       │

                       ▼

              Streamlit Interface

                       │

        ┌──────────────┼──────────────┐

        │              │              │

        ▼              ▼              ▼

Knowledge Base     Chat System     Pipeline Selection

Manager                          (Core / Agentic)

        │              │              │

        └──────────────┼──────────────┘

                       ▼

             Agentic RAG Backend

                       ▼

              Final Generated Answer
User Interface Components

The Streamlit interface consists of several integrated modules designed to simplify interaction with the RAG system.

5.1 Knowledge Base Manager
Brief Description

The Knowledge Base Manager provides complete lifecycle management for uploaded documents. Rather than requiring manual indexing through command-line scripts, users can upload PDF documents, monitor indexing status, build the knowledge base, and manage stored documents directly through the interface.

The manager maintains synchronization between uploaded files, metadata, and the FAISS vector database, ensuring that retrieval always operates on the latest indexed document collection.

Purpose

Provide centralized management of the document repository.

Responsibilities
Upload PDF documents
Display uploaded documents
Show indexing status
Build Knowledge Base
Delete selected documents
Delete all documents
Maintain metadata consistency
Features
Multi-document upload
Knowledge Base status
Index build progress
Metadata tracking
Document lifecycle management
5.2 Build Knowledge Base
Brief Description

Building the Knowledge Base converts uploaded documents into a searchable semantic index. During this process, each document is processed through the complete ingestion pipeline, including OCR (if necessary), chunk generation, embedding creation, and FAISS index construction.

To improve efficiency, the system compares uploaded documents with stored metadata and avoids rebuilding the index if no changes have occurred.

Workflow
Upload Documents

↓

Compare Metadata

↓

Changes Detected?

↓

No → Knowledge Base Already Up-to-Date

↓

Yes

↓

Extract Text

↓

Chunk Documents

↓

Generate Embeddings

↓

Build FAISS Index

↓

Save Metadata

↓

Knowledge Base Ready
Advantages
Prevents unnecessary index rebuilding.
Reduces processing time.
Maintains synchronization.
Supports incremental document management.
5.3 Pipeline Selection
Brief Description

The interface allows users to switch seamlessly between the traditional Core RAG pipeline and the advanced Self-Correcting Agentic RAG pipeline. This functionality enables direct comparison of both architectures while using the same document collection and user query.

Available Modes
Core RAG

Traditional retrieval and generation pipeline.

Agentic RAG

Planner → Retrieval → ReRanking → Reflection → Self-Correction.

5.4 Interactive Chat Interface
Brief Description

The chat interface provides a conversational interaction model similar to modern AI assistants. Users can submit natural language questions related to indexed documents and receive grounded responses generated entirely within the local environment.

Each query is processed through the selected RAG pipeline before the generated response is displayed.

Features
Natural language interaction
Real-time response generation
Context-aware answers
Local inference
Streamlit chat components
5.5 Retrieved Context Visualization
Brief Description

To improve transparency, the interface displays the document chunks retrieved from the vector database. Users can inspect the supporting evidence used during answer generation, helping validate whether the generated response is grounded in relevant information.

This feature significantly improves explainability compared to conventional black-box language model interfaces.

Information Displayed
Retrieved chunks
Source document names
Supporting evidence
5.6 Agentic Metrics Dashboard
Brief Description

When the Agentic RAG pipeline is selected, the interface exposes internal reasoning metrics generated by the Reflection Agent. These metrics allow users to understand the quality and reliability of each generated response.

Metrics Displayed
Confidence Score
Similarity Score
Retrieval Score
Completeness Score
Uncertainty Score
Benefits
Transparent reasoning
Improved explainability
Easy debugging
Evaluation support
5.7 Planner Decision Visualization
Brief Description

The Planner Agent's decisions are displayed through the interface, allowing users to observe how the system classified the query and selected the retrieval strategy.

Displayed information includes:

Query type
Retrieval strategy
Dynamic Top-K value
5.8 Self-Correction Information
Brief Description

Whenever the Reflection Agent determines that the generated response does not satisfy the confidence threshold, the interface indicates that self-correction has been triggered.

Users can observe:

Retry count
Confidence values
Whether self-correction occurred
Final pipeline status

This feature highlights one of the distinguishing characteristics of the proposed Agentic RAG architecture.

Technologies Used
Component	Technology
Programming Language	Python
PDF Processing	PyMuPDF
OCR	Tesseract OCR
Embedding Model	Sentence Transformers (all-MiniLM-L6-v2)
Vector Database	FAISS
Semantic ReRanking	Cross Encoder
Small Language Model	Mistral (Quantized)
Local Runtime	Ollama
User Interface	Streamlit
Experiment Logging	CSV Logger
Evaluation & Visualization	Matplotlib
Final System Architecture
                    PDF Documents
                           │
                           ▼
               OCR + Text Extraction
                           │
                           ▼
                    Text Chunking
                           │
                           ▼
               MiniLM Embedding Model
                           │
                           ▼
                  FAISS Vector Database
                           │
                           ▼
                    Planner Agent
                           │
                           ▼
                  Retrieval Agent
                           │
                           ▼
               Cross Encoder ReRanking
                           │
                           ▼
                  Relevance Agent
                           │
                           ▼
                 Prompt Construction
                           │
                           ▼
              Mistral SLM (via Ollama)
                           │
                           ▼
                 Reflection Agent
                           │
                           ▼
                Confidence Evaluation
                           │
                           ▼
                  Self-Correction
                           │
                           ▼
                 Final AI Response
                           │
                           ▼
                Experiment Logger
                           │
                           ▼
                  Streamlit User Interface
Key Features of the Final System

The completed system provides the following capabilities:

Fully local Retrieval-Augmented Generation
Privacy-preserving document processing
Self-Correcting Agentic RAG architecture
Planner-based adaptive retrieval
Dynamic Top-K selection
Cross Encoder semantic reranking
OCR support for scanned PDFs
Multi-document knowledge base
Knowledge Base Manager
Interactive Streamlit chat interface
Retrieved context visualization
Confidence score evaluation
Reflection-based reasoning
Explainable AI decisions
Experiment logging
Performance evaluation
Modular and extensible software architecture
Current Project Status
Phase	Status
Phase 1 – Project Foundation	✅ Completed
Phase 2 – Core Private RAG System	✅ Completed
Phase 3 – Self-Correcting Agentic RAG	✅ Completed
Phase 4 – Retrieval Improvements & Knowledge Base Management	✅ Completed
Phase 5 – Interactive Streamlit User Interface	✅ Completed
Project Outcome

The project successfully evolved from a traditional document retrieval system into a fully local, privacy-preserving, self-correcting Agentic Retrieval-Augmented Generation framework. By integrating semantic retrieval, intelligent planning, reranking, reflection, confidence evaluation, self-correction, OCR support, comprehensive experiment logging, and an interactive Streamlit interface, the system demonstrates a practical and extensible architecture for secure enterprise document question answering. The modular design also provides a strong foundation for future enhancements such as hybrid retrieval, GraphRAG, multimodal document understanding, and advanced agent orchestration.