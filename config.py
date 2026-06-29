"""
Project Configuration File
--------------------------
Central configuration for the Private Agentic RAG System.

Author: Kathula Deepak
Project: Design and Evaluation of a Self-Correcting,
Privacy-Preserving Agentic RAG System using Quantized
Small Language Models
"""

##########################################################
# SLM CONFIGURATION
##########################################################

SLM_MODEL = "mistral"
# Examples:
# "mistral"
# "phi3:mini"

##########################################################
# EMBEDDING MODEL
##########################################################

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

##########################################################
# CHUNKING CONFIGURATION
##########################################################

CHUNK_SIZE = 500

CHUNK_OVERLAP = 50

##########################################################
# RETRIEVAL CONFIGURATION
##########################################################

TOP_K = 2

##########################################################
# DOCUMENT PATHS
##########################################################

DOCUMENT_FOLDER = "data/documents"

##########################################################
# VECTOR DATABASE
##########################################################

VECTOR_DB_FOLDER = "data/vector_db"

FAISS_INDEX = "data/vector_db/faiss.index"

METADATA_FILE = "data/vector_db/metadata.pkl"

##########################################################
# LOGGING
##########################################################

LOG_FOLDER = "logs"

ENABLE_LOGGER = True

##########################################################
# PROJECT INFO
##########################################################

PROJECT_NAME = "Private Agentic RAG"

VERSION = "Phase2"