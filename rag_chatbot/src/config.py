CHUNK_SIZE = 512
"""Maximum number of tokens allowed per chunk during PDF processing."""

# you can switch between embedding models by uncommenting the one you want
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
"""Embedding model name used for vector generation."""

LLM_NAME = "microsoft/Phi-4-mini-instruct"
"""Language model used for answer generation and inference."""
# LLM_NAME = "MarsGray/Phi-4-mini-instruct-lora-model"

RAW_PDF_DIR = "data/albania.pdf"
"""Path to the input PDF file to be extracted and processed."""

JSONL_OUTPUT_DIR = "processed/jsonl"
"""Directory where processed JSONL chunk files will be stored."""

VECTOR_STORE_PATH = "processed/vector_store/"
"""Directory where the vector store (e.g., ChromaDB) will be saved."""



