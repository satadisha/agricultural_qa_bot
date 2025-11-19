CHUNK_SIZE = 512  
"""int: Maximum number of tokens allowed per text chunk during PDF processing."""

# you can switch between embedding models by uncommenting the one you want
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
"""str: Name of the embedding model used for vector generation."""

LLM_NAME = "microsoft/Phi-4-mini-instruct"
"""str: Name of the language model used for answer generation and inference."""
# LLM_NAME = "MarsGray/Phi-4-mini-instruct-lora-model"

RAW_PDF_DIR = "data/albania.pdf"
"""str: Path to the input PDF file that will be extracted and processed."""

JSONL_OUTPUT_DIR = "processed/jsonl"
"""str: Directory where processed JSONL chunk files will be stored."""

VECTOR_STORE_PATH = "processed/vector_store/"
"""str: Directory path where the vector store (ChromaDB or similar) will be saved."""



