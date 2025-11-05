CHUNK_SIZE = 512  # maximum number of tokens per text chunk

# you can switch between embedding models by uncommenting the one you want
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"  # main embedding model used
LLM_NAME = "microsoft/Phi-4-mini-instruct"  # language model used for generating answers
# LLM_NAME = "MarsGray/Phi-4-mini-instruct-lora-model"  # alternative fine-tuned model

RAW_PDF_DIR = "data/albania.pdf"  # path to the input PDF file
JSONL_OUTPUT_DIR = "processed/jsonl"  # directory where processed JSONL files are stored
VECTOR_STORE_PATH = "processed/vector_store/"  # location of the vector database




