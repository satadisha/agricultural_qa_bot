# Seedbot: Agricultural Law Q&A Chatbot

A RAG-powered chatbot helping farmers navigate agricultural laws across 40+ countries. Built with open-source LLMs in collaboration with [A Growing Culture](https://www.agrowingculture.org/).

## Features

- Multilingual support for queries and documents
- Country-specific legal information from UPOV dataset
- Semantic search with custom reranking
- Web interface and CLI tools
- Built with Phi-4 and multilingual-e5 models

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process documents
python process_pdf.py --input_dir data/chile --chroma_dir processed/chroma_db

# Start web server
uvicorn fastapi_app.main:app --reload
```

Visit `http://127.0.0.1:8000` to use the chatbot.

## Usage

### Web Interface
```bash
uvicorn fastapi_app.main:app --reload
```

### CLI Chat
```bash
python chat.py --country albania --db_path processed/chroma_db
```

### Single Query
```bash
python get_response.py --country paraguay --query "Your question here" --db_path processed/chroma_db
```

## Configuration

Edit `src/config.py`:
```python
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
LLM_NAME = "microsoft/Phi-4-mini-instruct"
CHUNK_SIZE = 512
```

## Project Structure

```
rag_chatbot/
├── data/              # Raw PDFs by country
├── processed/         # JSONL and ChromaDB
├── src/               # Core processing modules
├── fastapi_app/       # Web application
├── test/              # Test scripts
└── *.py               # CLI tools
```

## How It Works

1. **Retrieval**: Query embedded → ChromaDB search → Reranking with keyword/article bonuses
2. **Generation**: Context + prompt → Phi-4 → Concise answer in same language

## Team

Built by students from Morehouse, Howard, Florida A&M, Cal State Fresno, and Prairie View A&M as part of UChicago's Data Science for Social Impact Program.

**Mentors**: Dr. Satadisha Saha Bhowmick, Summer Han, Dr. Kriti Sehgal

## Acknowledgments

- [A Growing Culture](https://www.agrowingculture.org/)
- [UPOV](https://www.upov.int/) for datasets
- University of Chicago DSSI Program
