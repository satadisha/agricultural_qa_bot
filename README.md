# SeedBot — RAG Chatbot for Seed Laws

> A lightweight Retrieval-Augmented Generation (RAG) chatbot that answers questions about seed laws and related regulations from PDFs — with grounded citations.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![Status](https://img.shields.io/badge/status-experimental-lightgrey)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

**SeedBot** ingests legal PDFs, converts them to structured text, builds embeddings, and retrieves the most relevant passages for a user’s question. It generates concise answers **with citations** to the original text.

---

## Features
- **End-to-end pipeline:** Extract → Section Splitter → Chunking → Embed Store → Process PDF → Get Best Chunk.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Store:** FAISS or SQLite.
- **Simple CLI** for local Q&A.

---

## Quick Start
```bash
# 1) Clone & enter
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>

# 2) Create environment & install dependencies
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3) Add your PDFs
mkdir -p data/pdfs
unzip upov_docs.zip -d data/pdfs

# 4) Run files in order

# (1) Extract PDFs → JSON
python extract.py --in data/pdfs --out processed/docs_json

# (2) Section Splitter → structured sections
python section_splitter.py --in processed/docs_json --out processed/docs_sections

# (3) Chunking → text chunks
python chunking.py --in processed/docs_sections --out processed/docs_chunks

# (4) Embed Store → FAISS index
python embed_store.py --in processed/docs_chunks --index processed/faiss_index

# (5) Process PDF → full pipeline (optional combined command)
# For a single PDF
python process_pdf.py --input_file data/albania.pdf --output_dir processed/jsonl --chroma_dir processed/chroma_db

# For all PDFs in a folder
python process_pdf.py --input_dir data/pdfs --output_dir processed/jsonl --chroma_dir processed/chroma_db

# (6) Get Best Chunk → retrieve answer
python best_chunk.py --index processed/faiss_index --query "What must appear on a seed label?"
