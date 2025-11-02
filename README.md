# SeedBot — RAG Chatbot for Seed Laws

> A lightweight Retrieval-Augmented Generation (RAG) chatbot that answers questions about seed laws and related regulations from PDFs — with grounded citations.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)](#)
[![Status](https://img.shields.io/badge/status-experimental-lightgrey)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

[![SeedBot](seedbot.farm)](#seedbot.farm) ingests legal PDFs, converts them to structured text, builds embeddings, and retrieves the most relevant passages for a user’s question. It then generates concise answers **with citations** to the original source snippets. The repository includes a local CLI for quick Q&A and a **FastAPI** service for programmatic use.

---

## Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI (local chat)](#cli-local-chat)
  - [FastAPI server](#fastapi-server)
- [Configuration](#configuration)
- [Data & Artifacts](#data--artifacts)
- [Project Structure](#project-structure)
- [API / CLI Reference](#api--cli-reference)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features
- **PDF → text** pipeline with simple preprocessing and optional metadata (`upov_docs_metadata.csv`).
- **Embeddings** (e.g., `sentence-transformers/all-MiniLM-L6-v2`) and **FAISS/SQLite** vector search (configurable).
- **RAG**: retrieve top-k chunks, optionally rerank, and generate answers with **source citations**.
- **Interfaces**: local CLI (`chat.py`, `chat1.py`) and a **FastAPI** service (`fastapi_app/`).
- **Reproducible**: processed text and vector artifacts cached under `processed/`.

---

## Demo
- **Ask**: “What must appear on a seed label?”  
- **Returns**: A short answer plus citations like `state_x_seed_law_2020.pdf (p.21)` and a snippet from the source.

---

## Quick Start

```bash
# 1) Clone & enter
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>

# 2) Create environment & install deps
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3) Put PDFs in data/pdfs (or unzip sample)
unzip upov_docs.zip -d data/pdfs

# 4) Build the index (extract → chunk → embed)
python process_pdf.py   # run with -h to see options

# 5) Ask a question (CLI)
python chat.py --question "What must appear on a seed label?" --k 5
