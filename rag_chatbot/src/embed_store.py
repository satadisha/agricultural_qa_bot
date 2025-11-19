"""
PDF → Section → Chunk → Embedding → Chroma Pipeline

This module processes a PDF file by:
1. Extracting raw text.
2. Splitting it into logical sections.
3. Chunking each section into small token-limited pieces.
4. Embedding each chunk using SentenceTransformer.
5. Storing chunks + metadata + embeddings into a Chroma vector database.
6. Saving a JSONL file containing all processed chunks.

It also provides a helper function (`store_to_chroma`) for inserting
already-processed entries into ChromaDB.

Configuration such as embedding model name, tokenizer, and directories
is imported from `src.config`.
"""

import os
import json
from pathlib import Path
from hashlib import sha256

from src.extract import extract_text_from_pdf
from src.chunking import chunk_text
from src.section_splitter import split_into_section
from src.config import EMBEDDING_MODEL_NAME, JSONL_OUTPUT_DIR

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from transformers import AutoTokenizer
import tiktoken
from langdetect import detect

# Initialize tokenizer and encoder
enc = tiktoken.get_encoding("cl100k_base")  # for token counting
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
MAX_TOKENS = 512


def count_tokens(text: str) -> int:
    """
    Count tokens in a string using the tiktoken encoder.

    Args:
        text (str): Input text.

    Returns:
        int: Number of tokens.
    """
    return len(enc.encode(text))


def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """
    Truncate text so it does not exceed max_tokens.

    Args:
        text (str): Input text.
        max_tokens (int): Maximum allowed tokens.

    Returns:
        str: Text truncated to max_tokens.
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def detect_language(text: str) -> str:
    """
    Detect the language of a text block.

    Args:
        text (str): Input text.

    Returns:
        str: Language code (ISO 639-1) or "unknown".
    """
    try:
        return detect(text)
    except Exception:
        return "unknown"


def get_fingerprint(text: str, length: int = 50) -> str:
    """
    Create a stable hash fingerprint using the first `length` words of the text.

    Args:
        text (str): Chunk text.
        length (int): Number of words to fingerprint.

    Returns:
        str: SHA256 hexadecimal digest.
    """
    return sha256(" ".join(text.split()[:length]).encode()).hexdigest()


def process_pdf_to_chroma(
    pdf_path: Path,
    chroma_path: Path,
    collection_name: str = "regulations",
    jsonl_dir: Path = JSONL_OUTPUT_DIR
):
    """
    Process a PDF and store its chunk embeddings in ChromaDB + JSONL.

    Workflow:
    - Extract text from PDF.
    - Split into hierarchical sections.
    - Chunk each section to token-safe pieces.
    - Detect metadata (country, language).
    - Deduplicate using fingerprint hashes.
    - Embed chunks with SentenceTransformer.
    - Store (text, metadata, embedding) into Chroma.
    - Save all chunks to a JSONL file.

    Args:
        pdf_path (Path): Path to the PDF file.
        chroma_path (Path): Directory containing the persistent ChromaDB.
        collection_name (str): Name of the Chroma collection.
        jsonl_dir (Path): Directory to store JSONL chunk outputs.

    Returns:
        None
    """

    # extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    if not text or not text.strip():
        print(f"❌ No usable text in {pdf_path.name}")
        return

    # split text into structured sections
    sections = split_into_section(text)

    # setup Chroma database
    client = PersistentClient(path=chroma_path)
    embedder = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedder)

    # prepare JSONL file
    jsonl_path = jsonl_dir / f"{pdf_path.stem}.jsonl"
    os.makedirs(jsonl_dir, exist_ok=True)

    # metadata extraction
    country = pdf_path.stem.split("_")[0].lower()
    language = detect_language(text)
    seen_fingerprints = set()

    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for sec_idx, section in enumerate(sections, 1):
            section_heading = section.get("heading", "").strip()
            sub_heading_raw = section.get("sub_heading", [])
            sub_heading = ", ".join(sub_heading_raw) if isinstance(sub_heading_raw, list) else str(sub_heading_raw)

            # break section into text chunks
            chunks = chunk_text(section["text"], section_heading=section_heading)

            for chunk_idx, chunk in enumerate(chunks, 1):
                raw_text = (
                    f"{section_heading}\n\n{chunk['text']}".strip()
                    if section_heading else chunk["text"]
                )
                full_text = truncate_text(raw_text)

                # dedupe
                fingerprint = get_fingerprint(full_text)
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)

                # embed
                embedding = embedder([full_text])[0]
                chunk_id = f"{country}{pdf_path.stem}-s{sec_idx}-c{chunk_idx}"

                metadata = {
                    "chunk_id": chunk_id,
                    "tokens": count_tokens(full_text),
                    "country": country,
                    "language": language,
                    "source_file": pdf_path.name,
                    "section_heading": section_heading,
                    "sub_heading": sub_heading
                }

                # refresh + store
                collection.delete(ids=[chunk_id])
                collection.add(
                    documents=[full_text],
                    metadatas=[metadata],
                    embeddings=[embedding],
                    ids=[chunk_id]
                )

                # write JSONL line
                f_out.write(json.dumps({
                    "chunk_id": chunk_id,
                    "text": full_text,
                    "embedding": embedding.tolist(),
                    "metadata": metadata
                }, ensure_ascii=False) + "\n")

    print(f"✅ Processed {pdf_path.name}")
    print(f"📦 Stored in ChromaDB: {chroma_path}")
    print(f"📝 JSONL written to: {jsonl_path}")


def store_to_chroma(entries, chroma_path, collection_name="regulations"):
    """
    Store preprocessed chunk entries into an existing Chroma collection.

    Args:
        entries (list[dict]): Each entry must contain:
            - text (str)
            - embedding (list[float])
            - metadata (dict)
        chroma_path (str or Path): Path to persistent Chroma directory.
        collection_name (str): Name of collection receiving entries.

    Returns:
        None
    """

    client = PersistentClient(path=chroma_path)
    embedder = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedder)

    for entry in entries:
        text = entry["text"]
        metadata = entry["metadata"]
        embedding = entry["embedding"]
        chunk_id = metadata["chunk_id"]

        # overwrite existing chunk id
        collection.delete(ids=[chunk_id])
        collection.add(
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding],
            ids=[chunk_id]
        )

