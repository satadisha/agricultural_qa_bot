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

# Initialize tokenizer and tiktoken encoder
enc = tiktoken.get_encoding("cl100k_base")
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
MAX_TOKENS = 512


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    input_ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


def get_fingerprint(text: str, length: int = 50) -> str:
    return sha256(" ".join(text.split()[:length]).encode()).hexdigest()


def process_pdf_to_chroma(
    pdf_path: Path,
    chroma_path: Path,
    collection_name: str = "regulations",
    jsonl_dir: Path = JSONL_OUTPUT_DIR
):
    text = extract_text_from_pdf(pdf_path)
    if not text or not text.strip():
        print(f"❌ No usable text in {pdf_path.name}")
        return

    sections = split_into_section(text)

    # Setup Chroma
    client = PersistentClient(path=chroma_path)
    embedder = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedder)

    # Prepare output
    jsonl_path = jsonl_dir / f"{pdf_path.stem}.jsonl"
    os.makedirs(jsonl_dir, exist_ok=True)

    country = pdf_path.stem.split("_")[0].lower()
    language = detect_language(text)
    seen_fingerprints = set()

    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for sec_idx, section in enumerate(sections, 1):
            section_heading = section.get("heading", "").strip()
            sub_heading_raw = section.get("sub_heading", [])
            sub_heading = ", ".join(sub_heading_raw) if isinstance(sub_heading_raw, list) else str(sub_heading_raw)

            chunks = chunk_text(section["text"], section_heading=section_heading)

            for chunk_idx, chunk in enumerate(chunks, 1):
                raw_text = f"{section_heading}\n\n{chunk['text']}".strip() if section_heading else chunk['text']
                full_text = truncate_text(raw_text)

                fingerprint = get_fingerprint(full_text)
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)

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

                # Avoid duplication and store
                collection.delete(ids=[chunk_id])
                collection.add(
                    documents=[full_text],
                    metadatas=[metadata],
                    embeddings=[embedding],
                    ids=[chunk_id]
                )

                # Save to JSONL
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
    client = PersistentClient(path=chroma_path)
    embedder = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedder)

    for entry in entries:
        text = entry["text"]
        metadata = entry["metadata"]
        embedding = entry["embedding"]
        chunk_id = metadata["chunk_id"]

        collection.delete(ids=[chunk_id])
        collection.add(
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding],
            ids=[chunk_id]
        )
