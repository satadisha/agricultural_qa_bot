import os
import json
import argparse
from pathlib import Path
from src.extract import extract_text_from_pdf          # extracts text from PDFs
from src.section_splitter import split_into_section    # splits text into sections
from src.chunking import chunk_text                    # breaks text into smaller chunks
from src.embed_store import store_to_chroma            # stores chunks and embeddings
from src.config import EMBEDDING_MODEL_NAME, JSONL_OUTPUT_DIR  # config settings

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import tiktoken
from langdetect import detect  # detect language of the text

# Setup tokenizer and encoder
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
enc = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 512  # maximum token limit per chunk


def count_tokens(text: str) -> int:
    # count the number of tokens in text
    return len(enc.encode(text))


def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    # shorten text if it exceeds token limit
    input_ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def detect_language(text: str) -> str:
    # detect the language of a given text
    try:
        return detect(text)
    except:
        return "unknown"


def fingerprint(text: str) -> str:
    # create a short unique fingerprint from the first 50 words
    return " ".join(text.lower().split()[:50])  # Normalize and take first 50 words


def process_single_pdf(pdf_path: Path, jsonl_dir: Path, chroma_path: Path):
    print(f"\n📄 Processing: {pdf_path.name}")

    # extract text from the PDF file
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text or not raw_text.strip():
        print(f"❌ No usable text found in {pdf_path.name}")
        return

    # split extracted text into logical sections
    sections = split_into_section(raw_text)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)  # load embedding model

    # create folders for output files
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_dir / f"{pdf_path.stem}.jsonl"

    # gather metadata (country name and language)
    country = pdf_path.stem.split("_")[0].lower()
    language = detect_language(raw_text)
    seen_fingerprints = set()  # avoid duplicates
    jsonl_entries = []  # store chunks before writing to file

    for section_index, section in enumerate(sections):
        section_heading = section.get("heading", "")
        sub_heading_raw = section.get("sub_heading", [])
        sub_heading = ", ".join(sub_heading_raw) if isinstance(sub_heading_raw, list) else str(sub_heading_raw)

        # split each section into overlapping text chunks
        chunks = chunk_text(
            section_text=section["text"],
            section_heading=section_heading,
            max_tokens=MAX_TOKENS,
            sentence_overlap=2
        )

        for chunk_index, chunk in enumerate(chunks):
            text_chunk = chunk["text"]
            raw_text = f"{section_heading}\n\n{text_chunk}" if section_heading else text_chunk
            full_text = truncate_text(raw_text, max_tokens=MAX_TOKENS)

            # skip duplicate chunks
            fp = fingerprint(full_text)
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)

            # generate embedding vector for the chunk
            embedding = model.encode(full_text, show_progress_bar=False).tolist()
            chunk_id = f"{country}{pdf_path.stem}-s{section_index+1}-c{chunk_index+1}"

            # store related metadata
            metadata = {
                "chunk_id": chunk_id,
                "tokens": count_tokens(full_text),
                "country": country,
                "language": language,
                "source_file": pdf_path.name,
                "section_heading": section_heading,
                "sub_heading": sub_heading
            }

            # create a JSON entry for this chunk
            entry = {
                "chunk_id": chunk_id,
                "text": full_text,
                "embedding": embedding,
                "metadata": metadata
            }

            jsonl_entries.append(entry)

    # Save chunks and metadata as JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for entry in jsonl_entries:
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Store all chunks into ChromaDB
    store_to_chroma(jsonl_entries, collection_name="regulations", chroma_path=chroma_path)

    # print completion info
    print(f"✅ Processed {pdf_path.name}")
    print(f"📦 ChromaDB stored at: {chroma_path}")
    print(f"📝 JSONL written to: {jsonl_path}")


def main():
    # setup command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to a single PDF file")
    parser.add_argument("--input_dir", type=str, help="Folder containing multiple PDF files")
    parser.add_argument("--output_dir", type=str, default="processed/jsonl", help="Where to store JSONL files")
    parser.add_argument("--chroma_dir", type=str, default="processed/chroma_db", help="Where to store Chroma DB")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    chroma_dir = Path(args.chroma_dir)

    # process a single file if specified
    if args.input_file:
        pdf_path = Path(args.input_file)
        if pdf_path.exists() and pdf_path.suffix.lower() == ".pdf":
            process_single_pdf(pdf_path, jsonl_dir=output_dir, chroma_path=chroma_dir)
        else:
            print("❌ Invalid or missing PDF file.")
    # or process all PDFs in a folder
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            print("❌ No PDF files found in input directory.")
            return
        for pdf_path in pdf_files:
            process_single_pdf(pdf_path, jsonl_dir=output_dir, chroma_path=chroma_dir)
    else:
        print("❌ Please provide either --input_fil_

