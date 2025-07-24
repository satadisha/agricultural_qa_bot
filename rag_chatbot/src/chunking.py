from typing import List, Dict
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize

# nltk.download('punkt')  # Uncomment on first run
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def chunk_text(
    section_text: str,
    section_heading: str = "",
    max_tokens: int = 512,
    max_chars: int = 1000,
    sentence_overlap: int = 2
) -> List[Dict]:
    heading_tokens = count_tokens(section_heading)
    token_budget = max_tokens - heading_tokens

    if token_budget <= 0:
        section_heading = ""
        token_budget = max_tokens

    sentences = sent_tokenize(section_text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    current_chars = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        sentence_chars = len(sentence)

        # If sentence is too long, truncate it
        if sentence_tokens > token_budget or sentence_chars > max_chars:
            sentence = tokenizer.decode(tokenizer.encode(sentence)[:token_budget])
            sentence_tokens = count_tokens(sentence)
            sentence_chars = len(sentence)

        if (current_tokens + sentence_tokens > token_budget) or (current_chars + sentence_chars > max_chars):
            if current_chunk:
                chunk_text_combined = " ".join(current_chunk)
                chunks.append({
                    "section_heading": section_heading,
                    "text": chunk_text_combined.strip()
                })
                # Add overlap
                overlap = current_chunk[-sentence_overlap:] if sentence_overlap > 0 else []
                current_chunk = list(overlap)
                current_tokens = sum(count_tokens(s) for s in current_chunk)
                current_chars = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_tokens += sentence_tokens
        current_chars += sentence_chars

    if current_chunk:
        chunk_text_combined = " ".join(current_chunk)
        chunks.append({
            "section_heading": section_heading,
            "text": chunk_text_combined.strip()
        })

    return chunks
