from typing import List, Dict
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize

# nltk.download('punkt')  # Uncomment on first run to download sentence tokenizer
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")  # load tokenizer

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a given text using the loaded tokenizer.

    Args:
        text (str): The input text string to be tokenized.

    Returns:
        int: The number of tokens in the text.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))

def chunk_text(
    section_text: str,
    section_heading: str = "",
    max_tokens: int = 512,
    max_chars: int = 1000,
    sentence_overlap: int = 2
) -> List[Dict]:
    """
    Split text into token-limited, overlapping chunks.

    Each chunk attempts to stay within both token and character limits.
    Sentence boundaries are preserved where possible, and an adjustable
    number of overlapping sentences are retained between adjacent chunks
    to maintain context continuity.

    Args:
        section_text (str): The text body to be split into chunks.
        section_heading (str, optional): A heading or context prefix whose
            tokens count toward the token limit. Defaults to "".
        max_tokens (int, optional): Maximum token count per chunk.
            Defaults to 512.
        max_chars (int, optional): Maximum character count per chunk.
            Defaults to 1000.
        sentence_overlap (int, optional): Number of overlapping sentences
            to keep between chunks. Defaults to 2.

    Returns:
        List[Dict]: A list of dictionaries, each with keys:
            - "section_heading": heading applied to that section
            - "text": the chunked text body
    """
    heading_tokens = count_tokens(section_heading)
    token_budget = max_tokens - heading_tokens

    # if heading uses all the tokens, reset it
    if token_budget <= 0:
        section_heading = ""
        token_budget = max_tokens

    sentences = sent_tokenize(section_text)  # split text into sentences
    chunks = []  # store chunks
    current_chunk = []  # holds current group of sentences
    current_tokens = 0
    current_chars = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)  # count tokens in sentence
        sentence_chars = len(sentence)  # count characters in sentence

        # if a sentence is too long, shorten it to fit within token/character limits
        if sentence_tokens > token_budget or sentence_chars > max_chars:
            sentence = tokenizer.decode(tokenizer.encode(sentence)[:token_budget])
            sentence_tokens = count_tokens(sentence)
            sentence_chars = len(sentence)

        # check if adding this sentence would exceed the limits
        if (current_tokens + sentence_tokens > token_budget) or (current_chars + sentence_chars > max_chars):
            if current_chunk:
                # combine sentences into a single chunk
                chunk_text_combined = " ".join(current_chunk)
                chunks.append({
                    "section_heading": section_heading,
                    "text": chunk_text_combined.strip()
                })
                # keep some overlap from the end of the previous chunk
                overlap = current_chunk[-sentence_overlap:] if sentence_overlap > 0 else []
                current_chunk = list(overlap)
                current_tokens = sum(count_tokens(s) for s in current_chunk)
                current_chars = sum(len(s) for s in current_chunk)

        # add sentence to the current chunk
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
        current_chars += sentence_chars

    # add the last chunk if anything remains
    if current_chunk:
        chunk_text_combined = " ".join(current_chunk)
        chunks.append({
            "section_heading": section_heading,
            "text": chunk_text_combined.strip()
        })

    return chunks  # return all chunks

