# get_response.py

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
import time

from src.best_chunk import get_top_chunks
from src.config import EMBEDDING_MODEL_NAME

warnings.filterwarnings("ignore", category=RuntimeWarning, module='sklearn')

# Set up model and tokenizer
LLM_NAME = "microsoft/Phi-4-mini-instruct"
print(f"Loading {LLM_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
)


def build_prompt(context: str, question: str) -> str:
    """Build a chat prompt using system instructions, context, and the user question."""
    chat_prompt = [
        {
            "role": "system",
            "content": (
                "You are a legal assistant. "
                "Answer the user's question using only the provided context. "
                "If the answer is not in the context, say 'I do not know.' "
                "Keep responses clear and concise (max 4 sentences). "
                "Respond in the same language as the question."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n---\n"
                f"Now here is the question you need to answer.\n\n"
                f"Question: {question}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)


def generate_response(prompt: str) -> str:
    """Run inference using the selected LLM and return the generated text."""
    start = time.time()
    output = READER_LLM(prompt)
    end = time.time()
    response = output[0]["generated_text"].strip()
    print(f"\nResponse generated in {end - start:.2f} seconds.\n")
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual RAG QA using Phi-4.")
    parser.add_argument("--country", type=str, required=True, help="Country filter.")
    parser.add_argument("--query", type=str, required=True, help="User query.")
    parser.add_argument("--db_path", type=str, required=True, help="Path to Chroma DB.")
    parser.add_argument("--show_chunks", action="store_true", help="Show retrieved chunks.")
    args = parser.parse_args()

    print(f"\nSearching for: country='{args.country}', query='{args.query}'")

    try:
        top_chunks = get_top_chunks(
            country=args.country,
            query=args.query,
            chroma_path=args.db_path,
            top_k=5
        )
    except Exception as e:
        print(f"Failed to retrieve chunks: {e}")
        exit()

    if not top_chunks:
        print("No relevant chunks found.")
        exit()

    if args.show_chunks:
        print("\n--- Top Retrieved Chunks ---")
        for i, (score, cos_score, text, meta) in enumerate(top_chunks, 1):
            print(f"Rank {i} | Score: {score:.4f} | Cosine: {cos_score:.4f}")
            print(f"Text: {text}\nMetadata: {meta}\n{'-'*80}")

    print(f"\nGenerating Answer using {LLM_NAME}...\n")
    context = "\n\n".join([text for (_, _, text, _) in top_chunks])
    prompt = build_prompt(context=context, question=args.query)
    answer = generate_response(prompt)

    print("\nFinal Answer:\n")
    print(answer)






# Example
# python get_response.py --country paraguay --query "How is the National Seeds Council (CONASE) established and who comprises its membership?" --db_path processed/chroma_db
# python get_response.py --country albania --query "nsi and upov stand for?" --db_path processed/chroma_db --show_chunks
