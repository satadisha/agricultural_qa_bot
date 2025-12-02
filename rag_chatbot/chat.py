import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
import time
import json

from src.best_chunk import get_top_chunks
from src.config import EMBEDDING_MODEL_NAME, LLM_NAME

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

# load model
LLM_NAME = LLM_NAME
print(f"Loading {LLM_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    device_map='auto',
    torch_dtype=torch.float16,
)

READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=False,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=200,
)


def build_prompt(context, question):
    """Build a chat-style prompt containing system instructions, context, and the user question."""
    prompt_parts = [
        {
            "role": "system",
            "content": (
                "You are a helpful legal assistant. "
                "Answer the user's question using only the provided context. "
                "Keep answers clear and concise (max 5 sentences). "
                "If the information is not in the context, say 'I do not know.' "
                "Reply in the same language as the question."
            )
        }
    ]

    prompt_parts.append({
        "role": "user",
        "content": f"Context:\n{context}\n---\nQuestion: {question}"
    })

    return tokenizer.apply_chat_template(prompt_parts, tokenize=False, add_generation_prompt=True)


def generate_response(prompt):
    """Generate a model response and log the response time."""
    start = time.time()
    output = READER_LLM(prompt)
    end = time.time()
    response = output[0]["generated_text"].strip()
    print(f"\nResponse generated in {end - start:.2f} seconds.\n")
    return response


def chat_loop(country, chroma_path):
    """Interactive chat loop for querying country-specific regulatory chunks."""
    print(f"Country context: {country}")
    chat_history = []

    while True:
        user_query = input("\nAsk a question (or type 'exit' to quit):\n> ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        try:
            top_chunks = get_top_chunks(
                country=country,
                query=user_query,
                chroma_path=chroma_path,
                top_k=5,
            )
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            continue

        if not top_chunks:
            print("No relevant chunks found.")
            continue

        context = "\n\n".join([text for (_, _, text, _) in top_chunks])
        prompt = build_prompt(context=context, question=user_query)
        answer = generate_response(prompt)

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chat interface using a local LLM and Chroma context retrieval.")
    parser.add_argument("--country", type=str, required=True, help="Country to filter context.")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the Chroma DB directory.")
    args = parser.parse_args()

    chat_loop(country=args.country, chroma_path=args.db_path)

# run
# python chat.py --country albania --db_path processed/chroma_db
