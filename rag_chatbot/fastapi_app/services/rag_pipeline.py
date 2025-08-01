import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # Adds rag_chatbot/ to import path

from src.best_chunk import get_top_chunks
from chat import build_prompt, generate_response
from transformers import AutoTokenizer

# Load tokenizer globally for reuse
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")

# Optional: Store chat memory per session (can be improved with state/session later)
chat_memory = {}

def get_chat_response(query: str, country: str) -> str:
    chroma_path = "processed/chroma_db"  # default path — you can make this configurable
    history = chat_memory.get(country, [])

    try:
        top_chunks = get_top_chunks(
            country=country,
            query=query,
            chroma_path=chroma_path,
            top_k=7,
        )
    except Exception as e:
        return f"Error retrieving context: {str(e)}"

    if not top_chunks:
        return "No relevant chunks found."

    context = "\n\n".join([text for (_, _, text, _) in top_chunks])
    prompt = build_prompt(context=context, question=query)
    answer = generate_response(prompt)

    history.append({"user": query, "bot": answer})
    chat_memory[country] = history

    return answer
