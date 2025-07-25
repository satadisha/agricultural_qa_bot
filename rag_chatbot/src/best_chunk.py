# best_chunk.py

import re
import numpy as np
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sklearn.metrics.pairwise import cosine_similarity
from src.config import EMBEDDING_MODEL_NAME

# -- Hyperparameters --
KEYWORD_BONUS_WEIGHT = 0.05
EXACT_ARTICLE_MATCH_BONUS = 1.0
IMPORTANT_KEYWORDS = [
    "purpose", "goal", "objective", "aim",
    "definition", "refers to", "means",
    "foreign", "breeder", "right", "application",
    "article", "law", "protection", "variety",
    "nsi", "institution", "authority", "registration"
]

def keyword_overlap_score(text: str, query: str) -> int:
    return sum(1 for kw in IMPORTANT_KEYWORDS if kw.lower() in text.lower() and kw.lower() in query.lower())

def extract_article_number_from_query(query: str) -> str:
    match = re.search(r'\barticle\s+(\d+)', query.lower())
    return f"article {match.group(1)}" if match else None

def rerank_chunks(results, query_embedding, query_text):
    documents = results["documents"][0]
    embeddings = np.array(results["embeddings"][0])
    metadatas = results["metadatas"][0]

    query_vec = np.array(query_embedding).reshape(1, -1)
    base_scores = cosine_similarity(query_vec, embeddings)[0]

    article_match = extract_article_number_from_query(query_text)

    reranked = []
    for score, doc, meta in zip(base_scores, documents, metadatas):
        bonus = 0.0

        # Boost for keyword relevance
        bonus += keyword_overlap_score(doc, query_text) * KEYWORD_BONUS_WEIGHT

        # Boost for article number match
        sub_heading = meta.get("sub_heading", "").lower()
        if article_match and article_match in sub_heading:
            bonus += EXACT_ARTICLE_MATCH_BONUS

        final_score = score + bonus
        reranked.append((final_score, score, doc.strip(), meta))

    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked

def get_top_chunks(country, query, chroma_path, top_k=10):
    print(f"\n🔍 Running multilingual chunk search for country: '{country}', query: '{query}'")

    client = PersistentClient(path=chroma_path)
    embedder = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    collection = client.get_or_create_collection(name="regulations", embedding_function=embedder)

    query_embedding = embedder([query])[0]

    results = collection.query(
        query_texts=[query],
        n_results=top_k * 3,  # Fetch more to allow better reranking
        where={"country": country.lower()},
        include=["documents", "metadatas", "embeddings"]
    )

    if not results.get("documents") or not results["documents"][0]:
        print("⚠️ No documents found for this country.")
        return []

    reranked = rerank_chunks(results, query_embedding, query)
    return reranked[:top_k]
