# best_chunk.py

import re
import numpy as np
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sklearn.metrics.pairwise import cosine_similarity
from src.config import EMBEDDING_MODEL_NAME

# -- Hyperparameters --
KEYWORD_BONUS_WEIGHT = 0.05  # small boost for keyword matches
EXACT_ARTICLE_MATCH_BONUS = 1.0  # larger boost for article number matches
IMPORTANT_KEYWORDS = [  # key words to prioritize in scoring
    "purpose", "goal", "objective", "aim",
    "definition", "refers to", "means",
    "foreign", "breeder", "right", "application",
    "article", "law", "protection", "variety",
    "nsi", "institution", "authority", "registration"
]

def keyword_overlap_score(text: str, query: str) -> int:
    # count how many important keywords appear in both the text and the query
    return sum(1 for kw in IMPORTANT_KEYWORDS if kw.lower() in text.lower() and kw.lower() in query.lower())

def extract_article_number_from_query(query: str) -> str:
    # extract an article number like "Article 5" from the query
    match = re.search(r'\barticle\s+(\d+)', query.lower())
    return f"article {match.group(1)}" if match else None

def rerank_chunks(results, query_embedding, query_text):
    # rerank retrieved chunks using cosine similarity and keyword/article boosts
    documents = results["documents"][0]
    embeddings = np.array(results["embeddings"][0])
    metadatas = results["metadatas"][0]

    query_vec = np.array(query_embedding).reshape(1, -1)
    base_scores = cosine_similarity(query_vec, embeddings)[0]  # base similarity scores

    article_match = extract_article_number_from_query(query_text)  # find article number in query if any

    reranked = []
    for score, doc, meta in zip(base_scores, documents, metadatas):
        bonus = 0.0

        # small bonus for keyword overlap
        bonus += keyword_overlap_score(doc, query_text) * KEYWORD_BONUS_WEIGHT

        # big bonus if the article number matches
        sub_heading = meta.get("sub_heading", "").lower()
        if article_match and article_match in sub_heading:
            bonus += EXACT_ARTICLE_MATCH_BONUS

        final_score = score + bonus  # combine base score and bonuses
        reranked.append((final_score, score, doc.strip(), meta))

    # sort chunks from best to worst based on final score
    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked

def get_top_chunks(country, query, chroma_path, top_k=10):
    # main function to search ChromaDB and get the best chunks for a query
    print(f"\n🔍 Running multilingual chunk search for country: '{country}', query: '{query}'")

    client = PersistentClient(path=chroma_path)
    embedder = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    collection = client.get_or_create_collection(name="regulations", embedding_function=embedder)

    # create embedding for the query
    query_embedding = embedder([query])[0]

    # search database for matching chunks from the given country
    results = collection.query(
        query_texts=[query],
        n_results=top_k * 3,  # fetch more results to allow better reranking
        where={"country": country.lower()},
        include=["documents", "metadatas", "embeddings"]
    )

    # if nothing is found, print a warning
    if not results.get("documents") or not results["documents"][0]:
        print("⚠️ No documents found for this country.")
        return []

    # rerank and return the top K chunks
    reranked = rerank_chunks(results, query_embedding, query)
    return reranked[:top_k]

