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
     """
    Count overlap of important keywords between a text chunk and the query.

    Args:
        text (str): The candidate document text.
        query (str): The user’s query text.

    Returns:
        int: Number of IMPORTANT_KEYWORDS present in both `text` and `query`.
    """
    return sum(1 for kw in IMPORTANT_KEYWORDS if kw.lower() in text.lower() and kw.lower() in query.lower())

def extract_article_number_from_query(query: str) -> str:
    """
    Extract a normalized article reference (e.g., 'article 5') from the query.

    Args:
        query (str): Query text that may contain an article reference like 'Article 5'.

    Returns:
        str | None: A lowercase string 'article N' if found, otherwise None.
    """
    match = re.search(r'\barticle\s+(\d+)', query.lower())
    return f"article {match.group(1)}" if match else None

def rerank_chunks(results, query_embedding, query_text):
    """
    Rerank retrieved chunks by cosine similarity plus heuristic boosts.

    The final score is:
        final_score = cosine_similarity(query_vec, doc_vec) + bonuses
    where bonuses include:
        - keyword overlap (scaled by KEYWORD_BONUS_WEIGHT)
        - exact match of 'article N' in the chunk's sub_heading (EXACT_ARTICLE_MATCH_BONUS)

    Args:
        results (dict): Output from `collection.query(...)` including keys
            'documents', 'metadatas', and 'embeddings'. Each is a list with
            results for the (single) query at index 0.
        query_embedding (np.ndarray): Embedding vector for the query (1D).
        query_text (str): Raw query text (for keywords/article extraction).

    Returns:
        list[tuple[float, float, str, dict]]: A list of tuples:
            (final_score, base_cosine, document_text, metadata)
        sorted from best to worst by `final_score`.
    """
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
    """
    Query ChromaDB for a given country + query, rerank, and return top-K chunks.

    This function:
        1) Initializes a PersistentClient and sentence-transformer embedder.
        2) Queries the 'regulations' collection filtered by country.
        3) Reranks results using cosine similarity + heuristic boosts.
        4) Returns the top `top_k` tuples.

    Args:
        country (str): Country code/name stored in metadata (e.g., 'korea').
        query (str): User’s natural-language query.
        chroma_path (str | None): Filesystem path to the ChromaDB directory.
        top_k (int): Number of top chunks to return after reranking. Default is 10.

    Returns:
        list[tuple[float, float, str, dict]]: Top-K tuples of the form
            (final_score, base_cosine, document_text, metadata).
        If no results are found, returns an empty list.
    """
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

