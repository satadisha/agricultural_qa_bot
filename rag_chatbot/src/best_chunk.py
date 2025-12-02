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
    Compute how many predefined important keywords appear in both the document
    text and the query text.

    This is used as a heuristic boost during reranking to favor chunks that share
    meaningful terminology with the user’s query.

    Args:
        text (str): A candidate document chunk.
        query (str): The user’s query text.

    Returns:
        int: Count of keywords from IMPORTANT_KEYWORDS that appear in both
             `text` and `query`.
    """
    return sum(1 for kw in IMPORTANT_KEYWORDS if kw.lower() in text.lower() and kw.lower() in query.lower())

def extract_article_number_from_query(query: str) -> str:
    """
    Detect and normalize any 'Article N' reference found in the user’s query.

    If the query includes a phrase such as "Article 5" or "article 12", this
    helper extracts and returns the canonical lowercase form "article 5". This
    is later used to give a bonus score when a chunk’s subheading matches the
    same article number.

    Args:
        query (str): Query text that may contain an article reference.

    Returns:
        str | None: Normalized form "article <number>" if found; otherwise None.
    """
    match = re.search(r'\barticle\s+(\d+)', query.lower())
    return f"article {match.group(1)}" if match else None

def rerank_chunks(results, query_embedding, query_text):
    """
    Rerank ChromaDB query results using cosine similarity and additional heuristics.

    The ranking procedure combines:
        - Base semantic similarity (cosine)
        - Keyword overlap bonus
        - Exact 'article N' subheading match bonus

    The goal is to surface chunks that are not only semantically relevant but
    also structurally aligned with the user's intent (especially legal article
    references).

    Args:
        results (dict): Raw output from ChromaDB `collection.query(...)`. Must
            contain keys: "documents", "embeddings", and "metadatas".
        query_embedding (np.ndarray): Vector embedding of the query.
        query_text (str): Original query string (used for keyword and article
            extraction).

    Returns:
        list[tuple[float, float, str, dict]]:
            A sorted list where each entry contains:
                (final_score, base_cosine_score, document_text, metadata)
            Sorted by final_score descending.
    """
    documents = results["documents"][0]
    embeddings = np.array(results["embeddings"][0])
    metadatas = results["metadatas"][0]

    query_vec = np.array(query_embedding).reshape(1, -1)
    base_scores = cosine_similarity(query_vec, embeddings)[0]

    article_match = extract_article_number_from_query(query_text)

    reranked = []
    for score, doc, meta in zip(base_scores, documents, metadatas):
        bonus = 0.0

        # Keyword-based heuristic boost
        bonus += keyword_overlap_score(doc, query_text) * KEYWORD_BONUS_WEIGHT

        # Article number match boost
        sub_heading = meta.get("sub_heading", "").lower()
        if article_match and article_match in sub_heading:
            bonus += EXACT_ARTICLE_MATCH_BONUS

        final_score = score + bonus
        reranked.append((final_score, score, doc.strip(), meta))

    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked

def get_top_chunks(country, query, chroma_path, top_k=10):
    """
    Retrieve and rerank the top-K most relevant regulation chunks for a given
    country and query.

    This function:
        1. Loads the persistent ChromaDB database.
        2. Embeds the query using a SentenceTransformer model.
        3. Queries the "regulations" collection filtered by `country`.
        4. Reranks results using semantic similarity plus heuristic boosts.
        5. Returns the top K chunks.

    Args:
        country (str): Country identifier stored in metadata (e.g., "korea").
        query (str): User’s natural-language question.
        chroma_path (str | None): Path to the ChromaDB persistent directory.
        top_k (int): Number of reranked results to return.

    Returns:
        list[tuple[float, float, str, dict]]:
            Top-K reranked chunks, each formatted as:
            (final_score, base_cosine_score, document_text, metadata)

        Returns an empty list if no relevant documents are found.
    """
    print(f"\n🔍 Running multilingual chunk search for country: '{country}', query: '{query}'")

    client = PersistentClient(path=chroma_path)
    embedder = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    collection = client.get_or_create_collection(name="regulations", embedding_function=embedder)

    # Create query embedding
    query_embedding = embedder([query])[0]

    # Retrieve candidate chunks
    results = collection.query(
        query_texts=[query],
        n_results=top_k * 3,
        where={"country": country.lower()},
        include=["documents", "metadatas", "embeddings"]
    )

    if not results.get("documents") or not results["documents"][0]:
        print(" No documents found for this country.")
        return []

    # Return top-K reranked results
    reranked = rerank_chunks(results, query_embedding, query)
    return reranked[:top_k]

