"""
tools/content_similarity.py
---------------------------
A3 — Content Similarity

Detects suspiciously similar review texts using TF-IDF vectorization
and cosine similarity.

Flag triggers when >15% of unique reviewers are involved in similar pairs
(similarity threshold = 0.75).

No chart — structured similar pairs list is sufficient for LLM reasoning.
"""

from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def content_similarity(df: pd.DataFrame, stem: str) -> dict:
    """
    Detects similar review texts using TF-IDF + cosine similarity.

    For every pair (i, j):
      if cosine_similarity(text_i, text_j) >= 0.75 → flagged as similar pair

    Flag triggers when:
      len(unique reviewers in similar pairs) / total_reviews > 0.15

    Args:
        df   : review DataFrame
        stem : filename stem (unused, kept for consistent interface)

    Returns:
        dict with total_reviews, similar_pairs, flagged_reviewers,
        flagged_pct and flag.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    SIMILARITY_THRESHOLD = 0.75

    df          = df.copy()
    df["_text"] = df["Customer Review"].fillna("").astype(str).str.strip()
    valid       = df[df["_text"].str.len() > 3].reset_index(drop=True)
    total       = len(df)
    valid_count = len(valid)

    if valid_count < 2:
        return {
            "total_reviews"       : total,
            "valid_reviews"       : valid_count,
            "similar_pairs"       : [],
            "flagged_reviewers"   : 0,
            "flagged_pct"         : 0.0,
            "flag"                : False,
            "note"                : "Not enough reviews to compute similarity.",
        }

    # TF-IDF vectorization
    vectorizer   = TfidfVectorizer(min_df=1, stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(valid["_text"])
    sim_matrix   = cos_sim(tfidf_matrix)

    # Find similar pairs and track unique reviewers involved.
    # Only keep the MAX_PAIRS highest-scoring pairs to prevent token explosion
    # from O(N²) pair lists on large datasets (250 reviews → 31k pairs).
    MAX_PAIRS = 25
    similar_pairs = []
    flagged_set   = set()

    for i in range(valid_count):
        for j in range(i + 1, valid_count):
            score = float(sim_matrix[i][j])
            if score >= SIMILARITY_THRESHOLD:
                name_a = str(valid.loc[i, "Name"])
                name_b = str(valid.loc[j, "Name"])
                similar_pairs.append({
                    "reviewer_a": name_a,
                    "reviewer_b": name_b,
                    "similarity": round(score, 4),
                    "text_a"    : valid.loc[i, "_text"][:120],
                    "text_b"    : valid.loc[j, "_text"][:120],
                })
                flagged_set.add(name_a)
                flagged_set.add(name_b)

    flagged_count    = len(flagged_set)
    flag             = flagged_count / total > 0.15
    total_pairs      = len(similar_pairs)

    # Sort by similarity descending, then cap — agent sees most suspicious pairs
    similar_pairs.sort(key=lambda p: p["similarity"], reverse=True)
    truncated = total_pairs > MAX_PAIRS
    similar_pairs = similar_pairs[:MAX_PAIRS]

    return {
        "total_reviews"       : total,
        "valid_reviews"       : valid_count,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "total_similar_pairs" : total_pairs,
        "similar_pairs"       : similar_pairs,
        "pairs_truncated"     : truncated,
        "flagged_reviewers"   : flagged_count,
        "flagged_pct"         : round(flagged_count / total * 100, 1),
        "flag"                : flag,
    }