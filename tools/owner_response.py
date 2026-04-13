"""
tools/owner_response.py
-----------------------
A4 — Owner Response Analysis

A4a — Copy-paste detection:
  Uses TF-IDF cosine similarity on owner response texts.
  Flag triggers when >50% of reviews received a near-identical response.
  Similarity threshold = 0.85 (higher than A3 since owner responses
  naturally use similar polite language).

A4b — Response timing analysis:
  Computes gap between review date and owner response date.
  Descriptive stats only — not used for flagging.
  Provides context for the LLM to reason over.

No chart — structured output is sufficient for LLM reasoning.
"""

from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import relative_to_days


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def owner_response_analysis(df: pd.DataFrame, stem: str) -> dict:
    """
    A4a: Detects copy-paste owner responses using TF-IDF + cosine similarity.
    A4b: Computes descriptive response timing stats (context only, no flag).

    Flag triggers when:
      len(unique reviews with copy-paste response) / total_responded > 0.50

    Args:
        df   : review DataFrame
        stem : filename stem (unused, kept for consistent interface)

    Returns:
        dict with copy_paste results and timing stats.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    SIMILARITY_THRESHOLD = 0.85

    df = df.copy()

    # Filter to reviews that have an owner response
    df["_response"] = df["Owner Response"].fillna("").astype(str).str.strip()
    responded       = df[df["_response"].str.len() > 3].reset_index(drop=True)
    total           = len(df)
    total_responded = len(responded)

    # -----------------------------------------------------------------------
    # A4a — Copy-paste detection
    # -----------------------------------------------------------------------

    if total_responded < 2:
        copy_paste = {
            "total_responded" : total_responded,
            "similar_pairs"   : [],
            "flagged_reviews" : 0,
            "flagged_pct"     : 0.0,
            "flag"            : False,
            "note"            : "Not enough responses to compute similarity.",
        }
    else:
        vectorizer   = TfidfVectorizer(min_df=1, stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(responded["_response"])
        sim_matrix   = cos_sim(tfidf_matrix)

        # Only keep the MAX_PAIRS highest-scoring pairs to prevent token explosion
        # from O(N²) pair lists (200 responses → ~20k pairs).
        MAX_PAIRS   = 25
        similar_pairs = []
        flagged_set   = set()  # tracks review indices with copy-paste responses

        for i in range(total_responded):
            for j in range(i + 1, total_responded):
                score = float(sim_matrix[i][j])
                if score >= SIMILARITY_THRESHOLD:
                    similar_pairs.append({
                        "reviewer_a"  : str(responded.loc[i, "Name"]),
                        "reviewer_b"  : str(responded.loc[j, "Name"]),
                        "similarity"  : round(score, 4),
                        "response_a"  : responded.loc[i, "_response"][:120],
                        "response_b"  : responded.loc[j, "_response"][:120],
                    })
                    flagged_set.add(i)
                    flagged_set.add(j)

        flagged_count = len(flagged_set)
        flag          = flagged_count / total_responded > 0.50
        total_pairs   = len(similar_pairs)

        # Sort by similarity descending, then cap — agent sees most suspicious pairs
        similar_pairs.sort(key=lambda p: p["similarity"], reverse=True)
        truncated   = total_pairs > MAX_PAIRS
        similar_pairs = similar_pairs[:MAX_PAIRS]

        copy_paste = {
            "total_responded"     : total_responded,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "total_similar_pairs" : total_pairs,
            "similar_pairs"       : similar_pairs,
            "pairs_truncated"     : truncated,
            "flagged_reviews"     : flagged_count,
            "flagged_pct"         : round(flagged_count / total_responded * 100, 1),
            "flag"                : flag,
        }

    # -----------------------------------------------------------------------
    # A4b — Response timing analysis (descriptive only, no flag)
    # -----------------------------------------------------------------------

    timing_gaps = []
    for _, row in responded.iterrows():
        review_days  = relative_to_days(str(row.get("Review Time Line", "")))
        response_days = relative_to_days(str(row.get("Owner Response Timeline", "")))
        if review_days != float("inf") and response_days != float("inf"):
            gap = abs(review_days - response_days)
            timing_gaps.append(gap)

    if timing_gaps:
        timing = {
            "responses_with_timing": len(timing_gaps),
            "mean_gap_days"        : round(float(np.mean(timing_gaps)), 2),
            "std_gap_days"         : round(float(np.std(timing_gaps)), 2),
            "min_gap_days"         : round(float(np.min(timing_gaps)), 2),
            "max_gap_days"         : round(float(np.max(timing_gaps)), 2),
            "note"                 : "Descriptive only — not used for flagging.",
        }
    else:
        timing = {
            "responses_with_timing": 0,
            "note"                 : "Could not compute timing gaps from available data.",
        }

    return {
        "total_reviews" : total,
        "copy_paste"    : copy_paste,
        "timing"        : timing,
    }