"""
tools/reviewer_profiling.py
---------------------------
A2 — Reviewer Profiling

Assigns each reviewer a credibility tier (Ghost/Low/Medium/High)
based on number_of_reviews, num_photos and is_local_guide.

Detects if low-credibility reviewers (Ghost + Low) are:
  1. Dominant overall (> 60% of all reviewers)
  2. Concentrated in a specific time window (> 70% in any bucket)

Returns structured dict with tier counts, clustering analysis and chart.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import relative_to_days

CHARTS_DIR = Path("outputs/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assign_credibility_tier(w: float) -> str:
    if w == 0.00:  return "Ghost"
    if w <= 0.33:  return "Low"
    if w <= 0.66:  return "Medium"
    return "High"


def compute_w_credibility(is_local_guide, number_of_reviews, num_photos) -> float:
    """
    Same formula as scorer.py — kept local to avoid circular imports.

    Rules:
      num_reviews = 0          → 0.0 (hard zero)
      is_local_guide = True    → +1
      num_reviews 15+          → +4 | 6-14 → +3 | 1-5 → +2
      num_photos >= 1          → +1
      W = raw / 6
    """
    try:
        n_reviews = int(number_of_reviews)
    except:
        return 0.0
    if n_reviews == 0:
        return 0.0

    raw = 0
    if is_local_guide is True or str(is_local_guide).strip().lower() in ("true", "1", "yes"):
        raw += 1
    if n_reviews >= 15:   raw += 4
    elif n_reviews >= 6:  raw += 3
    else:                 raw += 2

    try:
        n_photos = int(num_photos)
    except:
        n_photos = 0
    if n_photos >= 1:
        raw += 1

    return raw / 6


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def reviewer_profiling(df: pd.DataFrame, stem: str) -> dict:
    """
    Assigns credibility tiers and detects temporal clustering
    of low-credibility reviewers (Ghost + Low combined).

    Flag triggers when:
      - Overall: (Ghost + Low) / total > 60%
      - Per bucket: (Ghost + Low) / bucket_total > 70%

    Args:
        df   : review DataFrame
        stem : filename stem for chart naming

    Returns:
        dict with tier counts, low_cred_ratio, flag_overall,
        clustering list and chart path.
    """
    df = df.copy()

    # Compute W_credibility and tier per reviewer
    df["_w_cred"] = df.apply(
        lambda r: compute_w_credibility(
            r["is_local_guide"],
            r["number_of_reviews"],
            r["num_photos"],
        ), axis=1
    )
    df["_tier"]   = df["_w_cred"].apply(assign_credibility_tier)
    df["_days"]   = df["Review Time Line"].apply(
        lambda x: relative_to_days(str(x))
    )
    df["_bucket"] = df["_days"].apply(
        lambda d: f"{int(d // 30)}m ago" if d != float("inf") else "unknown"
    )

    total = len(df)

    # Tier counts
    tier_counts = df["_tier"].value_counts().to_dict()
    tiers = {}
    for t in ["Ghost", "Low", "Medium", "High"]:
        c = tier_counts.get(t, 0)
        tiers[t] = {"count": c, "pct": round(c / total * 100, 1)}

    low_cred_total = tiers["Ghost"]["count"] + tiers["Low"]["count"]
    low_cred_ratio = round(low_cred_total / total, 4)
    flag_overall   = low_cred_ratio > 0.60

    # One-shot accounts: reviewers whose only ever review is this one.
    # Strong fake signal in aggregate — real customers occasionally have
    # low review counts, but a campaign of one-shot accounts is anomalous.
    one_shot_count = int((df["number_of_reviews"] == 1).sum())
    one_shot_pct   = round(one_shot_count / total * 100, 1)
    one_shot_flag  = one_shot_pct > 30

    # Temporal clustering
    clustering = []
    for bucket, group in df.groupby("_bucket"):
        if bucket == "unknown":
            continue
        bucket_total  = len(group)
        ghost_count   = int((group["_tier"] == "Ghost").sum())
        low_count     = int((group["_tier"] == "Low").sum())
        low_cred_pct  = round((ghost_count + low_count) / bucket_total * 100, 1)
        flagged       = low_cred_pct > 70
        if flagged or bucket_total >= 3:
            clustering.append({
                "bucket"      : bucket,
                "total"       : int(bucket_total),
                "ghost_count" : ghost_count,
                "low_count"   : low_count,
                "low_cred_pct": low_cred_pct,
                "flagged"     : flagged,
            })

    def bucket_key(b):
        try:    return int(b["bucket"].replace("m ago", ""))
        except: return 9999
    clustering.sort(key=bucket_key)

    # Plot 1 — overall tier distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    tier_labels = ["Ghost", "Low", "Medium", "High"]
    tier_values = [tiers[t]["count"] for t in tier_labels]
    tier_colors = ["#E24B4A", "#EF9F27", "#378ADD", "#1D9E75"]

    axes[0].bar(tier_labels, tier_values, color=tier_colors)
    axes[0].set_title("Reviewer credibility distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(tier_values):
        axes[0].text(
            i, v + 0.3,
            f"{tiers[tier_labels[i]]['pct']}%",
            ha="center", fontsize=9
        )

    # Plot 2 — stacked bar per bucket
    if clustering:
        buckets    = [c["bucket"] for c in clustering]
        ghost_vals = [c["ghost_count"] for c in clustering]
        low_vals   = [c["low_count"] for c in clustering]
        med_high   = [
            c["total"] - c["ghost_count"] - c["low_count"]
            for c in clustering
        ]
        axes[1].bar(buckets, ghost_vals, color="#E24B4A", label="Ghost")
        axes[1].bar(
            buckets, low_vals, color="#EF9F27", label="Low",
            bottom=ghost_vals
        )
        axes[1].bar(
            buckets, med_high, color="#1D9E75", label="Med/High",
            bottom=[g + l for g, l in zip(ghost_vals, low_vals)]
        )
        axes[1].set_title("Credibility tier per time bucket")
        axes[1].set_ylabel("Count")
        axes[1].legend(fontsize=8)
        plt.setp(
            axes[1].xaxis.get_majorticklabels(),
            rotation=45, ha="right", fontsize=8
        )
    else:
        axes[1].text(
            0.5, 0.5, "No bucket data",
            ha="center", va="center",
            transform=axes[1].transAxes
        )

    plt.suptitle(
        f"Reviewer profiling — {stem.replace('_', ' ').title()}",
        fontsize=11
    )
    plt.tight_layout()

    chart_path = CHARTS_DIR / f"{stem}_reviewer_profile.png"
    plt.savefig(chart_path, dpi=130)
    plt.close()

    return {
        "total_reviewers"  : total,
        "tiers"            : tiers,
        "low_cred_ratio"   : low_cred_ratio,
        "flag_overall"     : flag_overall,
        "one_shot_reviewers": {"count": one_shot_count, "pct": one_shot_pct},
        "one_shot_flag"    : one_shot_flag,
        "clustering"       : clustering,
        "chart"            : str(chart_path),
    }