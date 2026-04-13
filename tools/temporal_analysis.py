"""
tools/temporal_analysis.py
--------------------------
A1 — Temporal Analysis

Groups reviews by month bucket, generates a bar chart, computes
a rolling baseline and detects spike months (> mean + 2σ).

Returns structured dict with month_buckets, spikes_detected and chart path.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import relative_to_days

CHARTS_DIR = Path("outputs/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def temporal_analysis(df: pd.DataFrame, stem: str) -> dict:
    """
    Groups reviews by month bucket and detects spikes.

    Spike detection: any bucket with count > mean + 2σ is flagged.
    Red bars on chart = spike months, green = normal.

    Args:
        df   : review DataFrame
        stem : filename stem for chart naming

    Returns:
        dict with total_reviews, month_buckets, baseline_mean,
        spikes_detected and chart path.
    """
    df = df.copy()
    df["_days"] = df["Review Time Line"].apply(
        lambda x: relative_to_days(str(x))
    )

    def days_to_month_bucket(days):
        if days == float("inf"):
            return "unknown"
        return f"{int(days // 30)}m ago"

    df["_bucket"]  = df["_days"].apply(days_to_month_bucket)
    counts         = df["_bucket"].value_counts()
    known          = {k: v for k, v in counts.items() if k != "unknown"}
    unknown_count  = counts.get("unknown", 0)

    def bucket_sort_key(b):
        try:    return int(b.replace("m ago", ""))
        except: return 9999

    sorted_buckets = sorted(known.keys(), key=bucket_sort_key)
    sorted_counts  = [known[b] for b in sorted_buckets]

    # Spike detection
    if len(sorted_counts) > 2:
        mean_c = float(np.mean(sorted_counts))
        std_c  = float(np.std(sorted_counts))
        spikes = [
            {
                "bucket"          : b,
                "count"           : int(c),
                "times_above_mean": round(c / mean_c, 2),
            }
            for b, c in zip(sorted_buckets, sorted_counts)
            if c > mean_c + 2 * std_c
        ]
    else:
        mean_c = float(np.mean(sorted_counts)) if sorted_counts else 0.0
        std_c  = 0.0
        spikes = []

    # Plot
    colors = [
        "#E24B4A" if (std_c > 0 and c > mean_c + 2 * std_c) else "#1D9E75"
        for c in sorted_counts
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(sorted_buckets, sorted_counts, color=colors)
    ax.axhline(
        y=mean_c, color="#888780", linestyle="--",
        linewidth=1, label=f"baseline mean ({mean_c:.1f})"
    )
    ax.set_xlabel("Months ago")
    ax.set_ylabel("Review count")
    ax.set_title(
        f"Review volume over time — {stem.replace('_', ' ').title()}\n"
        f"Red bars = spike (> mean + 2σ)"
    )
    ax.legend(fontsize=8)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    chart_path = CHARTS_DIR / f"{stem}_temporal.png"
    plt.savefig(chart_path, dpi=130)
    plt.close()

    return {
        "total_reviews"   : len(df),
        "month_buckets"   : dict(zip(sorted_buckets, sorted_counts)),
        "unknown_timeline": int(unknown_count),
        "baseline_mean"   : round(mean_c, 2),
        "spikes_detected" : spikes,
        "flag"            : len(spikes) > 0,
        "chart"           : str(chart_path),
    }