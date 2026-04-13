"""
tools/
------
Anomaly detection tool implementations for the BRS agent.

Each tool is a standalone module that accepts a pandas DataFrame
and stem string, performs analysis, and returns a structured dict.

Tools:
    temporal_analysis    — A1: review volume over time + spike detection
    reviewer_profiling   — A2: credibility tier distribution + clustering
    content_similarity   — A3: TF-IDF similarity across review texts
    owner_response       — A4: copy-paste detection + timing stats
    run_python           — open-ended: execute arbitrary Python code
"""

from tools.temporal_analysis import temporal_analysis
from tools.reviewer_profiling import reviewer_profiling
from tools.content_similarity import content_similarity
from tools.owner_response import owner_response_analysis
from tools.run_python import run_python

__all__ = [
    "temporal_analysis",
    "reviewer_profiling",
    "content_similarity",
    "owner_response_analysis",
    "run_python",
]