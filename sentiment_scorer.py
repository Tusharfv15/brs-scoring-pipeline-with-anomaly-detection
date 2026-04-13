"""
sentiment_scorer.py
-------------------
Adds a `sentiment_score` column to a cleaned review .xlsx file.

Batches reviews in groups of 20, sends each batch to GPT-4o mini
using structured outputs (Pydantic), and writes scores back to the file.

Usage:
    python sentiment_scorer.py excel/pita_shree_handicraft_udaipur.xlsx

Options:
    --batch-size   Number of reviews per API call (default: 20)
    --col          Review text column name (default: auto-detect)
    --sheet        Sheet index or name (default: 0)
"""

import sys
import argparse
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, field_validator
from openai import OpenAI
from dotenv import load_dotenv

from utils import load_places_json, write_back_excel

load_dotenv(override=True)

client = OpenAI()
MODEL  = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ReviewSentiment(BaseModel):
    index:     int
    sentiment: Literal[-1, 0, 1]

    @field_validator("sentiment")
    @classmethod
    def must_be_valid(cls, v):
        if v not in {-1, 0, 1}:
            raise ValueError(f"Sentiment must be -1, 0 or 1. Got {v}")
        return v


class BatchSentimentResponse(BaseModel):
    results: list[ReviewSentiment]


# ---------------------------------------------------------------------------
# System prompt builder (uses business context from places JSON)
# ---------------------------------------------------------------------------

def build_system_prompt(name: str, address: str, category: str = "Unknown") -> str:
    return f"""You are an expert sentiment analyser specialising in customer reviews for local businesses.

You are analysing reviews for the following business:
  Name     : {name}
  Address  : {address}
  Category : {category}

Use this context to better interpret vague or ambiguous reviews.
The business category helps disambiguate sentiment — for example,
"decent prices" may be neutral for a restaurant but positive for a
premium retail store.

Scoring scale:
  +1 = positive  (satisfied, praises the business)
   0 = neutral, mixed, unclear, or off-topic
  -1 = negative  (dissatisfied, mentions problems or warns others)

Rules:
  - Score based on review text only — do not infer beyond what is written
  - Reviews in any language are valid — score them as written
"""


# ---------------------------------------------------------------------------
# Auto-detect review text column
# ---------------------------------------------------------------------------

REVIEW_COLUMN = "Customer Review"


# ---------------------------------------------------------------------------
# Score a single batch
# ---------------------------------------------------------------------------

def score_batch(batch: list[tuple[int, str]], system_prompt: str) -> list[ReviewSentiment]:
    """
    Send a batch of (index, text) pairs to GPT-4o mini.
    Returns a list of ReviewSentiment objects.
    """
    reviews_text = "\n".join(
        f'{idx}: "{text}"' for idx, text in batch
    )

    user_prompt = f"Score each of the following reviews:\n\n{reviews_text}"

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format=BatchSentimentResponse,
        temperature=0,
    )

    parsed = response.choices[0].message.parsed
    return parsed.results


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_file(file_path: Path, batch_size: int = 20, col: str = None, sheet=0):
    print(f"\n-> {file_path.name}")

    # Load business context from places_json
    biz         = load_places_json(file_path.stem)
    biz_name     = biz.get("name", file_path.stem.replace("_", " ").title())
    biz_address  = biz.get("address", "Unknown")
    biz_category = biz.get("category", "Unknown")
    if biz:
        print(f"  [INFO] Business context loaded: {biz_name} ({biz_category})")
    else:
        print(f"  [WARN] places_json/{file_path.stem}.json not found — using filename as business name.")

    system_prompt = build_system_prompt(biz_name, biz_address, biz_category)

    df = pd.read_excel(file_path, sheet_name=sheet)
    if df.empty:
        print("  [SKIP] Sheet is empty.")
        return

    # Resolve review text column
    if col:
        review_col = col
    else:
        review_col = REVIEW_COLUMN

    if review_col not in df.columns:
        print(f"  [ERROR] Column '{review_col}' not found. Available: {list(df.columns)}")
        return
    print(f"  [INFO] Using review column: '{review_col}'")

    # Skip if already scored
    if "sentiment_score" in df.columns:
        print("  [SKIP] 'sentiment_score' column already exists. Use --force to re-score.")
        return

    # Build index → text list (empty/NaN → score 0 directly)
    texts      = df[review_col].fillna("").astype(str).tolist()
    scores     = [None] * len(texts)

    # Pre-assign 0 for empty/very short texts
    to_score   = []
    for i, text in enumerate(texts):
        if len(text.strip()) < 3:
            scores[i] = 0
        else:
            to_score.append((i, text.strip()))

    print(f"  [INFO] {len(to_score)} reviews to score, {len(texts) - len(to_score)} pre-assigned 0")

    # Batch and score
    total_batches = (len(to_score) + batch_size - 1) // batch_size
    for batch_num in range(total_batches):
        batch = to_score[batch_num * batch_size : (batch_num + 1) * batch_size]
        print(f"  [BATCH {batch_num + 1}/{total_batches}] Scoring {len(batch)} reviews...")

        try:
            results = score_batch(batch, system_prompt)
            for r in results:
                scores[r.index] = r.sentiment
        except Exception as e:
            print(f"  [ERROR] Batch {batch_num + 1} failed: {e}")
            for idx, _ in batch:
                scores[idx] = 0

    # Fill any remaining None with 0
    scores = [s if s is not None else 0 for s in scores]

    # Write back to excel
    df["sentiment_score"] = scores
    write_back_excel(file_path, df, sheet)

    print(f"  [DONE] sentiment_score column added. Saved in-place.")
    print(f"  [SUMMARY] Distribution: { {s: scores.count(s) for s in sorted(set(scores))} }")





# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Add sentiment_score column to a review .xlsx file using GPT-4o mini.",
        epilog="""
Examples:
  python sentiment_scorer.py excel/pita_shree_handicraft_udaipur.xlsx
  python sentiment_scorer.py excel/pita_shree_handicraft_udaipur.xlsx --batch-size 10
  python sentiment_scorer.py excel/pita_shree_handicraft_udaipur.xlsx --col "Customer Review"
        """
    )
    parser.add_argument("filepath",
                        help="Path to the cleaned .xlsx file")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of reviews per API call (default: 20)")
    parser.add_argument("--col",        default=None,
                        help="Review text column name (default: auto-detect)")
    parser.add_argument("--sheet",      default=0,
                        help="Sheet index or name (default: 0)")
    parser.add_argument("--force",      action="store_true",
                        help="Re-score even if sentiment_score column already exists")

    args      = parser.parse_args()
    file_path = Path(args.filepath)
    sheet     = int(args.sheet) if isinstance(args.sheet, str) and args.sheet.isdigit() else args.sheet

    if not file_path.exists():
        print(f"ERROR: File '{file_path}' does not exist.")
        sys.exit(1)

    if args.force:
        df = pd.read_excel(file_path, sheet_name=sheet)
        if "sentiment_score" in df.columns:
            df.drop(columns=["sentiment_score"], inplace=True)
            write_back_excel(file_path, df, sheet)
            print("  [FORCE] Dropped existing sentiment_score column.")

    score_file(file_path, batch_size=args.batch_size, col=args.col, sheet=sheet)
    print("\nAll done.")


if __name__ == "__main__":
    main()