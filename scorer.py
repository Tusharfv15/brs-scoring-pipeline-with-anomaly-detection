"""
scorer.py
---------
BRS (Business Reliability Score) scoring pipeline.

Computes M1 through M5 and produces a final score (0-100).

Inputs:
    excel/<stem>.xlsx        — cleaned reviews with sentiment_score column
    places_json/<stem>.json  — Places API data

Usage:
    python scorer.py excel/pita_shree_handicraft_udaipur.xlsx
"""

import sys
import math
import logging
import argparse
from pathlib import Path

import pandas as pd

from utils import (
    relative_to_days,
    recency_multiplier,
    load_places_json,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-8s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

M1_LAMBDA = 0.5  # Normalisation denominator for M1 aggregation.
                 # Represents the expected mean score_i of a typical legitimate
                 # business. Lower = more generous, higher = more strict.
                 # Sensitivity analysis shows 0.5 is the optimal value —
                 # below 0.4 most businesses hit the cap, above 0.7 even
                 # strong businesses score low.


# ---------------------------------------------------------------------------
# M1 helpers
# ---------------------------------------------------------------------------

def compute_w_credibility(is_local_guide, number_of_reviews, num_photos) -> float:
    """
    Compute reviewer credibility weight.

    Rules:
      - number_of_reviews = 0 or unknown → hard zero (review ignored)
      - Local Guide                       → +1
      - number_of_reviews 15+             → +4 | 6-14 → +3 | 1-5 → +2
      - num_photos >= 1                   → +1

    Max raw = 6
    W_credibility = raw / 6  (range 0.0 to 1.0)
    """
    try:
        n_reviews = int(number_of_reviews)
    except (ValueError, TypeError):
        log.debug(f"    [W_CRED] number_of_reviews='{number_of_reviews}' unreadable → hard zero")
        return 0.0

    if n_reviews == 0:
        log.debug(f"    [W_CRED] number_of_reviews=0 → hard zero")
        return 0.0

    raw = 0

    if is_local_guide is True or str(is_local_guide).strip().lower() in ("true", "1", "yes"):
        raw += 1
        log.debug(f"    [W_CRED] local_guide=True → +1")

    if n_reviews >= 15:
        raw += 4
        log.debug(f"    [W_CRED] reviews={n_reviews} (≥15) → +4")
    elif n_reviews >= 6:
        raw += 3
        log.debug(f"    [W_CRED] reviews={n_reviews} (6-14) → +3")
    else:
        raw += 2
        log.debug(f"    [W_CRED] reviews={n_reviews} (1-5) → +2")

    try:
        n_photos = int(num_photos)
    except (ValueError, TypeError):
        n_photos = 0

    if n_photos >= 1:
        raw += 1
        log.debug(f"    [W_CRED] photos={n_photos} (≥1) → +1")

    w = raw / 6
    log.debug(f"    [W_CRED] raw={raw} → W_credibility={w:.4f}")
    return w


def compute_w_recency(review_timeline) -> float:
    """
    Compute recency weight from review timeline string.
    Returns 0.25 if string not recognised (with a warning log).
    """
    days = relative_to_days(str(review_timeline))

    if days == float("inf"):
        log.warning(
            f"    [W_RECENCY] Unrecognised timeline value: '{review_timeline}' "
            f"→ defaulting to 0.25 (3+ years bucket)"
        )
        days = 9999

    w = recency_multiplier(days)
    log.debug(f"    [W_RECENCY] '{review_timeline}' → {days} days → W_recency={w}")
    return w


# ---------------------------------------------------------------------------
# M1 — Weighted Review Score (0-55 pts)
# ---------------------------------------------------------------------------

def compute_m1(df: pd.DataFrame) -> tuple[float, dict]:
    """
    M1 = min((mean(score_i) / 0.5) * 55, 55)

    score_i = W_credibility * W_recency * sentiment_score

    Returns (m1_score, m1_detail) where m1_detail contains
    per-review breakdown for agent context.

    Raises ValueError if sentiment_score column is missing.
    """
    log.info("[M1] Computing weighted review score...")

    if "sentiment_score" not in df.columns:
        raise ValueError(
            "[M1] 'sentiment_score' column not found in excel. "
            "Run sentiment_scorer.py first before scoring."
        )

    required_cols = ["is_local_guide", "number_of_reviews", "num_photos",
                     "Review Time Line", "sentiment_score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[M1] Required column '{col}' not found in excel.")

    scores     = []
    per_review = []

    for idx, row in df.iterrows():
        log.debug(f"  [M1] Row {idx} — reviewer: {row.get('Name', 'unknown')}")

        w_cred    = compute_w_credibility(
                        row["is_local_guide"],
                        row["number_of_reviews"],
                        row["num_photos"]
                    )
        w_recency = compute_w_recency(row["Review Time Line"])

        try:
            sentiment = float(row["sentiment_score"])
        except (ValueError, TypeError):
            log.warning(f"  [M1] Row {idx}: invalid sentiment_score='{row['sentiment_score']}' → defaulting to 0")
            sentiment = 0.0

        score_i = w_cred * w_recency * sentiment
        log.debug(
            f"  [M1] Row {idx}: W_cred={w_cred:.3f} × W_rec={w_recency:.2f} "
            f"× S={sentiment} = {score_i:.4f}"
        )
        scores.append(score_i)
        per_review.append({
            "name"          : str(row.get("Name", f"row_{idx}")),
            "review_timeline": str(row.get("Review Time Line", "")),
            "w_credibility" : round(w_cred, 4),
            "w_recency"     : round(w_recency, 4),
            "sentiment"     : int(sentiment),
            "score_i"       : round(score_i, 4),
        })

    if not scores:
        log.warning("[M1] No reviews found → M1 = 0")
        return 0.0, {}

    mean_score = sum(scores) / len(scores)
    m1         = min((mean_score / M1_LAMBDA) * 55, 55)
    m1         = max(m1, 0.0)

    detail = {
        "lambda"        : M1_LAMBDA,
        "mean_score_i"  : round(mean_score, 4),
        "total_reviews" : len(scores),
        "per_review"    : per_review,
    }

    log.info(f"[M1] {len(scores)} reviews | mean_score={mean_score:.4f} → M1={m1:.2f}/55")
    return round(m1, 2), detail


# ---------------------------------------------------------------------------
# M2 — Scale & Rating Score (0-20 pts)
# ---------------------------------------------------------------------------

def compute_m2(biz: dict) -> float:
    """
    M2 = M2a + M2b

    M2a (13 pts) = ((avg_rating - 1) / 4) × 13
    M2b (7 pts)  = min((log10(total_rating_count) / log10(3000)) × 7, 7)

    Raises ValueError if required fields are missing or invalid.
    """
    log.info("[M2] Computing scale & rating score...")

    # --- M2a ---
    raw_rating = biz.get("average_rating", "N/A")
    if raw_rating == "N/A" or raw_rating is None:
        raise ValueError(
            "[M2] 'average_rating' not found in places JSON. "
            "Re-run sort_reviews_by_date.py to fetch Places API data."
        )
    try:
        avg_rating = float(raw_rating)
    except (ValueError, TypeError):
        raise ValueError(f"[M2] 'average_rating' value '{raw_rating}' is not a valid number.")

    if not (1.0 <= avg_rating <= 5.0):
        raise ValueError(f"[M2] 'average_rating' value {avg_rating} is out of range (1.0–5.0).")

    m2a = ((avg_rating - 1) / 4) * 13
    log.info(f"[M2a] avg_rating={avg_rating} → M2a={m2a:.2f}/13")

    # --- M2b ---
    raw_count = biz.get("total_rating_count", "N/A")
    if raw_count == "N/A" or raw_count is None:
        raise ValueError(
            "[M2] 'total_rating_count' not found in places JSON. "
            "Re-run sort_reviews_by_date.py to fetch Places API data."
        )
    try:
        total_count = int(raw_count)
    except (ValueError, TypeError):
        raise ValueError(f"[M2] 'total_rating_count' value '{raw_count}' is not a valid integer.")

    if total_count <= 0:
        raise ValueError(f"[M2] 'total_rating_count' value {total_count} must be greater than 0.")

    m2b = min((math.log10(total_count) / math.log10(3000)) * 7, 7)
    log.info(f"[M2b] total_rating_count={total_count} → M2b={m2b:.2f}/7")

    m2     = round(m2a + m2b, 2)
    detail = {
        "avg_rating"         : avg_rating,
        "total_rating_count" : total_count,
        "M2a"                : round(m2a, 2),
        "M2b"                : round(m2b, 2),
    }
    log.info(f"[M2] M2a={m2a:.2f} + M2b={m2b:.2f} → M2={m2:.2f}/20")
    return m2, detail


# ---------------------------------------------------------------------------
# M3 — Owner Engagement Score (0-15 pts)
# ---------------------------------------------------------------------------

def tiered_response_score(rate: float, max_pts: float) -> float:
    """
    Tiered scoring for response rate.

    ≥ 50%  → max_pts       (full marks)
    30–49% → max_pts × 0.67
    10–29% → max_pts × 0.40
    1–9%   → max_pts × 0.13
    0%     → 0.0
    """
    if rate >= 0.50:   return max_pts
    if rate >= 0.30:   return round(max_pts * 0.67, 2)
    if rate >= 0.10:   return round(max_pts * 0.40, 2)
    if rate > 0.00:    return round(max_pts * 0.13, 2)
    return 0.0

def compute_m3(df: pd.DataFrame) -> float:
    """
    M3 = M3a + M3b  (max 15 pts)

    Both M3a and M3b use tiered response scoring:
      ≥ 50%  → 7.5 pts  (full marks)
      30–49% → 5.0 pts
      10–29% → 3.0 pts
      1–9%   → 1.0 pts
      0%     → 0.0 pts

    M3a (7.5 pts) : response rate across all scraped reviews
    M3b (7.5 pts) : response rate for reviews within last 180 days
                    if recent_reviews = 0 → M3b = 0

    Owner responded = Owner Response cell is non-empty and non-NaN.

    Raises ValueError if required columns are missing.
    """
    log.info("[M3] Computing owner engagement score...")

    required_cols = ["Owner Response", "Review Time Line"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[M3] Required column '{col}' not found in excel.")

    total = len(df)
    if total == 0:
        raise ValueError("[M3] No reviews found in excel.")

    # M3a — overall response rate
    responded = df["Owner Response"].apply(
        lambda x: pd.notna(x) and str(x).strip() != ""
    )
    response_count = responded.sum()
    response_rate  = response_count / total
    m3a = tiered_response_score(response_rate, 7.5)
    log.info(f"[M3a] {response_count}/{total} reviews responded ({response_rate:.1%}) → M3a={m3a:.2f}/7.5")

    # M3b — recent response rate (≤ 180 days)
    df = df.copy()
    df["_days"] = df["Review Time Line"].apply(
        lambda x: relative_to_days(str(x))
    )
    recent_mask      = df["_days"] <= 180
    recent_total     = recent_mask.sum()
    recent_responded = (recent_mask & responded).sum()

    if recent_total == 0:
        m3b = 0.0
        log.warning("[M3b] No reviews within last 180 days → M3b = 0")
    else:
        recent_rate = recent_responded / recent_total
        m3b = tiered_response_score(recent_rate, 7.5)
        log.info(f"[M3b] {recent_responded}/{recent_total} recent reviews responded ({recent_rate:.1%}) → M3b={m3b:.2f}/7.5")

    m3     = round(m3a + m3b, 2)
    detail = {
        "total_reviews"    : total,
        "responded"        : int(response_count),
        "response_rate"    : round(float(response_rate), 4),
        "M3a"              : round(m3a, 2),
        "recent_reviews"   : int(recent_total),
        "recent_responded" : int(recent_responded),
        "recent_rate"      : round(float(recent_responded / recent_total) if recent_total > 0 else 0, 4),
        "M3b"              : round(m3b, 2),
    }
    log.info(f"[M3] M3a={m3a:.2f} + M3b={m3b:.2f} → M3={m3:.2f}/15")
    return m3, detail


# ---------------------------------------------------------------------------
# M4 — Digital Footprint Score (0-15 pts)
# ---------------------------------------------------------------------------

def check_website(url: str, timeout: int = 5) -> bool:
    """
    Attempt a GET request to the website.
    Returns True if status code is 200, False otherwise.
    """
    import requests
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        log.debug(f"    [WEBSITE] {url} → status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        log.warning(f"    [WEBSITE] Request failed for '{url}': {e}")
        return False


def is_present(value) -> bool:
    """Check if a value is meaningfully present — handles None, N/A, empty string."""
    if value is None:
        return False
    return str(value).strip().lower() not in ("n/a", "", "none", "null")


def compute_m4(biz: dict) -> float:
    """
    M4 = website score + phone score + address score

    Website  : present + status 200 → +5 | present but unreachable → +2 | missing → 0
    Phone    : present → +3 | missing → 0
    Address  : present → +2 | missing → 0

    Max = 10 pts
    """
    log.info("[M4] Computing digital footprint score...")

    m4 = 0.0

    # Website
    website = biz.get("website")
    if is_present(website):
        log.debug(f"  [M4] Website found: {website} — checking availability...")
        if check_website(website):
            m4 += 5
            log.info(f"  [M4] Website reachable (200) → +5")
        else:
            m4 += 2
            log.info(f"  [M4] Website listed but unreachable → +2")
    else:
        log.info(f"  [M4] No website listed → +0")

    # Phone
    phone = biz.get("phone_number")
    if is_present(phone):
        m4 += 3
        log.info(f"  [M4] Phone present ({phone}) → +3")
    else:
        log.info(f"  [M4] No phone listed → +0")

    # Address
    address = biz.get("address")
    if is_present(address):
        m4 += 2
        log.info(f"  [M4] Address present → +2")
    else:
        log.info(f"  [M4] No address listed → +0")

    m4     = round(m4, 2)
    detail = {
        "website"           : str(website) if is_present(website) else None,
        "website_reachable" : check_website(website) if is_present(website) else False,
        "phone"             : str(phone) if is_present(phone) else None,
        "address"           : str(address) if is_present(address) else None,
    }
    log.info(f"[M4] M4={m4:.2f}/10")
    return m4, detail


# ---------------------------------------------------------------------------
# Review stats — sentiment breakdown + timeline distribution
# ---------------------------------------------------------------------------

def compute_review_stats(df: pd.DataFrame) -> dict:
    """
    Compute sentiment breakdown and timeline distribution for CLI display.
    """
    total = len(df)

    # Sentiment breakdown
    sentiment_counts = {1: 0, 0: 0, -1: 0}
    for s in df["sentiment_score"].fillna(0).astype(int):
        if s in sentiment_counts:
            sentiment_counts[s] += 1

    # Timeline distribution
    timeline_buckets = {
        "< 3 months"  : 0,
        "3–12 months" : 0,
        "1–2 years"   : 0,
        "2–3 years"   : 0,
        "3+ years"    : 0,
    }
    for tl in df["Review Time Line"].fillna(""):
        days = relative_to_days(str(tl))
        if days <= 90:    timeline_buckets["< 3 months"]  += 1
        elif days <= 365: timeline_buckets["3–12 months"] += 1
        elif days <= 730: timeline_buckets["1–2 years"]   += 1
        elif days <= 1095:timeline_buckets["2–3 years"]   += 1
        else:             timeline_buckets["3+ years"]    += 1

    return {
        "total"            : total,
        "sentiment_counts" : sentiment_counts,
        "timeline_buckets" : timeline_buckets,
    }


# ---------------------------------------------------------------------------
# Final aggregation
# ---------------------------------------------------------------------------

def compute_final_score(m1, m2, m3, m4) -> float:
    return round(min(m1 + m2 + m3 + m4, 100), 2)


def save_output(
    file_path: Path,
    biz: dict,
    m1: float, m2: float, m3: float, m4: float,
    final_score: float,
    stats: dict,
    m1_detail: dict,
    m2_detail: dict,
    m3_detail: dict,
    m4_detail: dict,
):
    import json
    business_name, location = file_path.stem.split("_")[:-1], file_path.stem.split("_")[-1]
    business_name = " ".join(business_name).title()

    result = {
        "business"       : biz.get("name", business_name),
        "location"       : biz.get("address", location),
        "category"       : biz.get("category", "Unknown"),
        "final_score"    : final_score,
        "breakdown"      : {
            "M1" : { "score": m1, "max": 55 },
            "M2" : { "score": m2, "max": 20 },
            "M3" : { "score": m3, "max": 15 },
            "M4" : { "score": m4, "max": 10 },
        },
        "M1_detail"      : m1_detail,
        "M2_detail"      : m2_detail,
        "M3_detail"      : m3_detail,
        "M4_detail"      : m4_detail,
        "review_stats"   : stats,
    }

    out_dir  = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{file_path.stem}_score.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info(f"[OUTPUT] Saved to {out_path}")
    return result


def print_summary(result: dict):
    b     = result["breakdown"]
    stats = result["review_stats"]
    total = stats["total"]
    sc    = stats["sentiment_counts"]
    tl    = stats["timeline_buckets"]

    print(f"\n{'='*48}")
    print(f"  {result['business']}")
    print(f"  {result['location']}")
    print(f"{'='*48}")

    # Sentiment breakdown
    print(f"  Reviews analysed : {total}")
    print(f"")
    print(f"  Sentiment breakdown:")
    print(f"    Positive (+1) : {sc[1]:>4}  ({sc[1]/total*100:>5.1f}%)")
    print(f"    Neutral  ( 0) : {sc[0]:>4}  ({sc[0]/total*100:>5.1f}%)")
    print(f"    Negative (-1) : {sc[-1]:>4}  ({sc[-1]/total*100:>5.1f}%)")

    # Timeline distribution
    print(f"")
    print(f"  Review timeline:")
    for bucket, count in tl.items():
        print(f"    {bucket:<14} : {count:>4}  ({count/total*100:>5.1f}%)")

    # Scores
    print(f"")
    print(f"{'─'*48}")
    print(f"  M1  Weighted reviews   : {b['M1']['score']:>6.2f} / {b['M1']['max']}")
    print(f"  M2  Scale & rating     : {b['M2']['score']:>6.2f} / {b['M2']['max']}")
    print(f"  M3  Owner engagement   : {b['M3']['score']:>6.2f} / {b['M3']['max']}")
    print(f"  M4  Digital footprint  : {b['M4']['score']:>6.2f} / {b['M4']['max']}")
    print(f"{'─'*48}")
    print(f"  Final Score            : {result['final_score']:>6.2f} / 100")
    print(f"{'='*48}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute BRS score for a business.",
        epilog="""
Examples:
  python scorer.py excel/pita_shree_handicraft_udaipur.xlsx
        """
    )
    parser.add_argument("filepath", help="Path to the cleaned .xlsx file")
    parser.add_argument("--sheet",  default=0, help="Sheet index or name (default: 0)")
    parser.add_argument("--quiet",  action="store_true", help="Suppress debug logs")

    args      = parser.parse_args()
    file_path = Path(args.filepath)
    sheet     = int(args.sheet) if isinstance(args.sheet, str) and args.sheet.isdigit() else args.sheet

    if args.quiet:
        logging.getLogger().setLevel(logging.INFO)

    if not file_path.exists():
        log.error(f"File '{file_path}' does not exist.")
        sys.exit(1)

    df  = pd.read_excel(file_path, sheet_name=sheet)
    biz = load_places_json(file_path.stem)

    if not biz:
        log.warning(f"places_json/{file_path.stem}.json not found — M2/M4 will be affected.")

    # M1
    try:
        m1, m1_detail = compute_m1(df)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    # M2
    try:
        m2, m2_detail = compute_m2(biz)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    # M3
    try:
        m3, m3_detail = compute_m3(df)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    # M4
    try:
        m4, m4_detail = compute_m4(biz)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    # Final score
    final_score = compute_final_score(m1, m2, m3, m4)
    stats       = compute_review_stats(df)
    result      = save_output(
                    file_path, biz,
                    m1, m2, m3, m4,
                    final_score, stats,
                    m1_detail, m2_detail, m3_detail, m4_detail
                  )
    print_summary(result)


if __name__ == "__main__":
    main()