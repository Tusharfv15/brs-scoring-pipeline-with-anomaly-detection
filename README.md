# BRS — Business Reliability Score Pipeline

A four-stage pipeline that scores a small business using publicly available Google Maps signals and then runs an LLM agent to detect review manipulation and produce a credit risk narrative.

---

## Table of Contents

- [File Layout](#file-layout)
- [Setup](#setup)
- [Pipeline Overview](#pipeline-overview)
- [Stage 1 — Clean & Fetch](#stage-1--clean--fetch-sort_reviews_by_datepy)
- [Stage 2 — Sentiment Scoring](#stage-2--sentiment-scoring-sentiment_scorerpy)
- [Stage 3 — BRS Scoring](#stage-3--brs-scoring-scorerpy)
- [Stage 4 — Anomaly Detection](#stage-4--anomaly-detection-anomaly_agentpy)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Tech Stack](#tech-stack)

---

## File Layout

```
excel/                          ← raw review sheets (.xlsx), one per business — modified in-place
places_json/                    ← Google Places API data (.json), one per business
outputs/
  <stem>_score.json             ← BRS scoring output (produced by scorer.py)
  <stem>_anomaly.json           ← agent narrative + chart list (produced by anomaly_agent.py)
  <stem>_report.html            ← self-contained HTML report (produced by anomaly_agent.py)
  charts/                       ← chart PNGs embedded in the HTML report
tools/                          ← anomaly detection tool modules (A1–A4 + run_python)
```

---

## Setup

**1. Create and activate a virtual environment**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

| Variable | Required by | Notes |
|---|---|---|
| `OPENAI_API_KEY` | `sentiment_scorer.py`, `anomaly_agent.py` | Always required |
| `GOOGLE_MAPS_API_KEY` | `sort_reviews_by_date.py` | Can be skipped with `--skip-fetch` if `places_json/` files already exist |

---

## Pipeline Overview

```
excel/<stem>.xlsx  (raw)
        │
        ▼
 sort_reviews_by_date.py        ← cleans metadata, sorts rows, fetches Places API
        │                          writes places_json/<stem>.json
        ▼
excel/<stem>.xlsx  (cleaned + sorted)
        │
        ▼
 sentiment_scorer.py            ← adds sentiment_score column via GPT-4o mini
        │                          modifies excel/<stem>.xlsx in-place
        ▼
excel/<stem>.xlsx  (with sentiment_score)
        │
        ▼
 scorer.py                      ← computes M1–M4, writes outputs/<stem>_score.json
        │
        ▼
 anomaly_agent.py               ← LLM agent detects manipulation, writes HTML report
        │
        ▼
outputs/<stem>_report.html
```

---

## Stage 1 — Clean & Fetch (`sort_reviews_by_date.py`)

Processes the raw Excel file in-place and fetches business metadata from the Google Places API.

```bash
python sort_reviews_by_date.py excel/<stem>.xlsx
python sort_reviews_by_date.py excel/<stem>.xlsx --skip-fetch    # skip Places API call
python sort_reviews_by_date.py excel/<stem>.xlsx --no-clean      # skip metadata expansion
python sort_reviews_by_date.py excel/<stem>.xlsx --category "Restaurant"
```

**What it does:**

1. **Expands `Reviewer Metadata`** — parses the raw metadata string into three separate columns:
   - `number_of_reviews` — reviewer's total lifetime review count
   - `num_photos` — reviewer's total photo count
   - `is_local_guide` — True / False

2. **Sorts rows by recency** — auto-detects the date column and sorts reviews from most recent to oldest using the relative-date strings ("a month ago", "2 years ago", etc.)

3. **Fetches Google Places API data** — searches for the business by name + location (parsed from the filename), retrieves rating, review count, phone, website, address, and category. Saves to `places_json/<stem>.json`.

**Output:** `places_json/<stem>.json`

```json
{
  "name": "Spice Naturale",
  "address": "...",
  "category": "Restaurant",
  "phone_number": "+91 ...",
  "website": "https://...",
  "average_rating": 4.3,
  "total_rating_count": 154,
  "business_status": "OPERATIONAL"
}
```

---

## Stage 2 — Sentiment Scoring (`sentiment_scorer.py`)

Adds a `sentiment_score` column to the cleaned Excel file using GPT-4o mini with structured outputs. Modifies the file in-place.

```bash
python sentiment_scorer.py excel/<stem>.xlsx
python sentiment_scorer.py excel/<stem>.xlsx --batch-size 10
python sentiment_scorer.py excel/<stem>.xlsx --force    # re-score even if column exists
```

**What it does:**

- Loads business context from `places_json/<stem>.json` to inform the LLM (name, address, category)
- Batches reviews in groups of 20 (configurable) and sends each batch to GPT-4o mini
- Uses Pydantic structured outputs to guarantee valid scores — only `{-1, 0, 1}` are accepted
- Pre-assigns `0` to empty or very short reviews without an API call
- Skips re-scoring if `sentiment_score` column already exists (use `--force` to override)

**Scoring scale:**

| Score | Meaning |
|---|---|
| `+1` | Positive — reviewer is satisfied, praises the business |
| `0` | Neutral, mixed, unclear, or off-topic |
| `-1` | Negative — reviewer is dissatisfied, mentions problems |
<img width="1398" height="361" alt="image" src="https://github.com/user-attachments/assets/14bab311-26ed-47fd-98bc-490b4a535afa" />


The business category from `places_json/` is passed to the model as context — it affects how ambiguous phrases like "decent prices" are scored relative to category norms.

---

## Stage 3 — BRS Scoring (`scorer.py`)

Reads the cleaned, sentiment-scored Excel and the Places API JSON, computes four scoring modules, and writes a score JSON to `outputs/`.

```bash
python scorer.py excel/<stem>.xlsx
python scorer.py excel/<stem>.xlsx --quiet    # suppress debug logs
```

### Scoring Modules

#### M1 — Weighted Review Score (max 55 pts)

The core signal. Every review gets a `score_i`:

```
score_i = W_credibility × W_recency × sentiment_score
```

**W_credibility** — reviewer profile quality:

| Condition | Points |
|---|---|
| `number_of_reviews == 0` | Hard zero — review ignored entirely |
| Local Guide | +1 |
| Reviews ≥ 15 | +4 · Reviews 6–14 → +3 · Reviews 1–5 → +2 |
| Photos ≥ 1 | +1 |
| **Max raw = 6, W = raw / 6** | Range: 0.0 – 1.0 |

**W_recency** — weight decays with review age:

| Age | Weight |
|---|---|
| ≤ 90 days | 1.00 |
| ≤ 1 year | 0.85 |
| ≤ 2 years | 0.65 |
| ≤ 3 years | 0.45 |
| > 3 years | 0.25 |

**Aggregation:**
```
M1 = min((mean(score_i) / λ) × 55,  55)
λ = 0.5  ← expected mean score_i of a legitimate business
```

#### M2 — Scale & Rating (max 20 pts)

Sourced from `places_json/`:

```
M2a = ((avg_rating - 1) / 4) × 13        ← 13 pts for rating quality
M2b = min((log10(total_ratings) / log10(3000)) × 7, 7)  ← 7 pts for volume
```

#### M3 — Owner Engagement (max 15 pts)

Computed twice — over all reviews (M3a) and over the last 180 days only (M3b):

| Response rate | Score |
|---|---|
| ≥ 50% | 7.5 pts |
| 30–49% | 5.0 pts |
| 10–29% | 3.0 pts |
| 1–9% | 1.0 pts |
| 0% | 0.0 pts |

```
M3 = M3a (all reviews) + M3b (last 180 days)
```

#### M4 — Digital Footprint (max 10 pts)

Website reachability is checked live via HTTP GET:

| Signal | Points |
|---|---|
| Website + HTTP 200 | +5 |
| Website listed but unreachable | +2 |
| Phone number present | +3 |
| Address present | +2 |

<img width="1604" height="603" alt="image" src="https://github.com/user-attachments/assets/fdf6a694-0e41-4651-a2f5-41d6a3cef2a6" />


### Output — `outputs/<stem>_score.json`

Contains the final score, full M1–M4 breakdown, detailed sub-scores, and review stats (sentiment counts + timeline buckets). This file is the direct input to Stage 4.

---

## Stage 4 — Anomaly Detection (`anomaly_agent.py`)

Runs an LLM agent (OpenAI function calling) that reasons over the score JSON and the raw review DataFrame, calls analysis tools in a loop, and produces a structured credit risk narrative with an HTML report.

```bash
python anomaly_agent.py excel/<stem>.xlsx
python anomaly_agent.py excel/<stem>.xlsx --max-iter 15
```

### What the Agent Receives

At the start of each run the agent is given:

- **System prompt** — role definition, full BRS formula explanation, available data columns, tool constraints, and analysis guidelines
- **User prompt** — aggregated stats from the score JSON: final score, M1–M4 breakdown, sentiment counts, timeline distribution

Per-review detail (`M1_detail.per_review`) is stripped from the score JSON before the agent sees it to prevent token overflow. The agent accesses per-review data only by writing code via `run_python`.

### Agent Loop

The agent runs for up to `--max-iter` iterations (default 10). Each iteration:

1. Full message history is sent to the model
2. Model either calls a tool or stops
3. Tool result (JSON + chart images) is appended to message history
4. Next iteration begins

Once the model stops calling tools (or max iterations is reached), a final synthesis call produces the structured four-section report.

### Predefined Tools (A1–A4) — each called at most once

#### A1 — `temporal_analysis`
Groups reviews into monthly buckets and flags any month where volume exceeds `mean + 2σ`. Produces a bar chart at `outputs/charts/<stem>_temporal.png`.

#### A2 — `reviewer_profiling`
Assigns each reviewer a credibility tier (Ghost / Low / Medium / High) using the same W_credibility formula as M1. Detects if low-credibility reviewers cluster in a specific time window. Also reports **one-shot accounts** (`number_of_reviews == 1`) — reviewers whose only ever review is this business.

Flags trigger when:
- Overall: (Ghost + Low) / total > 60%
- Per bucket: (Ghost + Low) / bucket total > 70%
- One-shot accounts > 30% of all reviewers

#### A3 — `content_similarity`
TF-IDF cosine similarity across all review texts (threshold: 0.75). Flags when > 15% of reviewers are involved in similar pairs. Returns the top 25 highest-similarity pairs; total pair count is preserved in `total_similar_pairs`.

#### A4 — `owner_response_analysis`
TF-IDF cosine similarity across owner response texts (threshold: 0.85). Flags when > 50% of responded reviews received a near-identical reply. Also computes descriptive response timing stats for context. Returns the top 25 highest-similarity pairs.

### Open-ended Tool — `run_python`

The agent writes and executes arbitrary Python code against the live data. Available variables:

| Variable | Contents |
|---|---|
| `df` | Full review DataFrame (9 columns) |
| `score` | Complete BRS score dict |
| `biz` | Places API JSON dict |
| `CHARTS_DIR` | `Path("outputs/charts/")` |

Pre-injected libraries: `pd`, `np`, `plt`, `sns`, `sklearn`, `TfidfVectorizer`, `cosine_similarity`, `difflib`, `json`, `re`, `math`, `collections`, `itertools`, `Path`.

`import` and `from X import Y` are both blocked. stdout is captured and returned to the agent, capped at 8,000 characters.
<img width="6789" height="1951" alt="image" src="https://github.com/user-attachments/assets/2f0bc1df-acc2-453c-9e70-eb5c034ac3a8" />

### Output Files

| File | Contents |
|---|---|
| `outputs/<stem>_score.json` | BRS score (from Stage 3) |
| `outputs/<stem>_anomaly.json` | Agent narrative + list of chart paths |
| `outputs/<stem>_report.html` | Self-contained HTML report with embedded charts |

The HTML report is fully self-contained — all charts are base64-encoded inline. Open it directly in any browser.

---

## Running the Full Pipeline

```bash
python sort_reviews_by_date.py excel/<stem>.xlsx --category "Restaurant"
python sentiment_scorer.py     excel/<stem>.xlsx
python scorer.py               excel/<stem>.xlsx
python anomaly_agent.py        excel/<stem>.xlsx
```

---

## Tech Stack

| Layer | Library / Service |
|---|---|
| **Language** | Python 3.11 |
| **LLM — Anomaly Agent** | OpenAI `gpt-5.1-2025-11-13` via function calling |
| **LLM — Sentiment Scoring** | OpenAI `gpt-4o-mini` via structured outputs (Pydantic) |
| **Business Metadata** | Google Places API (New) — text search + place details |
| **Data handling** | pandas, openpyxl |
| **ML / NLP** | scikit-learn (TF-IDF, cosine similarity) |
| **Charting** | matplotlib, seaborn |
| **HTTP** | requests |
| **Env / Config** | python-dotenv |
| **Data validation** | Pydantic v2 |
