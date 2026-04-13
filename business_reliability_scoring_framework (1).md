# Business Reliability Scoring System
## Evaluation Framework — Task 2

---

## Overview

The goal is to assess the creditworthiness of a small business (₹5–10 lakh credit)
without access to financial statements, using only publicly available signals.

The system produces a **Business Reliability Score from 0 to 100**.

### Data Sources

| Source | Signals |
|---|---|
| Google Places API | average_rating, total_rating_count, website, phone_number, address, business_status |
| Scraped Reviews (CSV) | reviewer_name, review_text, review_timeline, owner_response, owner_response_timeline, number_of_reviews, num_photos, is_local_guide |
| Optional | GSTIN, UDYAM registration number |

---

## Scoring Modules

| Module | What it Measures | Max Points |
|---|---|---|
| M1 — Weighted Review Score | Per-review quality: credibility × recency × sentiment | 55 |
| M2 — Scale & Rating | Aggregate star rating + review volume | 20 |
| M3 — Owner Engagement | Response rate + response recency | 15 |
| M4 — Digital Footprint | Website, phone, address presence | 10 |
| M5 — Boost Signals | GSTIN / UDYAM registration | +5 |
| **Total** | | **100** |

---

## M1 — Weighted Review Score (55 pts)

This is the most important module. It evaluates each review individually and
aggregates them into a single score. It carries the highest weight (55 pts)
because review quality — who is saying it, how recent it is, and what they
said — is the strongest available proxy for business reliability.

### Per-Review Formula

```
score_i = W_credibility × W_recency × Sentiment
```

### Why Multiplicative?

W_credibility and W_recency are trust modifiers. Sentiment is the actual signal.
The question being answered is: "What did this reviewer say, and how much should
we trust it?"

If any modifier is 0, the review contributes nothing — only multiplication
guarantees this. Additive or weighted sum approaches allow untrustworthy signals
to bleed into the score regardless.

---

### Step 1 — W_credibility (Reviewer Credibility Weight)

**Hard Rule:**
```
If num_reviews = 0 or unknown → W_credibility = 0 (review is completely ignored)
```

**Otherwise:**
```
Points:
  is_local_guide = TRUE   → +1
  num_reviews ≥ 15        → +4
  num_reviews 6–14        → +3
  num_reviews 1–5         → +2
  num_photos ≥ 1          → +1

Raw range: 0 to 6
W_credibility = raw / 6   → range: 0.00 to 1.00
```

**All Cases:**

| Local Guide | num_reviews | photos | raw | W_credibility |
|---|---|---|---|---|
| Any | 0 or unknown | Any | — | 0.00 |
| No | 1–5 | No | 2 | 0.33 |
| No | 1–5 | Yes | 3 | 0.50 |
| No | 6–14 | No | 3 | 0.50 |
| No | 6–14 | Yes | 4 | 0.67 |
| No | 15+ | No | 4 | 0.67 |
| No | 15+ | Yes | 5 | 0.83 |
| Yes | 1–5 | No | 3 | 0.50 |
| Yes | 1–5 | Yes | 4 | 0.67 |
| Yes | 6–14 | No | 4 | 0.67 |
| Yes | 6–14 | Yes | 5 | 0.83 |
| Yes | 15+ | No | 5 | 0.83 |
| Yes | 15+ | Yes | 6 | 1.00 |

---

### Step 2 — W_recency (Recency Weight)

Google Maps provides relative time strings (e.g. "a month ago", "2 years ago").
These are mapped to approximate days, then assigned a recency multiplier.

```
≤ 90 days   (≤ 3 months)   → 1.00
≤ 365 days  (3–12 months)  → 0.85
≤ 730 days  (1–2 years)    → 0.65
≤ 1095 days (2–3 years)    → 0.45
> 1095 days (3+ years)     → 0.25
```

A 3-month threshold is used — reviews within 90 days receive full weight.
Beyond that, weight decays meaningfully to reflect reduced signal reliability
for credit assessment.

---

### Step 3 — Sentiment Score (LLM Assigned)

The review text is passed to GPT-4o mini which assigns a sentiment score.
A simplified 3-point scale is used — sufficient for credit signal and less
prone to LLM inconsistency than a 5-point scale.

```
+1 → Positive  (satisfied, praises the business)
 0 → Neutral, mixed, unclear, or off-topic
-1 → Negative  (dissatisfied, mentions problems or warns others)
```

Temperature is set to 0 for deterministic outputs across runs.
Business name and address are injected into the system prompt as context
to help the LLM interpret ambiguous reviews accurately.

Reviews are batched in groups of 20 per API call for efficiency.

---

### Step 4 — Aggregate to M1

```
mean_score = mean of all score_i values
M1 = min((mean_score / λ) × 55, 55)
```

**λ (Lambda) — The Normalisation Hyperparameter**

λ is the denominator used to normalise the mean score before scaling to 55 pts.
It represents the expected mean score of a typical legitimate business:

```
Average reviewer  : 1–5 reviews, no photos, not Local Guide
                    → W_credibility = 2/6 = 0.33
Mix of review ages: some recent (1.0), many 3–12 months (0.90)
                    → avg W_recency ≈ 0.75
Mostly positive   : sentiment = +1 for most reviews
                    → S = 1.0

Expected score_i  : 0.33 × 0.75 × 1.0 ≈ 0.25–0.50
Expected mean     : ~0.50 for a legitimately good business
```

λ = 0.5 is the natural baseline — a business with average reviewer
credibility, reasonably fresh reviews, and positive sentiment scores
M1 = 55. Anything below calibrates proportionally.

**Why λ is sensitive:**
λ directly controls the scale of M1. A lower λ inflates all scores; a higher λ
suppresses them. Sensitivity analysis across λ ∈ [0.1, 1.0] shows that:
- λ ≤ 0.40 → M1 caps at 55 for most real businesses (too generous)
- λ = 0.50 → M1 reflects genuine quality differences (chosen)
- λ ≥ 0.80 → M1 is suppressed even for strong businesses (too harsh)

**Chosen value: λ = 0.5**

---

## M2 — Scale & Rating Score (20 pts)

Captures the overall reputation and size of the business using Places API data.

### M2a — Average Rating (13 pts)

```
M2a = ((avg_rating - 1) / 4) × 13
```

| avg_rating | M2a |
|---|---|
| 1.0 | 0.0 |
| 3.0 | 6.5 |
| 4.0 | 9.75 |
| 5.0 | 13.0 |

### M2b — Review Volume (7 pts)

Log scale is used so small businesses are not unfairly penalized.

```
M2b = min((log10(total_rating_count) / log10(3000)) × 7, 7)
```

| total_rating_count | M2b |
|---|---|
| 10 | 2.0 |
| 100 | 4.0 |
| 500 | 5.7 |
| 1000 | 6.4 |
| 3000 | 7.0 |

```
M2 = M2a + M2b
```

---

## M3 — Owner Engagement Score (15 pts)

Measures whether the business owner actively responds to customers — a proxy
for operational aliveness and customer care. Weighted at 15 pts (not higher)
because owner responses are easy to automate or template and do not strongly
differentiate genuine businesses from inactive ones.

Both M3a and M3b use a tiered scoring approach rather than a linear scale.
A linear scale penalizes a 90% response rate almost the same as a 60% rate,
which is not meaningful. Tiered scoring rewards crossing key engagement
thresholds — particularly the 50% mark which indicates active engagement.

### Tiered Response Scoring

Both M3a and M3b use identical tiers:

```
≥ 50%  → 7.5 pts  (full marks — actively engaged)
30–49% → 5.0 pts  (moderate engagement)
10–29% → 3.0 pts  (low engagement)
1–9%   → 1.0 pts  (minimal engagement)
0%     → 0.0 pts  (no engagement)
```

### M3a — Response Rate (7.5 pts)

```
response_rate = reviews_with_owner_reply / total_scraped_reviews
M3a = tiered_response_score(response_rate, max=7.5)
```

A review is considered responded to if the Owner Response cell is non-empty
and non-null.

### M3b — Response Recency (7.5 pts)

Only looks at reviews from the last 6 months to check if the business is
currently active.

```
recent_reviews   = reviews where Review Time Line ≤ 180 days
recent_responded = subset of above with an owner reply
recent_rate      = recent_responded / recent_reviews

M3b = tiered_response_score(recent_rate, max=7.5)

If recent_reviews = 0 → M3b = 0
```

```
M3 = M3a + M3b  (max 15)
```

---

## M4 — Digital Footprint Score (10 pts)

Checks whether the business has a basic verifiable online presence.
Website score requires an actual HTTP GET request — a listed website that
returns status 200 earns full points; a listed but unreachable website
earns partial credit.

```
Website present + status 200  → +5 pts
Website present but unreachable → +2 pts
Website missing                → 0 pts
Phone present                  → +3 pts
Address present                → +2 pts

M4 = sum (max 10)
```

---

## M5 — Boost Signals (optional, +5 pts)

Government registration adds a layer of formal legitimacy. These are not
mandatory as many micro/MSMEs do not publicly list them.

```
GSTIN verified  → +3 pts
UDYAM verified  → +2 pts

M5 = sum (max 5)
```

---

## Final Score

```
Total = M1 + M2 + M3 + M4 + M5
Final Score = min(Total, 100)
```

---
---

## Edge Cases & Known Limitations

### Dynamic Review Count Thresholds (Future Consideration)

**What:**
The current W_credibility thresholds for `number_of_reviews` are static:
```
≥ 15  → +4
6–14  → +3
1–5   → +2
```

**The Problem:**
The Google Maps reviewer population is not uniform across India. In a Tier-1
city like Mumbai or Bangalore, a reviewer with 15 reviews is fairly common —
it does not signal exceptional engagement. In a Tier-2 city like Udaipur, the
same reviewer with 15 reviews is genuinely active and stands out. Static
thresholds treat both identically, which may over-reward low-engagement
reviewers in large markets and under-reward credible reviewers in smaller ones.

**Two Possible Approaches:**

Option A — City Tier Based:
```
Tier-1 : ≥ 20 → +4 | 8–19 → +3 | 1–7 → +2
Tier-2 : ≥ 15 → +4 | 6–14 → +3 | 1–5 → +2  (current)
Tier-3 : ≥ 10 → +4 | 4–9  → +3 | 1–3 → +2
```

Option B — Percentile Based (data-driven):
```
≥ p75 of reviewer population → +4
p25–p75                      → +3
< p25                        → +2
```

**Why Not Implemented Yet:**
- Option A requires maintaining and validating a city tier lookup table per
  pin code — significant engineering overhead for a prototype.
- Option B makes scores non-comparable across businesses. A reviewer with
  8 reviews could earn +4 for one business and +2 for another depending on
  the local reviewer distribution, making the system harder to audit and explain.
- Static thresholds are transparent, consistent, and defensible for a v1 system.
  The same input always produces the same output regardless of which business
  is being evaluated.

**When To Revisit:**
Once enough businesses have been scored across multiple city tiers, a
calibration dataset can be built to empirically determine where the threshold
should sit per market. Dynamic thresholds are a strong v2 feature once that
data exists.
