"""
anomaly_agent.py
----------------
Agentic anomaly detection layer for the BRS scoring pipeline.

The agent receives:
  - Scoring results from outputs/<stem>.json
  - Raw review data from excel/<stem>.xlsx
  - Available tools for data analysis

It reasons over the data, calls tools as needed, and produces:
  - Manipulation flags
  - Context adjustment
  - Narrative synthesis

Usage:
    python anomaly_agent.py excel/spice_naturale_bengaluru.xlsx
"""

import sys
import json
import base64
import argparse
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from utils import load_places_json
from tools.temporal_analysis import temporal_analysis
from tools.reviewer_profiling import reviewer_profiling
from tools.content_similarity import content_similarity
from tools.owner_response import owner_response_analysis
from tools.run_python import run_python

load_dotenv(override=True)

client = OpenAI()
MODEL  = "gpt-5.1-2025-11-13"


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function calling schema)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "temporal_analysis",
            "description": (
                "Analyses review volume over time. Groups reviews by month, "
                "generates a bar chart, computes a rolling baseline and detects "
                "any months where review volume spikes significantly above normal. "
                "Use this first to get an overview of review activity over time."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reviewer_profiling",
            "description": (
                "Analyses the credibility distribution of reviewers. "
                "Assigns each reviewer a credibility tier (Ghost/Low/Medium/High) "
                "based on their number of reviews, photos and Local Guide status. "
                "Detects if low-credibility reviewers are concentrated in a specific "
                "time window. Use this after temporal_analysis to check if any spike "
                "was driven by low-credibility accounts."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "content_similarity",
            "description": (
                "Detects suspiciously similar review texts using TF-IDF cosine "
                "similarity. Identifies pairs of reviews that are nearly identical "
                "or templated. Flag triggers if >15% of unique reviewers wrote "
                "similar text."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "owner_response_analysis",
            "description": (
                "Analyses owner responses for copy-paste patterns using TF-IDF "
                "cosine similarity. Also provides descriptive timing statistics "
                "between review date and owner response date for context only. "
                "Flag triggers if >50% of reviews received a near-identical response."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute arbitrary Python code against the review dataframe and "
                "scoring results to explore patterns and cross-signal relationships "
                "possible with the available data columns but not covered by A1-A4. "
                "Available data variables: df (review DataFrame with 9 columns), "
                "score (full BRS scoring dict), biz (places JSON dict). "
                "Available libraries — do NOT import them, they are pre-injected: "
                "pd, np, plt, sns, sklearn, TfidfVectorizer, cosine_similarity, "
                "difflib, json, re, math, collections, itertools, Path, CHARTS_DIR. "
                "Always print() findings. Save charts to CHARTS_DIR."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why you are running this code",
                    },
                },
                "required": ["code", "reasoning"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt(category: str) -> str:
    return f"""You are an expert credit risk analyst specialising in evaluating
small businesses in India for short-term credit (Rs 5-10 lakh).

You have been given the BRS (Business Reliability Score) results for a business
computed from publicly available Google Maps signals. Your job is to:
  1. Analyse the scoring results and underlying review data
  2. Detect anomalies or manipulation patterns
  3. Provide context that helps a human reviewer interpret the score
  4. Decide which analyses to run and in what order
  5. Synthesize all findings into a clear, actionable narrative

---

HOW THE BRS SCORE WAS COMPUTED:

M1 - Weighted Review Score (max 55 pts)
  score_i = W_credibility x W_recency x sentiment
  W_credibility: num_reviews=0 -> 0.0 | local_guide -> +1 |
    reviews 15+ -> +4 | 6-14 -> +3 | 1-5 -> +2 | photos>=1 -> +1
    W = raw / 6
  W_recency: <=90d->1.0 | <=365d->0.85 | <=730d->0.65 | <=1095d->0.45 | >1095d->0.25
  sentiment: +1 positive | 0 neutral | -1 negative
  M1 = min((mean(score_i) / 0.5) x 55, 55)
  lambda=0.5 is the expected mean score_i of a typical legitimate business.

M2 - Scale & Rating (max 20 pts)
  M2a = ((avg_rating - 1) / 4) x 13
  M2b = min((log10(total_ratings) / log10(3000)) x 7, 7)

M3 - Owner Engagement (max 15 pts)
  Tiered: >=50% -> 7.5 | 30-49% -> 5.0 | 10-29% -> 3.0 | 1-9% -> 1.0 | 0% -> 0.0
  M3a: overall response rate | M3b: response rate last 180 days

M4 - Digital Footprint (max 10 pts)
  Website 200 -> +5 | unreachable -> +2 | phone -> +3 | address -> +2

---

AVAILABLE DATA COLUMNS IN df:
  Name                    - reviewer name (string)
  Review Time Line        - relative date e.g. "a month ago", "2 years ago"
  Customer Review         - review text (string)
  Owner Response          - owner reply text, empty if no response
  Owner Response Timeline - relative date of owner reply
  number_of_reviews       - reviewer lifetime review count (int)
  num_photos              - reviewer total photos (int)
  is_local_guide          - True/False
  sentiment_score         - -1, 0 or +1

---

TOOLS:

Predefined tools (A1-A4) — call each at most once:
  temporal_analysis       - review volume over time, spike detection
  reviewer_profiling      - credibility tier distribution, clustering
  content_similarity      - TF-IDF similarity across review texts
  owner_response_analysis - copy-paste detection, timing stats

run_python — for deeper investigation beyond predefined tools:
  - Use it to explore patterns and cross-signal relationships that are
    possible with the 9 data columns above but not covered by A1-A4.
  - Do NOT use it for basic exploration — sentiment counts, response
    rates, column names are already in the scoring summary
  - Do NOT use import statements or 'from X import Y' — both are blocked
    and will raise an ImportError. These are already pre-injected:
      pd, np, plt, sns, sklearn, TfidfVectorizer, cosine_similarity,
      difflib, json, re, math, collections, itertools, Path, CHARTS_DIR
  - df contains exactly the 9 columns listed above — no computed columns
  - Always print() every finding — unprinted values are invisible
  - When printing dataframes, never print more than 5 rows —
    use df.head(5) or df.sample(min(5, len(df))) not df.to_string()
  - Save charts: plt.savefig(CHARTS_DIR / "descriptive_name.png", dpi=130)
  - Charts saved to CHARTS_DIR appear in the final HTML report

---

BUSINESS CATEGORY CONTEXT:
  Category: {category}
  Calibrate expectations accordingly — a restaurant gets more reviews
  than a manufacturer. Anomalies should be judged relative to category norms.

---

ANALYSIS GUIDELINES:
  - The BRS score is deterministic. Do NOT suggest modifying it.
  - Only report what you actually computed — do not fabricate numbers.
  - Flag anomalies clearly with supporting data and credit risk implication.
  - If data looks clean, say so explicitly with evidence.
  - Write as a senior analyst to a credit committee.
  - Do not mention tools, code, or internal steps in your final report.
"""


# ---------------------------------------------------------------------------
# Image encoder
# ---------------------------------------------------------------------------

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Narrative markdown → HTML converter
# ---------------------------------------------------------------------------

def narrative_to_html(narrative: str) -> str:
    """
    Converts LLM markdown narrative to styled HTML.

    Handles:
      - Newlines → <br>
      - **bold** → <strong>bold</strong>
      - Numbered section headings (e.g. "1. SCORE TRUSTWORTHINESS")
        → <span class="narrative-heading">
    """
    import re

    # Newlines first so heading/bold patterns work on clean text
    html = narrative.replace("\n", "<br>")

    # **bold** → <strong>
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

    # Numbered section headings — match "1. ALL CAPS WORDS" pattern
    html = re.sub(
        r'(\d\.\s+[A-Z][A-Z\s&]+?)(\s*<br>)',
        r'<span class="narrative-heading">\1</span>\2',
        html,
    )

    return html


# ---------------------------------------------------------------------------
# Build user prompt
# ---------------------------------------------------------------------------

def build_user_prompt(score: dict, df: pd.DataFrame) -> str:
    """
    Builds the user prompt from scoring results.

    Only aggregated stats are included — per_review detail from M1_detail
    is intentionally excluded to avoid flooding the context window.
    The agent can access per-review data via run_python using df directly.
    """
    b  = score.get("breakdown", {})
    s  = score.get("review_stats", {})
    sc = s.get("sentiment_counts", {})
    tl = s.get("timeline_buckets", {})
    m1 = score.get("M1_detail", {})
    m2 = score.get("M2_detail", {})
    m3 = score.get("M3_detail", {})
    m4 = score.get("M4_detail", {})

    return f"""
Business : {score.get("business", "Unknown")}
Location : {score.get("location", "Unknown")}
Category : {score.get("category", "Unknown")}

--- BRS SCORE ---
Final Score : {score.get("final_score", "N/A")} / 100

Breakdown:
  M1 Weighted reviews   : {b.get("M1", {}).get("score", "N/A")} / {b.get("M1", {}).get("max", 55)}
  M2 Scale & rating     : {b.get("M2", {}).get("score", "N/A")} / {b.get("M2", {}).get("max", 20)}
  M3 Owner engagement   : {b.get("M3", {}).get("score", "N/A")} / {b.get("M3", {}).get("max", 15)}
  M4 Digital footprint  : {b.get("M4", {}).get("score", "N/A")} / {b.get("M4", {}).get("max", 10)}

--- M1 DETAIL ---
  Lambda              : {m1.get("lambda", 0.5)}
  Mean score_i        : {m1.get("mean_score_i", "N/A")}
  Total reviews       : {m1.get("total_reviews", "N/A")}

--- M2 DETAIL ---
  Avg rating          : {m2.get("avg_rating", "N/A")} / 5.0
  Total ratings       : {m2.get("total_rating_count", "N/A")}
  M2a (rating)        : {m2.get("M2a", "N/A")}
  M2b (volume)        : {m2.get("M2b", "N/A")}

--- M3 DETAIL ---
  Total reviews       : {m3.get("total_reviews", "N/A")}
  Responded           : {m3.get("responded", "N/A")}
  Response rate       : {m3.get("response_rate", "N/A")}
  M3a                 : {m3.get("M3a", "N/A")}
  Recent reviews      : {m3.get("recent_reviews", "N/A")}
  Recent responded    : {m3.get("recent_responded", "N/A")}
  Recent rate         : {m3.get("recent_rate", "N/A")}
  M3b                 : {m3.get("M3b", "N/A")}

--- M4 DETAIL ---
  Website             : {m4.get("website", "N/A")}
  Reachable           : {m4.get("website_reachable", "N/A")}
  Phone               : {m4.get("phone", "N/A")}
  Address             : {m4.get("address", "N/A")}

--- REVIEW STATS ---
  Total reviews       : {s.get("total", "N/A")}
  Sentiment:
    Positive (+1)     : {sc.get("1", sc.get(1, "N/A"))}
    Neutral  ( 0)     : {sc.get("0", sc.get(0, "N/A"))}
    Negative (-1)     : {sc.get("-1", sc.get(-1, "N/A"))}
  Timeline:
    < 3 months        : {tl.get("< 3 months", "N/A")}
    3-12 months       : {tl.get("3-12 months", tl.get("3-12 months", "N/A"))}
    1-2 years         : {tl.get("1-2 years", tl.get("1-2 years", "N/A"))}
    2-3 years         : {tl.get("2-3 years", tl.get("2-3 years", "N/A"))}
    3+ years          : {tl.get("3+ years", "N/A")}

Please analyse this business. Use the available tools to investigate
the data. Start with temporal_analysis and reviewer_profiling, then
use your judgment to decide what else to investigate.
"""


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(
    tool_name: str,
    tool_args: dict,
    df: pd.DataFrame,
    biz: dict,
    score: dict,
    stem: str,
) -> tuple:
    if tool_name == "temporal_analysis":
        result = temporal_analysis(df, stem)
        charts = [result["chart"]] if result.get("chart") else []

    elif tool_name == "reviewer_profiling":
        result = reviewer_profiling(df, stem)
        charts = [result["chart"]] if result.get("chart") else []

    elif tool_name == "content_similarity":
        result = content_similarity(df, stem)
        charts = []

    elif tool_name == "owner_response_analysis":
        result = owner_response_analysis(df, stem)
        charts = []

    elif tool_name == "run_python":
        code      = tool_args.get("code", "")
        reasoning = tool_args.get("reasoning", "")
        result    = run_python(code, reasoning, df, biz, score, stem)
        charts    = result.get("charts", [])

    else:
        result = {"error": f"Unknown tool: {tool_name}"}
        charts = []

    return result, charts


# ---------------------------------------------------------------------------
# Build tool result message with optional chart images
# ---------------------------------------------------------------------------

def build_tool_result_message(
    tool_call_id: str,
    result: dict,
    charts: list,
) -> dict:
    content = [
        {
            "type": "text",
            "text": json.dumps(result, indent=2, default=str),
        }
    ]
    for chart_path in charts:
        if chart_path and Path(chart_path).exists():
            b64 = encode_image(chart_path)
            content.append({
                "type"     : "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

    return {
        "role"        : "tool",
        "tool_call_id": tool_call_id,
        "content"     : content,
    }


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(
    df: pd.DataFrame,
    biz: dict,
    score: dict,
    stem: str,
    max_iterations: int = 10,
) -> tuple:
    """
    Runs the anomaly detection agent loop.
    Returns (narrative, all_charts).

    A1-A4 tools are each called at most once.
    run_python can be called multiple times.
    """
    category      = score.get("category", biz.get("category", "Unknown"))
    system_prompt = build_system_prompt(category)
    user_prompt   = build_user_prompt(score, df)

    messages    = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    all_charts  = []
    called_once = set()  # tracks predefined tools already called

    SINGLE_USE_TOOLS = {
        "temporal_analysis",
        "reviewer_profiling",
        "content_similarity",
        "owner_response_analysis",
    }

    print(f"\n[AGENT] Starting analysis: {score.get('business', stem)}")
    print(f"[AGENT] Category: {category}")

    for iteration in range(1, max_iterations + 1):
        print(f"\n[AGENT] Iteration {iteration}/{max_iterations}...")

        # Build available tools — exclude already called single-use tools
        available_tools = [
            t for t in TOOLS
            if t["function"]["name"] not in called_once
        ]

        # If only run_python is left and agent has nothing more to do,
        # it will naturally stop calling tools
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        if not message.tool_calls:
            print(f"[AGENT] Analysis complete after {iteration} iteration(s). Generating structured report...")
            # Force structured synthesis regardless of what agent said
            messages.append(message)
            messages.append({
                "role"   : "user",
                "content": (
                    "You have completed your analysis. "
                    "Produce a final structured report for the human credit reviewer. "
                    "Your report must contain exactly these four sections and nothing else:\n\n"
                    "1. SCORE TRUSTWORTHINESS\n"
                    "State whether the BRS score is trustworthy, inflated, or deflated. "
                    "Support your assessment with specific numbers and patterns you observed — "
                    "review volume trends, reviewer credibility distribution, sentiment breakdown, "
                    "response rates, content uniqueness, and any additional data-driven insights you generated during your analysis. Be thorough and data-driven. Only report what you actually computed and observed — do not infer, assume, or fabricate any numbers or patterns. "
                    "Do not mention tool names or internal processes.\n\n"
                    "2. FLAGS RAISED\n"
                    "List each anomaly or concern you identified. For each one:\n"
                    "  - Describe exactly what was observed (with numbers)\n"
                    "  - Explain why it is suspicious or noteworthy\n"
                    "  - State the credit risk implication\n"
                    "If no flags were raised, explicitly state the data looks clean and why.\n\n"
                    "3. CONTEXT NOTES\n"
                    "Provide rich interpretive context: how does the review volume compare "
                    "to what is normal for this business category and city tier? "
                    "What does the age of reviews tell us about the business trajectory? "
                    "Are there any patterns that are unusual but not necessarily manipulative?\n\n"
                    "4. RECOMMENDATION TO REVIEWER\n"
                    "One clear, specific paragraph telling the human reviewer exactly what "
                    "to verify or focus on before making a credit decision. "
                    "Reference specific observations from your analysis. "
                    "Do not be generic.\n\n"
                    "Write as a senior analyst presenting findings to a credit committee. "
                    "Do not mention tools, code, or internal analysis steps. "
                    "Do not add any other sections, offers, or follow-up questions."
                ),
            })
            final = client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
            return final.choices[0].message.content, all_charts

        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # Guard — skip if single-use tool already called
            if tool_name in SINGLE_USE_TOOLS and tool_name in called_once:
                print(f"[AGENT] -> {tool_name} (skipped — already called)")
                continue

            print(f"[AGENT] -> {tool_name}")
            if tool_name == "run_python":
                print(f"[AGENT]    reason: {tool_args.get('reasoning', '')}")

            result, charts = dispatch_tool(
                tool_name, tool_args, df, biz, score, stem
            )
            all_charts.extend(charts)

            # Log run_python failures explicitly
            if tool_name == "run_python" and not result.get("success", True):
                print(f"[AGENT]    code error: {result.get('error', '')[:120]}")
                print(f"[AGENT]    agent will self-correct...")

            # Truncate run_python output and error before they enter messages.
            # This applies regardless of success so that large tracebacks are
            # also capped.  ~2000 tokens budget = 8000 chars at 4 chars/token.
            MAX_OUTPUT_CHARS = 8000
            if tool_name == "run_python":
                output = result.get("output", "").strip()
                if output:
                    print(f"[AGENT]    output:")
                    for line in output.splitlines():
                        print(f"[AGENT]      {line}")
                    if len(output) > MAX_OUTPUT_CHARS:
                        result["output"] = (
                            output[:MAX_OUTPUT_CHARS]
                            + "\n[output truncated — print less data next time]"
                        )
                error = result.get("error") or ""
                if len(error) > MAX_OUTPUT_CHARS:
                    result["error"] = error[:MAX_OUTPUT_CHARS] + "\n[error truncated]"

            # Mark single-use tool as called
            if tool_name in SINGLE_USE_TOOLS:
                called_once.add(tool_name)
                print(f"[AGENT]    (locked — will not be called again)")

            # Extract flag from result — handles nested structures and False values
            if "flag" in result:
                flag = result["flag"]
            elif "flag_overall" in result:
                flag = result["flag_overall"]
            elif "copy_paste" in result:
                flag = result["copy_paste"].get("flag", "N/A")
            else:
                flag = "N/A"
            print(f"[AGENT]    flag: {flag}")
            if charts:
                print(f"[AGENT]    charts: {charts}")

            messages.append(
                build_tool_result_message(tool_call.id, result, charts)
            )

    print("[AGENT] Max iterations reached. Forcing synthesis.")
    messages.append({
        "role"   : "user",
        "content": (
            "You have completed your analysis. "
            "Produce a final structured report for the human credit reviewer. "
            "Your report must contain exactly these four sections and nothing else:\n\n"
            "1. SCORE TRUSTWORTHINESS\n"
            "State whether the BRS score is trustworthy, inflated, or deflated. "
            "Support your assessment with specific numbers and patterns you observed — "
            "review volume trends, reviewer credibility distribution, sentiment breakdown, "
            "response rates, content uniqueness, and any additional data-driven insights you generated during your analysis. Be thorough and data-driven. Only report what you actually computed and observed — do not infer, assume, or fabricate any numbers or patterns. "
            "Do not mention tool names or internal processes.\n\n"
            "2. FLAGS RAISED\n"
            "List each anomaly or concern you identified. For each one:\n"
            "  - Describe exactly what was observed (with numbers)\n"
            "  - Explain why it is suspicious or noteworthy\n"
            "  - State the credit risk implication\n"
            "If no flags were raised, explicitly state the data looks clean and why.\n\n"
            "3. CONTEXT NOTES\n"
            "Provide rich interpretive context: how does the review volume compare "
            "to what is normal for this business category and city tier? "
            "What does the age of reviews tell us about the business trajectory? "
            "Are there any patterns that are unusual but not necessarily manipulative?\n\n"
            "4. RECOMMENDATION TO REVIEWER\n"
            "One clear, specific paragraph telling the human reviewer exactly what "
            "to verify or focus on before making a credit decision. "
            "Reference specific observations from your analysis. "
            "Do not be generic.\n\n"
            "Write as a senior analyst presenting findings to a credit committee. "
            "Do not mention tools, code, or internal analysis steps. "
            "Do not add any other sections, offers, or follow-up questions."
        ),
    })
    final = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return final.choices[0].message.content, all_charts


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------

def generate_report(
    stem: str,
    score: dict,
    narrative: str,
    charts: list,
) -> Path:
    """
    Generates a self-contained HTML report combining:
      - Business info + BRS score breakdown
      - All charts embedded as base64 images
      - Agent narrative

    No LLM involved — pure Python template substitution.
    Zero extra tokens.
    """
    b  = score.get("breakdown", {})
    s  = score.get("review_stats", {})
    sc = s.get("sentiment_counts", {})
    tl = s.get("timeline_buckets", {})

    # Embed charts as base64
    charts_html = ""
    for chart_path in charts:
        if chart_path and Path(chart_path).exists():
            b64 = encode_image(chart_path)
            label = Path(chart_path).stem.replace(f"{stem}_", "").replace("_", " ").title()
            charts_html += f"""
            <div class="chart">
                <p class="chart-label">{label}</p>
                <img src="data:image/png;base64,{b64}" alt="{label}">
            </div>"""

    # Score breakdown rows
    breakdown_rows = ""
    for module, label in [
        ("M1", "Weighted Reviews"),
        ("M2", "Scale & Rating"),
        ("M3", "Owner Engagement"),
        ("M4", "Digital Footprint"),
    ]:
        score_val = b.get(module, {}).get("score", "N/A")
        max_val   = b.get(module, {}).get("max", "N/A")
        pct       = round(float(score_val) / float(max_val) * 100) if score_val != "N/A" else 0
        breakdown_rows += f"""
            <tr>
                <td>{module} — {label}</td>
                <td>{score_val} / {max_val}</td>
                <td>
                    <div class="bar">
                        <div class="bar-fill" style="width:{pct}%"></div>
                    </div>
                </td>
            </tr>"""

    # Sentiment + timeline
    sentiment_html = f"""
        <div class="stat-row">
            <span class="pos">+1 Positive: {sc.get("1", sc.get(1, "N/A"))}</span>
            <span class="neu"> 0 Neutral: {sc.get("0", sc.get(0, "N/A"))}</span>
            <span class="neg">-1 Negative: {sc.get("-1", sc.get(-1, "N/A"))}</span>
        </div>"""

    timeline_html = ""
    for bucket, count in tl.items():
        timeline_html += f"<span class='bucket'>{bucket}: <b>{count}</b></span>"

    # Narrative — convert markdown to styled HTML
    narrative_html = narrative_to_html(narrative)

    final_score = score.get("final_score", "N/A")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BRS Report — {score.get("business", stem)}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto;
          padding: 0 20px; color: #2c2c2a; background: #fff; }}
  h1   {{ font-size: 24px; margin-bottom: 4px; }}
  h2   {{ font-size: 18px; color: #444; margin-top: 32px; border-bottom: 1px solid #eee;
          padding-bottom: 6px; }}
  .meta {{ color: #666; font-size: 14px; margin-bottom: 24px; }}
  .score-box {{ background: #f5f5f0; border-radius: 10px; padding: 20px 28px;
                display: inline-block; margin: 16px 0; }}
  .score-num {{ font-size: 48px; font-weight: bold; color: #1D9E75; }}
  .score-max {{ font-size: 20px; color: #888; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
  td    {{ padding: 8px 10px; font-size: 14px; border-bottom: 1px solid #f0f0e8; }}
  td:first-child {{ width: 220px; }}
  td:nth-child(2) {{ width: 80px; text-align: right; font-weight: bold; }}
  .bar  {{ background: #eee; border-radius: 4px; height: 10px; width: 200px; }}
  .bar-fill {{ background: #1D9E75; height: 10px; border-radius: 4px; }}
  .stat-row {{ display: flex; gap: 24px; font-size: 14px; margin: 10px 0; }}
  .pos  {{ color: #1D9E75; font-weight: bold; }}
  .neu  {{ color: #888780; font-weight: bold; }}
  .neg  {{ color: #E24B4A; font-weight: bold; }}
  .bucket {{ background: #f5f5f0; border-radius: 4px; padding: 3px 8px;
             font-size: 13px; margin: 2px; display: inline-block; }}
  .charts {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 16px; }}
  .chart  {{ flex: 1 1 400px; }}
  .chart-label {{ font-size: 13px; color: #666; margin-bottom: 4px; }}
  .chart img   {{ width: 100%; border-radius: 8px; border: 1px solid #eee; }}
  .narrative {{ background: #fafaf8; border-left: 4px solid #1D9E75;
                padding: 16px 20px; border-radius: 0 8px 8px 0;
                font-size: 14px; line-height: 1.7; margin-top: 12px; }}
  .narrative-heading {{ display: block; font-size: 16px; font-weight: bold;
                        color: #1D9E75; margin-top: 24px; margin-bottom: 6px;
                        letter-spacing: 0.4px; border-bottom: 1px solid #d4ede6;
                        padding-bottom: 4px; }}
  .narrative-heading:first-child {{ margin-top: 4px; }}
  .footer {{ font-size: 12px; color: #aaa; margin-top: 40px; text-align: center; }}
</style>
</head>
<body>

<h1>{score.get("business", stem)}</h1>
<div class="meta">
  {score.get("location", "")} &nbsp;|&nbsp;
  {score.get("category", "Unknown category")} &nbsp;|&nbsp;
  {s.get("total", "N/A")} reviews analysed
</div>

<div class="score-box">
  <span class="score-num">{final_score}</span>
  <span class="score-max"> / 100</span>
  <div style="font-size:13px;color:#888;margin-top:4px;">Business Reliability Score</div>
</div>

<h2>Score Breakdown</h2>
<table>{breakdown_rows}</table>

<h2>Review Stats</h2>
{sentiment_html}
<div style="margin-top:8px">{timeline_html}</div>

<h2>Charts</h2>
<div class="charts">{charts_html}</div>

<h2>Agent Analysis</h2>
<div class="narrative">{narrative_html}</div>

<div class="footer">
  Generated by BRS Anomaly Detection Agent &nbsp;|&nbsp;
  For human review only — not a credit decision
</div>

</body>
</html>"""

    out_dir  = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{stem}_report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[AGENT] Report saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

def save_agent_output(stem: str, score: dict, narrative: str, charts: list) -> Path:
    out_dir  = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{stem}_anomaly.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "narrative": narrative,
            "charts"   : charts,
        }, f, indent=2, ensure_ascii=False)
    print(f"[AGENT] Saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run anomaly detection agent on a scored business.",
        epilog="""
Examples:
  python anomaly_agent.py excel/spice_naturale_bengaluru.xlsx
  python anomaly_agent.py excel/spice_naturale_bengaluru.xlsx --max-iter 15
        """
    )
    parser.add_argument("filepath",   help="Path to the cleaned .xlsx file")
    parser.add_argument("--sheet",    default=0, help="Sheet index (default: 0)")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="Max tool iterations (default: 10)")

    args      = parser.parse_args()
    file_path = Path(args.filepath)
    sheet     = int(args.sheet) if isinstance(args.sheet, str) and args.sheet.isdigit() else args.sheet

    if not file_path.exists():
        print(f"ERROR: File '{file_path}' does not exist.")
        sys.exit(1)

    stem       = file_path.stem
    df         = pd.read_excel(file_path, sheet_name=sheet)
    df.columns = df.columns.str.strip()  # normalise column names — Excel often adds leading/trailing spaces
    biz        = load_places_json(stem)
    score_path = Path("outputs") / f"{stem}_score.json"

    if not score_path.exists():
        print(f"ERROR: '{score_path}' not found. Run scorer.py first.")
        sys.exit(1)

    with open(score_path, "r", encoding="utf-8") as f:
        score = json.load(f)

    # Strip per_review array — not needed by agent, causes token explosion
    if "M1_detail" in score and "per_review" in score["M1_detail"]:
        score["M1_detail"] = {
            k: v for k, v in score["M1_detail"].items()
            if k != "per_review"
        }

    narrative, charts = run_agent(df, biz, score, stem, max_iterations=args.max_iter)

    print(f"\n{'='*60}")
    print("AGENT NARRATIVE")
    print(f"{'='*60}")
    print(narrative)
    print(f"{'='*60}\n")

    save_agent_output(stem, score, narrative, charts)
    generate_report(stem, score, narrative, charts)


if __name__ == "__main__":
    main()