"""
denominator_sensitivity.py
--------------------------
Visualises how M1 score changes across different denominator values (0.1 to 1.0).
Demonstrates that the denominator is a sensitive hyperparameter.

Usage:
    python denominator_sensitivity.py excel/spice_naturale_bengaluru.xlsx
    python denominator_sensitivity.py excel/spice_naturale_bengaluru.xlsx --step 0.05
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from utils import relative_to_days, recency_multiplier


# ---------------------------------------------------------------------------
# W_credibility (inline — no dependency on scorer.py)
# ---------------------------------------------------------------------------

def compute_w_credibility(is_local_guide, number_of_reviews, num_photos) -> float:
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
# Compute raw scores from excel
# ---------------------------------------------------------------------------

def compute_raw_scores(file_path: Path, sheet=0) -> list[float]:
    df = pd.read_excel(file_path, sheet_name=sheet)

    if "sentiment_score" not in df.columns:
        print("ERROR: 'sentiment_score' column not found. Run sentiment_scorer.py first.")
        sys.exit(1)

    scores = []
    for _, row in df.iterrows():
        wc = compute_w_credibility(
            row["is_local_guide"],
            row["number_of_reviews"],
            row["num_photos"]
        )
        wr = recency_multiplier(relative_to_days(str(row["Review Time Line"])))
        s  = float(row["sentiment_score"])
        scores.append(wc * wr * s)

    return scores


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def run_sensitivity(scores: list[float], step: float = 0.05) -> tuple[list, list]:
    mean_score   = sum(scores) / len(scores)
    denominators = np.arange(0.1, 1.01, step)
    m1_values    = []

    for d in denominators:
        m1 = min((mean_score / d) * 55, 55)
        m1 = max(m1, 0.0)
        m1_values.append(round(m1, 2))

    return list(denominators), m1_values


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sensitivity(
    denominators: list,
    scores: list,
    chosen: float,
    chosen_max: int,
    business_name: str,
    output_path: Path,
    max_range: range = range(40, 65, 5)
):
    mean_score = sum(scores) / len(scores)
    colors     = ["#1D9E75", "#378ADD", "#7F77DD", "#D85A30", "#BA7517"]

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, max_val in enumerate(max_range):
        m1_values = [min(max((mean_score / d) * max_val, 0), max_val) for d in denominators]
        ax.plot(
            denominators, m1_values,
            color=colors[i % len(colors)], linewidth=2,
            label=f"max = {max_val} pts", zorder=3
        )
        chosen_idx = min(range(len(denominators)), key=lambda j: abs(denominators[j] - chosen))
        ax.scatter([chosen], [m1_values[chosen_idx]], color=colors[i % len(colors)], s=50, zorder=5)

    # Annotate chosen config
    chosen_m1 = min(max((mean_score / chosen) * chosen_max, 0), chosen_max)
    ax.axvline(x=chosen, color="#444441", linewidth=1.5, linestyle="--", zorder=4, label=f"Chosen λ = {chosen}")
    ax.annotate(
        f"  λ = {chosen}, max = {chosen_max}\n  M1 = {chosen_m1:.1f}",
        xy=(chosen, chosen_m1),
        xytext=(chosen + 0.06, chosen_m1 - 6),
        fontsize=9, color="#444441",
        arrowprops=dict(arrowstyle="-", color="#444441", lw=0.8)
    )

    ax.set_xlabel("λ (denominator)", fontsize=11)
    ax.set_ylabel("M1 score", fontsize=11)
    ax.set_title(
        f"M1 sensitivity to λ and max value — {business_name}\n"
        f"mean score_i = {mean_score:.4f}  |  dots mark chosen λ = {chosen} per curve",
        fontsize=10
    )
    ax.set_xlim(0.08, 1.05)
    ax.set_ylim(-1, max(max_range) + 10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  [SAVED] {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise M1 denominator sensitivity across 0.1 to 1.0.",
        epilog="""
Examples:
  python denominator_sensitivity.py excel/spice_naturale_bengaluru.xlsx
  python denominator_sensitivity.py excel/spice_naturale_bengaluru.xlsx --step 0.05 --chosen 0.5
        """
    )
    parser.add_argument("filepath",  help="Path to the cleaned .xlsx file with sentiment_score")
    parser.add_argument("--step",    type=float, default=0.05, help="Step size (default: 0.05)")
    parser.add_argument("--chosen",  type=float, default=0.5,  help="Chosen denominator to highlight (default: 0.5)")
    parser.add_argument("--sheet",   default=0, help="Sheet index or name (default: 0)")
    parser.add_argument("--output",  default=None, help="Output image path (default: outputs/<stem>_denominator_sensitivity.png)")

    args      = parser.parse_args()
    file_path = Path(args.filepath)
    sheet     = int(args.sheet) if isinstance(args.sheet, str) and args.sheet.isdigit() else args.sheet

    if not file_path.exists():
        print(f"ERROR: File '{file_path}' does not exist.")
        sys.exit(1)

    output_path = Path(args.output) if args.output else Path("outputs") / f"{file_path.stem}_denominator_sensitivity.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    business_name = file_path.stem.replace("_", " ").title()
    print(f"\n-> {file_path.name}")
    print(f"  [INFO] Business: {business_name}")

    scores = compute_raw_scores(file_path, sheet)
    mean_s = sum(scores) / len(scores)
    print(f"  [INFO] Reviews: {len(scores)} | Mean score_i: {mean_s:.4f}")

    denominators, m1_values = run_sensitivity(scores, step=args.step)

    mean_score = sum(scores) / len(scores)
    print(f"\n  Denominator → M1 (max=55)")
    print(f"  {'─'*26}")
    for d in denominators:
        m = min(max((mean_score / d) * 55, 0), 55)
        marker = " ← chosen" if abs(d - args.chosen) < args.step / 2 else ""
        print(f"  {d:.2f}        → {m:>5.2f}{marker}")

    plot_sensitivity(denominators, scores, args.chosen, 55, business_name, output_path)
    print("\nAll done.")


if __name__ == "__main__":
    main()