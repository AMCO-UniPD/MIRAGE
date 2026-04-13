#!/usr/bin/env python3
"""
Compute TrueSkill ratings from pairwise survey results.

Usage:  uv run src/survey_html/evaluate_survey.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import trueskill

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SURVEY_DIR = Path(__file__).parent
CSV_PATH   = SURVEY_DIR / "outputs" / "survey_results.csv"
OUT_PATH   = SURVEY_DIR / "outputs" / "ratings.txt"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
rows = []
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

print(f"Loaded {len(rows)} votes from {CSV_PATH}")

# ---------------------------------------------------------------------------
# TrueSkill
# ---------------------------------------------------------------------------
env = trueskill.TrueSkill(draw_probability=0.15)   # ~15% tie rate typical

ts_ratings: dict[str, trueskill.Rating] = defaultdict(env.create_rating)

for r in rows:
    ma, mb, vote = r["method_a"], r["method_b"], r["vote"]
    ra, rb = ts_ratings[ma], ts_ratings[mb]
    if vote == "A":
        new_ra, new_rb = env.rate_1vs1(ra, rb)
    elif vote == "B":
        new_rb, new_ra = env.rate_1vs1(rb, ra)
    else:  # TIE
        new_ra, new_rb = env.rate_1vs1(ra, rb, drawn=True)
    ts_ratings[ma], ts_ratings[mb] = new_ra, new_rb

# Conservative skill estimate: mu - 3*sigma  (trueskill.expose)
ts_ranked = sorted(ts_ratings.items(), key=lambda x: env.expose(x[1]), reverse=True)

# ---------------------------------------------------------------------------
# Win / tie / loss counts and pairwise table
# ---------------------------------------------------------------------------
wins    : dict[str, int] = defaultdict(int)
losses  : dict[str, int] = defaultdict(int)
ties    : dict[str, int] = defaultdict(int)
appeared: dict[str, int] = defaultdict(int)

pair_stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "wins": defaultdict(int), "ties": 0})

for r in rows:
    ma, mb, vote = r["method_a"], r["method_b"], r["vote"]
    appeared[ma] += 1
    appeared[mb] += 1
    key = tuple(sorted([ma, mb]))
    pair_stats[key]["n"] += 1
    if vote == "A":
        wins[ma]   += 1
        losses[mb] += 1
        pair_stats[key]["wins"][ma] += 1
    elif vote == "B":
        wins[mb]   += 1
        losses[ma] += 1
        pair_stats[key]["wins"][mb] += 1
    else:
        ties[ma] += 1
        ties[mb] += 1
        pair_stats[key]["ties"] += 1

def win_rate(m: str) -> float:
    n = appeared[m]
    return wins[m] / n * 100 if n else 0.0

# ---------------------------------------------------------------------------
# Per-dataset/category breakdown
# ---------------------------------------------------------------------------
cat_wins : dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
cat_total: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))

for r in rows:
    ds, cat = r["dataset"], r["category"]
    ma, mb, vote = r["method_a"], r["method_b"], r["vote"]
    cat_total[(ds, cat)][ma] += 1
    cat_total[(ds, cat)][mb] += 1
    if vote == "A":
        cat_wins[(ds, cat)][ma] += 1
    elif vote == "B":
        cat_wins[(ds, cat)][mb] += 1

# ---------------------------------------------------------------------------
# Build output text
# ---------------------------------------------------------------------------
lines: list[str] = []
W = 72

def sep(char=""): return char * W
def header(title): return f"\n{'━'*W}\n  {title}\n{'━'*W}"

lines.append("=" * W)
lines.append("  SURVEY EVALUATION RESULTS")
lines.append(f"  Source: {CSV_PATH}")
lines.append(f"  Total votes: {len(rows)}  |  Unique users: {len(set(r['email'] for r in rows))}")
lines.append("=" * W)

# --- TrueSkill ranking ---
lines.append(header("1. TRUESKILL RANKING  (mu − 3σ conservative estimate)"))
lines.append(f"  {'Rank':<6} {'Method':<20} {'mu':>7} {'sigma':>7} {'exposed':>9} {'appearances':>13}")
lines.append("  " + sep())
for rank, (m, rating) in enumerate(ts_ranked, 1):
    exposed = env.expose(rating)
    lines.append(f"  {rank:<6} {m:<20} {rating.mu:>7.2f} {rating.sigma:>7.2f} {exposed:>9.2f} {appeared[m]:>13}")

# --- Win rates ---
lines.append(header("2. WIN / TIE / LOSS (excluding ties from denominator)"))
lines.append(f"  {'Method':<20} {'Win%':>7} {'W':>6} {'T':>6} {'L':>6} {'Total':>7}")
lines.append("  " + sep())
methods_by_winrate = sorted(appeared.keys(), key=win_rate, reverse=True)
for m in methods_by_winrate:
    n_wl = wins[m] + losses[m]
    wr = wins[m] / n_wl * 100 if n_wl else 0.0
    lines.append(f"  {m:<20} {wr:>7.1f}% {wins[m]:>6} {ties[m]:>6} {losses[m]:>6} {appeared[m]:>7}")

# --- Pairwise table ---
lines.append(header("3. PAIRWISE BREAKDOWN"))
lines.append(f"  {'Pair':<42} {'N':>5} {'W(left)':>9} {'W(right)':>9} {'Ties':>6}")
lines.append("  " + sep())
for (m1, m2), d in sorted(pair_stats.items(), key=lambda x: -x[1]["n"]):
    w1 = d["wins"].get(m1, 0)
    w2 = d["wins"].get(m2, 0)
    pair_label = f"{m1}  vs  {m2}"
    lines.append(f"  {pair_label:<42} {d['n']:>5} {w1:>9} {w2:>9} {d['ties']:>6}")

# --- Per-category win rates ---
lines.append(header("4. PER DATASET/CATEGORY WIN RATE"))
for (ds, cat) in sorted(cat_total.keys()):
    totals = cat_total[(ds, cat)]
    winsmap = cat_wins[(ds, cat)]
    cat_header = f"  [{ds}/{cat}]"
    lines.append(cat_header)
    sorted_methods = sorted(totals.keys(), key=lambda m: winsmap.get(m, 0) / totals[m], reverse=True)
    for m in sorted_methods:
        n  = totals[m]
        w  = winsmap.get(m, 0)
        wr = w / n * 100 if n else 0.0
        lines.append(f"    {m:<20}  {wr:5.1f}%  ({w}/{n})")

lines.append("\n" + "=" * W)

output = "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# Write & print
# ---------------------------------------------------------------------------
OUT_PATH.write_text(output)
print(output)
print(f"\nResults written to: {OUT_PATH}")
