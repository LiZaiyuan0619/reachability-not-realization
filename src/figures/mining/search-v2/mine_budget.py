#!/usr/bin/env python3
"""Mine real per-K budget curves from w8_fairbudget_v2 snapshot JSON.
All values are REAL (mcts_passk / rand_passk / samp_passk / rand_majority),
zero interpolation. Output: data/budget_real.csv (long form)."""
import json, os, csv

SNAP = "workspace/myPaper/drllm/docs/data/snapshot/w8_fairbudget_v2"
OUT  = os.path.join(os.path.dirname(__file__), "data", "budget_real.csv")

# (dir, task, family, display label) — cross-family, cross-difficulty, all have K=200
CELLS = [
    ("l3b",        "arc_challenge", "Llama",   "Llama-3.2-3B / ARC-c"),
    ("l3b",        "dart",          "Llama",   "Llama-3.2-3B / DART"),
    ("l318bk200",  "dart_pool",     "Llama",   "Llama-3.1-8B / DART-pool"),
    ("q253bk200",  "arc_challenge", "Qwen",    "Qwen2.5-3B / ARC-c"),
    ("q34bk200",   "dart",          "Qwen",    "Qwen3-4B / DART"),
    ("q2514bk200", "arc_challenge", "Qwen",    "Qwen2.5-14B / ARC-c"),
    ("mis7bk200",  "dart",          "Mistral", "Mistral-7B / DART"),
]

rows = []
for d, task, fam, label in CELLS:
    f = os.path.join(SNAP, d, f"fairv2_{task}_seed0.json")
    s = json.load(open(f))["summary"]
    ks = sorted(int(k) for k in s["rand_passk"].keys())
    for k in ks:
        sk = str(k)
        rows.append({
            "cell": label, "family": fam, "task": task, "K": k,
            "rand_oracle":  round(s["rand_passk"][sk]   * 100, 2),
            "mcts_oracle":  round(s["mcts_passk"][sk]    * 100, 2),
            "samp_oracle":  round(s["samp_passk"][sk]    * 100, 2),
            "rand_majority":round(s["rand_majority"][sk] * 100, 2),
            "orig": round(s["orig_acc"] * 100, 2),
        })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows across {len(CELLS)} cells -> {OUT}")
for d, task, fam, label in CELLS:
    sub = [r for r in rows if r["cell"] == label]
    kmax = max(r["K"] for r in sub)
    last = [r for r in sub if r["K"] == kmax][0]
    print(f"  {label:<28} K={kmax:<4} rand={last['rand_oracle']:<6} mcts={last['mcts_oracle']:<6} majority={last['rand_majority']}")
