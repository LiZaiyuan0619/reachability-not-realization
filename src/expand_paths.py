"""W3-f data builder: expand soft-supervision records into (question, path) rows.

Monte-Carlo soft labels without touching the focal loss: each CORRECT path of a
sample becomes one training row (up to --max_paths, shortest + random others);
samples with NO correct path contribute one all-exec row (teaching "hard input
-> execute everything" — the input-dependence Dr.LLM's filtered data never had).
Completions are the path's OWN response (teacher forcing matches the path).

Output: train_router-compatible JSON in data/train_soft/expanded.json
(consume with train_router_multi.py, which allows repeated questions).
"""
import argparse
import glob
import json
import os
import random

L = 28


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glob", default="data/soft/soft_*.json")
    ap.add_argument("--max_paths", type=int, default=3)
    ap.add_argument("--unsolved_frac", type=float, default=1.0,
                    help="subsample failed (all-exec) samples; DR anti-pattern #6: "
                         "equal-weight failures reinforce all-execute collapse")
    ap.add_argument("--out", default="data/train_soft")
    args = ap.parse_args()

    rng = random.Random(0)
    out = {"exp": "soft_expanded", "model_name": None,
           "questions": [], "best_path": [], "best_response": []}
    n_solved = n_unsolved = n_rows = 0
    default = list(range(L))
    for f in sorted(glob.glob(args.in_glob)):
        obj = json.load(open(f))
        out["model_name"] = obj["model_name"]
        for rec in obj["records"]:
            good = [p for p in rec["paths"] if p["r"] > 0.5]
            if good:
                n_solved += 1
                good.sort(key=lambda p: len(p["layers"]))
                pick = [good[0]] + rng.sample(good[1:], min(args.max_paths - 1, len(good) - 1))
            else:
                if rng.random() > args.unsolved_frac:
                    continue
                n_unsolved += 1
                dflt = next((p for p in rec["paths"] if p["layers"] == default), None)
                pick = [dflt or {"layers": default, "resp": ""}]
            for p in pick:
                out["questions"].append(rec["question"])
                out["best_path"].append(p["layers"])
                out["best_response"].append(p["resp"])
                n_rows += 1

    print(f"solved {n_solved} unsolved {n_unsolved} -> rows {n_rows}")
    from collections import Counter
    dist = Counter()
    for bp in out["best_path"]:
        c = Counter(bp)
        for l in range(L):
            dist[min(c.get(l, 0), 2)] += 1
    tot = sum(dist.values())
    print(f"label dist: skip {dist[0]/tot:.3%} exec {dist[1]/tot:.3%} repeat {dist[2]/tot:.3%}")
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, "expanded.json"), "w"))
    print("saved", args.out, "model_name:", out["model_name"])


if __name__ == "__main__":
    main()
