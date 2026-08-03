"""W3-b: multi-path VOTE labels from existing MCTS good_paths.

Hypothesis: single-best-path labels are noisy ("first path that works"); the
per-layer marginal across ALL good paths of a sample is a cleaner supervision
signal. Vote = per-layer majority decision across the sample's good paths
(ties -> exec). Same JSON schema as data/train -> drop-in for train_router.py.

Run with --dry to only print label statistics (no files written).
"""
import argparse
import glob
import json
import os
from collections import Counter

L = 28


def path_to_dec(path):
    c = Counter(path)
    return [min(c.get(l, 0), 2) for l in range(L)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", default="data/train_vote")
    args = ap.parse_args()

    out = {"exp": "vote_labels", "model_name": None,
           "questions": [], "best_path": [], "best_response": []}
    n_paths_hist = Counter()
    changed, dist_best, dist_vote = 0, Counter(), Counter()
    for f in sorted(glob.glob("data/train/*.json")):
        obj = json.load(open(f))
        out["model_name"] = obj["model_name"]
        for q, bp, gps, ans in zip(obj["questions"], obj["best_path"],
                                   obj["good_paths"], obj["best_response"]):
            best_dec = path_to_dec(bp)
            paths = gps if gps else [bp]
            n_paths_hist[min(len(paths), 10)] += 1
            decs = [path_to_dec(p) for p in paths]
            vote = []
            for l in range(L):
                cnt = Counter(d[l] for d in decs)
                top = cnt.most_common()
                # tie -> exec (conservative)
                if len(top) > 1 and top[0][1] == top[1][1]:
                    vote.append(1)
                else:
                    vote.append(top[0][0])
            for v in best_dec:
                dist_best[v] += 1
            for v in vote:
                dist_vote[v] += 1
            if vote != best_dec:
                changed += 1
            # vote decisions -> path (layer executed `dec` times, in order)
            vpath = []
            for l in range(L):
                vpath += [l] * vote[l]
            out["questions"].append(q)
            out["best_path"].append(vpath)
            out["best_response"].append(ans)

    N = len(out["questions"])
    tot = N * L
    print(f"samples: {N}; good_paths per sample (capped at 10): {dict(sorted(n_paths_hist.items()))}")
    print(f"best labels: skip {dist_best[0]/tot:.3%} exec {dist_best[1]/tot:.3%} repeat {dist_best[2]/tot:.3%}")
    print(f"vote labels: skip {dist_vote[0]/tot:.3%} exec {dist_vote[1]/tot:.3%} repeat {dist_vote[2]/tot:.3%}")
    print(f"samples where vote != best: {changed} ({changed/N:.1%})")
    allexec = sum(1 for p in out["best_path"] if p == list(range(L)))
    print(f"vote all-exec samples: {allexec} ({allexec/N:.1%})")
    if not args.dry:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "vote.json"), "w") as f:
            json.dump(out, f)
        print("saved", args.out)


if __name__ == "__main__":
    main()
