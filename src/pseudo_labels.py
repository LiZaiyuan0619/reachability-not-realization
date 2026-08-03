"""W2-c: intrinsic-signal pseudo labels (sample-level), same schema as MCTS JSONs.

Controlled test of "is supervision the bottleneck?": keep router, training and
eval identical, swap only the label source. Labels: per-layer cos z-scored
within layer across samples; the top-q fraction most-redundant (layer, sample)
cells become SKIP, everything else EXEC (repeat unlabeled - E1 showed repeat
labels carry no signal). Expected under the granularity hypothesis: ~= MCTS
router (~= no-op); its value is completing the comparison matrix.

Reads results/e1_signals/signals.npz (order matches sorted data/train/*.json
iteration, the same order extract_signals.py used). Writes
data/train_pseudo_<tag>/pseudo.json reusing questions/best_response/model_name.
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", default="results/e1_signals/signals.npz")
    ap.add_argument("--skip_frac", type=float, required=True,
                    help="fraction of (sample,layer) cells labeled SKIP, e.g. 0.032 or 0.10")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    z = np.load(args.signals, allow_pickle=True)
    cos = z["cos_mean"]                     # [N, 28]
    N, L = cos.shape
    zs = (cos - cos.mean(0)) / (cos.std(0) + 1e-9)
    thr = np.quantile(zs, 1 - args.skip_frac)
    skip = zs > thr                          # most redundant cells
    print(f"N={N} skip cells: {skip.mean():.3%} (target {args.skip_frac:.1%}); "
          f"samples w/ >=1 skip: {(skip.any(1)).mean():.1%}")

    # rebuild records in the SAME order extract_signals.py consumed them
    rows = []
    for f in sorted(glob.glob("data/train/*.json")):
        obj = json.load(open(f))
        for q, ans in zip(obj["questions"], obj["best_response"]):
            rows.append((q, ans, obj["model_name"]))
    assert len(rows) == N, f"row/signal mismatch: {len(rows)} vs {N}"

    out = {"exp": f"pseudo_{args.tag}", "model_name": rows[0][2],
           "questions": [], "best_path": [], "best_response": []}
    for i, (q, ans, _) in enumerate(rows):
        path = [l for l in range(L) if not skip[i, l]]
        out["questions"].append(q)
        out["best_path"].append(path)
        out["best_response"].append(ans)
    d = f"data/train_pseudo_{args.tag}"
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "pseudo.json"), "w") as f:
        json.dump(out, f)
    print("saved", d, "skip/sample mean:", float(skip.sum(1).mean()))


if __name__ == "__main__":
    main()
