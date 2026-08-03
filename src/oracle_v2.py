"""W3-a: FULL-SWEEP oracle (200 sims, no early break) that RECORDS every
evaluated path + reward.

Three uses:
  1) tighter ceiling than the 50-sim early-break oracle;
  2) path-structure analysis: which skips/repeats actually rescue samples?
     how much of the gain is the static {20-23}-ish prune vs input-dependent?
  3) pilot data for SOFT supervision: per-layer skip marginals across all
     correct paths of a sample (is the marginal stable where the single best
     path is noise?).
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np

from data_generation import MCTS, MCTSConfig, MCTSModel
from eval_drllm import load_task

L = 28


class RecordingMCTS(MCTS):
    """Parent's search() with early-breaks removed; returns {path: reward}."""

    def full_search(self, input_text, correct_answer):
        from data_generation import MCTSNode
        root = MCTSNode(self.model.default_path, num_layers=self.model.num_layers)
        evaluated = {}
        for sim in range(self.config.num_simulations):
            node = self.select_node(root, sim + 1)
            if not node.is_fully_expanded() and node.visits > 0:
                node = node.expand()
            if node.path in evaluated:
                reward = evaluated[node.path]
            else:
                reward, _ = self.evaluate_path(input_text, correct_answer, node.path)
                evaluated[node.path] = reward
            node.backpropagate(reward)
        return evaluated


def dec_of(path_layers):
    c = Counter(path_layers)
    return [min(c.get(l, 0), 2) for l in range(L)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", required=True,
                    choices=["arc_easy", "arc_challenge", "dart", "dart_pool"])
    ap.add_argument("--per_level", type=int, default=60)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--num_simulations", type=int, default=200)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if args.task.startswith("dart"):
        samples = load_task(args.task, "short",
                            args.per_level if args.task == "dart" else 0)
        if args.task == "dart_pool":
            by = {}
            for s in samples:
                by.setdefault(s["level"], []).append(s)
            samples = [s for lv in sorted(by) for s in by[lv][: args.per_level]]
    else:
        samples = load_task(args.task)[: args.limit]
    print(f"oracle v2 full-sweep: {args.task} n={len(samples)} sims={args.num_simulations}")

    cfg = MCTSConfig(num_simulations=args.num_simulations,
                     model_name=args.model_path,
                     dataset="dart" if args.task.startswith("dart") else args.task)
    model = MCTSModel(args.model_path, rank=0)
    mcts = RecordingMCTS(model, cfg)

    out_recs = []
    t0 = time.time()
    for i, s in enumerate(samples):
        evaluated = mcts.full_search(s["input"], s["correct"])
        paths = [{"layers": list(p.layers), "r": float(r)} for p, r in evaluated.items()]
        default_r = evaluated.get(model.default_path)
        if default_r is None:
            default_r, _ = mcts.evaluate_path(s["input"], s["correct"], model.default_path)
        out_recs.append({"level": s.get("level"), "orig": float(default_r > 0.5),
                         "oracle": float(any(p["r"] > 0.5 for p in paths)),
                         "paths": paths})
        if (i + 1) % 20 == 0:
            o = sum(r["orig"] for r in out_recs) / len(out_recs)
            c = sum(r["oracle"] for r in out_recs) / len(out_recs)
            print(f"{i+1}/{len(samples)} orig={o:.3f} oracle={c:.3f} "
                  f"({(time.time()-t0)/(i+1):.1f}s/sample)", flush=True)

    n = len(out_recs)
    summary = {"task": args.task, "n": n, "sims": args.num_simulations,
               "orig_acc": sum(r["orig"] for r in out_recs) / n,
               "oracle_acc": sum(r["oracle"] for r in out_recs) / n}
    summary["ceiling_gain"] = summary["oracle_acc"] - summary["orig_acc"]

    # ---- path structure analysis ----
    rescued = [r for r in out_recs if r["orig"] == 0 and r["oracle"] == 1]
    skip_hist = np.zeros(L)
    rep_hist = np.zeros(L)
    static_like = 0
    static_set = {20, 21, 22, 23}
    for r in rescued:
        good = [p for p in r["paths"] if p["r"] > 0.5]
        best = min(good, key=lambda p: len(p["layers"]))
        d = dec_of(best["layers"])
        sk = {l for l in range(L) if d[l] == 0}
        skip_hist += np.array([d[l] == 0 for l in range(L)], float)
        rep_hist += np.array([d[l] == 2 for l in range(L)], float)
        if sk and sk <= static_set:
            static_like += 1
    summary["n_rescued"] = len(rescued)
    summary["rescued_skip_hist"] = [round(float(v), 1) for v in skip_hist]
    summary["rescued_repeat_hist"] = [round(float(v), 1) for v in rep_hist]
    summary["rescued_skipset_subset_of_20-23"] = static_like
    # soft-marginal stability pilot: per-layer skip marginal across correct paths
    margs = []
    for r in out_recs:
        good = [p for p in r["paths"] if p["r"] > 0.5]
        if len(good) >= 3:
            m = np.mean([[dec_of(p["layers"])[l] == 0 for l in range(L)] for p in good], 0)
            margs.append(m)
    if margs:
        M = np.stack(margs)
        summary["soft_marginal_samples"] = len(margs)
        summary["soft_marginal_mean_per_layer"] = [round(float(v), 3) for v in M.mean(0)]

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"oraclev2_{args.task}.json")
    json.dump({"summary": summary, "records": out_recs}, open(out, "w"))
    print("SUMMARY", json.dumps(summary, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
