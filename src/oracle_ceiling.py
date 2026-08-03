"""W2-a: test-time MCTS oracle ceiling on EVAL sets.

Before investing in any routing improvement, measure the ceiling: if an oracle
that searches layer paths per-sample (same MCTS as the supervision generator,
same 50-sim budget as our data-gen) cannot beat the default path on the eval
distribution, then sequence-level layer routing has no headroom in this setup
and the token-level direction is the only game in town.

Reports original acc vs oracle acc (any searched path correct) per level.
Requires the data_models drop-in (layer_indices mechanism) shadowed in.
"""
import argparse
import json
import os
import time

from data_generation import MCTS, MCTSConfig, MCTSModel
from eval_drllm import load_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", required=True,
                    choices=["arc_easy", "arc_challenge", "dart", "dart_pool"])
    ap.add_argument("--per_level", type=int, default=100, help="dart tasks: samples per level")
    ap.add_argument("--limit", type=int, default=300, help="arc tasks: total samples")
    ap.add_argument("--num_simulations", type=int, default=50, help="match data-gen budget")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if args.task.startswith("dart"):
        samples = load_task(args.task, "short",
                            args.per_level if args.task == "dart" else 0)
        if args.task == "dart_pool":  # load_task gives 200/level from the tail
            by = {}
            for s in samples:
                by.setdefault(s["level"], []).append(s)
            samples = [s for lv in sorted(by) for s in by[lv][: args.per_level]]
    else:
        samples = load_task(args.task)[: args.limit]
    print(f"oracle ceiling: {args.task} n={len(samples)} sims={args.num_simulations}")

    cfg = MCTSConfig(num_simulations=args.num_simulations,
                     model_name=args.model_path,
                     dataset="dart" if args.task.startswith("dart") else args.task)
    model = MCTSModel(args.model_path, rank=0)
    mcts = MCTS(model, cfg)

    records = []
    t0 = time.time()
    for i, s in enumerate(samples):
        orig_r, _ = mcts.evaluate_path(s["input"], s["correct"], model.default_path)
        good = mcts.search(s["input"], s["correct"], orig_r)
        oracle = 1.0 if (orig_r > 0.5 or good) else 0.0
        shortest = min((len(p.layers) for p in good), default=model.num_layers)
        records.append({"orig": float(orig_r > 0.5), "oracle": oracle,
                        "level": s["level"], "n_good": len(good), "shortest": shortest})
        if (i + 1) % 25 == 0:
            o = sum(r["orig"] for r in records) / len(records)
            c = sum(r["oracle"] for r in records) / len(records)
            print(f"{i+1}/{len(samples)} orig={o:.3f} oracle={c:.3f} "
                  f"({(time.time()-t0)/(i+1):.1f}s/sample)", flush=True)

    n = len(records)
    summary = {"task": args.task, "n": n, "sims": args.num_simulations,
               "orig_acc": sum(r["orig"] for r in records) / n,
               "oracle_acc": sum(r["oracle"] for r in records) / n}
    summary["ceiling_gain"] = summary["oracle_acc"] - summary["orig_acc"]
    if args.task.startswith("dart"):
        per = {}
        for lv in range(1, 6):
            rs = [r for r in records if r["level"] == lv]
            if rs:
                per[lv] = {"n": len(rs),
                           "orig": sum(r["orig"] for r in rs) / len(rs),
                           "oracle": sum(r["oracle"] for r in rs) / len(rs)}
        summary["per_level"] = per
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"oracle_{args.task}.json")
    json.dump({"summary": summary, "records": records}, open(out, "w"), indent=1)
    print("SUMMARY", json.dumps(summary, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
