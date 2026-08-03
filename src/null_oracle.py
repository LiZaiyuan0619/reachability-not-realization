"""Fairness control for the oracle numbers: how much of the MCTS-oracle gain is
just "re-rolling the dice" under an equal evaluation budget?

Three arms on the same samples/budget (arm 1 already measured by oracle_v2):
  arm 2 random-path : 200 evals, each a RANDOM 1-3-step edit (skip/repeat) of
                      the full path, greedy decoding. Same perturbation nature
                      as MCTS paths but no directed search.
  arm 3 best-of-N   : 200 evals, FULL path, temperature sampling (T=0.7).
                      Pure output-randomness re-rolling, no routing at all.
Oracle := any eval correct. If MCTS-oracle >> both nulls, the structured
routing contribution is real; if ~= arm3, the ceiling story collapses to
sampling. ARC (4-way) is where re-roll inflation is most suspected; DART's
free-form answers are the clean check.
"""
import argparse
import json
import os
import random
import time

from data_generation import MCTS, MCTSConfig, MCTSModel, LayerPath
from eval_drllm import load_task

L = 28


def random_edit_path(rng, n_edits, nl=L):
    path = list(range(nl))
    for _ in range(n_edits):
        op = rng.random()
        if op < 0.6 and len(path) > 3:  # skip 1-2 contiguous
            k = rng.choice([1, 2])
            s = rng.randrange(0, len(path) - k + 1)
            path = path[:s] + path[s + k:]
        else:  # repeat one position
            s = rng.randrange(0, len(path))
            if len(path) + 1 <= 2 * nl:
                path = path[: s + 1] + [path[s]] + path[s + 1:]
    return LayerPath(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", required=True,
                    choices=["arc_easy", "arc_challenge", "dart", "dart_pool"])
    ap.add_argument("--per_level", type=int, default=60)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
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
    print(f"null oracle: {args.task} n={len(samples)} budget={args.budget}")

    cfg = MCTSConfig(num_simulations=args.budget, model_name=args.model_path,
                     dataset="dart" if args.task.startswith("dart") else args.task)
    model = MCTSModel(args.model_path, rank=0)
    mcts = MCTS(model, cfg)  # reuse evaluate_path for grading parity
    nl = model.num_layers  # NOT the module-level L=28: 8B=32, qwen3B=36
    full = LayerPath(list(range(nl)))
    print(f"num_layers={nl}")

    rng = random.Random(0)
    recs = []
    t0 = time.time()
    for i, s in enumerate(samples):
        orig_r, _ = mcts.evaluate_path(s["input"], s["correct"], full)
        # arm 2: random path edits, greedy (dedupe like the tree does)
        rp_hit, seen = float(orig_r > 0.5), set()
        evals = 0
        while evals < args.budget and rp_hit == 0.0:
            p = random_edit_path(rng, rng.choice([1, 2, 3]), nl)
            key = tuple(p.layers)
            if key in seen:
                continue
            seen.add(key)
            evals += 1
            r, _ = mcts.evaluate_path(s["input"], s["correct"], p)
            if r > 0.5:
                rp_hit = 1.0
        # arm 3: best-of-N sampling on the full path
        bo_hit = float(orig_r > 0.5)
        n_samp = 0
        prompt = model.prepare_prompt(s["input"], model.tokenizer)
        import torch
        ids = model.tokenizer(prompt, return_tensors="pt").to(model.model.device)
        max_new = 15 if args.task.startswith("dart") else 10
        while n_samp < args.budget and bo_hit == 0.0:
            n_samp += 1
            with torch.no_grad():
                out = model.model.generate(
                    **ids, max_new_tokens=max_new, do_sample=True,
                    temperature=args.temperature, top_p=0.95,
                    pad_token_id=model.tokenizer.eos_token_id)
            resp = model.tokenizer.decode(out[0][ids.input_ids.shape[1]:],
                                          skip_special_tokens=True).strip()
            from eval_drllm import grade
            if grade(args.task, resp, s["correct"]) > 0.5:
                bo_hit = 1.0
        recs.append({"level": s.get("level"), "orig": float(orig_r > 0.5),
                     "randpath": rp_hit, "bestofn": bo_hit})
        if (i + 1) % 20 == 0:
            o = sum(r["orig"] for r in recs) / len(recs)
            a = sum(r["randpath"] for r in recs) / len(recs)
            b = sum(r["bestofn"] for r in recs) / len(recs)
            print(f"{i+1}/{len(samples)} orig={o:.3f} randpath={a:.3f} bestofN={b:.3f} "
                  f"({(time.time()-t0)/(i+1):.1f}s/sample)", flush=True)

    n = len(recs)
    summary = {"task": args.task, "n": n, "budget": args.budget,
               "temperature": args.temperature,
               "orig_acc": sum(r["orig"] for r in recs) / n,
               "randpath_oracle": sum(r["randpath"] for r in recs) / n,
               "bestofn_oracle": sum(r["bestofn"] for r in recs) / n}
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"null_{args.task}.json")
    json.dump({"summary": summary, "records": recs}, open(out, "w"), indent=1)
    print("SUMMARY", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
