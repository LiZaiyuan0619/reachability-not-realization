"""W8 fair-budget reroll audit -- GPU runner (produces the MISSING budget-matched
pass@k curves for random-layer-program and output-sampling, alongside MCTS).

Why this exists
---------------
The old audit (results/w4_null) reported random-path oracle @ budget=200 (99.7% on ARC)
vs MCTS-oracle @ 200 *simulations* (~80%). But MCTS only makes ~9-44 DISTINCT verifier
calls (the rest of its 200 sims re-visit cached paths), so the budgets were NOT matched
-- the exact hole a reviewer attacks. This script fixes it: every generator gets the
SAME candidate budget k, the SAME gold-answer verifier, the SAME decoding length, on the
SAME samples, and candidate #1 is ALWAYS the full/default greedy path so pass@1==baseline
for all arms. We record the 1-indexed rank of the first correct candidate per generator;
fair_budget_analyze.py turns those ranks into pass@k(k) curves + paired stats.

Generators (all graded by eval_drllm.grade -> ONE verifier):
  mcts : ordered DISTINCT paths from a 200-sim recording MCTS full-sweep (== oracle_v2 /
         == PoLar's offline program source), eval order, greedy. rank/ndistinct recorded.
  rand : full path, then distinct random 1-3 skip/repeat edits (null_oracle.random_edit_path),
         greedy, dedup, until first correct or k=kmax. THE control.
  samp : full path, then temperature samples on the fixed full path (output reroll only),
         until first correct or k=kmax. Isolates "more samples" from "different path".

Budget/fairness knobs: --kmax (candidate budget for rand/samp), --num_simulations (MCTS),
  --temperature, --seed (>=2 seeds for the table). Greedy for path arms, temperature for
  the sampling arm (that difference is inherent to the arm's meaning and is disclosed).

Needs the data_models drop-in (modeling_llama_FULL.py -> model.model.layer_indices),
exactly like oracle_v2.py / null_oracle.py. Run via rjob/w8_fairbudget.sh.

Self-test (no GPU): python fair_budget_oracle.py --selftest
"""
import argparse
import json
import os
import random
import time


# ----------------------------------------------------------------- pure-python core
def first_hit_rank(reward_seq):
    """1-indexed rank of first reward>0.5, else -1. Pure python (CPU-testable)."""
    for i, r in enumerate(reward_seq):
        if r > 0.5:
            return i + 1
    return -1


def reroll_until_hit(candidate_iter, grade_fn, kmax, first_reward=None):
    """Consume candidate texts (candidate #1 supplied separately as first_reward if the
    full-path greedy was already evaluated) until a correct one or kmax candidates.
    Returns (rank, n_evals, rewards) where rewards is the reward prefix actually seen.
    candidate_iter yields decoded strings for candidates 2..kmax (lazily, so we can stop)."""
    rewards = []
    if first_reward is not None:
        rewards.append(1.0 if first_reward > 0.5 else 0.0)
        if first_reward > 0.5:
            return 1, 1, rewards
    for text in candidate_iter:
        r = 1.0 if grade_fn(text) > 0.5 else 0.0
        rewards.append(r)
        if r > 0.5:
            return len(rewards), len(rewards), rewards
        if len(rewards) >= kmax:
            break
    return -1, len(rewards), rewards


def _selftest():
    """Validate rank / early-stop / candidate-1 bookkeeping with stub 'models'."""
    g = lambda t: 1.0 if t == "OK" else 0.0
    # full path wrong, 3rd random candidate correct, kmax=8
    rank, ne, rw = reroll_until_hit(iter(["x", "OK", "y"]), g, 8, first_reward=0.0)
    assert (rank, ne) == (3, 3), (rank, ne, rw)
    # full path already correct -> rank 1, no reroll consumed
    rank, ne, rw = reroll_until_hit(iter(["OK"]), g, 8, first_reward=1.0)
    assert (rank, ne) == (1, 1), (rank, ne)
    # never hits within kmax -> -1, stops exactly at kmax evals
    rank, ne, rw = reroll_until_hit(iter(["x"] * 100), g, 5, first_reward=0.0)
    assert (rank, ne) == (-1, 5), (rank, ne)
    # first_hit_rank basic
    assert first_hit_rank([0, 0, 1, 0]) == 3 and first_hit_rank([0, 0]) == -1
    # pass@k monotonic from ranks
    ranks = [1, 3, -1, 2, 6]
    pk = lambda k: sum(1 for r in ranks if 1 <= r <= k) / len(ranks)
    assert pk(1) <= pk(3) <= pk(8) and pk(1) == 0.2 and pk(3) == 0.6
    print("SELFTEST OK (rank / early-stop / candidate-1 / pass@k bookkeeping)")


# ----------------------------------------------------------------------- GPU driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="CPU-only logic check, no model")
    ap.add_argument("--model_path")
    ap.add_argument("--task", choices=["arc_easy", "arc_challenge", "dart", "dart_pool"])
    ap.add_argument("--per_level", type=int, default=60)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--kmax", type=int, default=64, help="candidate budget for rand/samp arms")
    ap.add_argument("--num_simulations", type=int, default=200, help="MCTS sim budget")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gens", default="mcts,rand,samp",
                    help="comma list subset of {mcts,rand,samp} to run")
    ap.add_argument("--out_dir")
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return
    assert args.model_path and args.task and args.out_dir, "need --model_path --task --out_dir"

    import torch
    from data_generation import MCTS, MCTSConfig, MCTSModel, MCTSNode, LayerPath
    from eval_drllm import load_task, grade
    from null_oracle import random_edit_path

    gens = args.gens.split(",")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.task.startswith("dart"):
        samples = load_task(args.task, "short", args.per_level if args.task == "dart" else 0)
        if args.task == "dart_pool":
            by = {}
            for s in samples:
                by.setdefault(s["level"], []).append(s)
            samples = [s for lv in sorted(by) for s in by[lv][: args.per_level]]
    else:
        samples = load_task(args.task)[: args.limit]
    print(f"fair-budget: {args.task} n={len(samples)} kmax={args.kmax} "
          f"sims={args.num_simulations} seed={args.seed} gens={gens}")

    cfg = MCTSConfig(num_simulations=args.num_simulations, model_name=args.model_path,
                     dataset="dart" if args.task.startswith("dart") else args.task)
    model = MCTSModel(args.model_path, rank=0)
    mcts = MCTS(model, cfg)
    nl = model.num_layers
    full = LayerPath(list(range(nl)))
    max_new = 15 if args.task.startswith("dart") else 10
    grade_fn_of = lambda correct: (lambda text: grade(args.task, text, correct))

    def mcts_ordered_rewards(inp, correct):
        """Recording full-sweep (oracle_v2 semantics) but RE-GRADED with the unified
        verifier so every arm shares grade(). Returns rewards in eval order (elem0=full)."""
        gf = grade_fn_of(correct)
        root = MCTSNode(model.default_path, num_layers=nl)
        evaluated, order = {}, []
        for sim in range(args.num_simulations):
            node = mcts.select_node(root, sim + 1)
            if not node.is_fully_expanded() and node.visits > 0:
                node = node.expand()
            if node.path in evaluated:
                reward = evaluated[node.path]
            else:
                _, raw = mcts.evaluate_path(inp, correct, node.path)
                reward = 1.0 if gf(raw) > 0.5 else 0.0
                evaluated[node.path] = reward
                order.append(reward)
            node.backpropagate(reward)
        return order

    def rand_candidates(inp):
        """Lazy generator of decoded texts for distinct random-edit paths (candidates 2..)."""
        seen = {tuple(full.layers)}
        while True:
            p = random_edit_path(random, random.choice([1, 2, 3]), nl)
            key = tuple(p.layers)
            if key in seen:
                continue
            seen.add(key)
            yield model.generate_with_path(inp, p, max_new_tokens=max_new)

    def samp_candidates(inp):
        """Lazy generator of temperature samples on the FIXED full path (candidates 2..)."""
        model.model.layer_indices = list(full.layers)
        prompt = model.prepare_prompt(inp, model.tokenizer)
        ids = model.tokenizer(prompt, return_tensors="pt").to(model.model.device)
        while True:
            with torch.no_grad():
                out = model.model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                                           temperature=args.temperature, top_p=0.95,
                                           pad_token_id=model.tokenizer.eos_token_id)
            yield model.tokenizer.decode(out[0][ids.input_ids.shape[1]:],
                                         skip_special_tokens=True).strip()

    recs = []
    t0 = time.time()
    for i, s in enumerate(samples):
        inp, correct = s["input"], s["correct"]
        gf = grade_fn_of(correct)
        # candidate #1 shared by ALL arms: full/default greedy
        full_text = model.generate_with_path(inp, full, max_new_tokens=max_new)
        r0 = 1.0 if gf(full_text) > 0.5 else 0.0
        rec = {"level": s.get("level"), "orig": r0, "kmax": args.kmax}
        if "mcts" in gens:
            seq = mcts_ordered_rewards(inp, correct)
            rec["mcts_rank"] = first_hit_rank(seq)
            rec["mcts_ndistinct"] = len(seq)
        if "rand" in gens:
            rank, ne, _ = reroll_until_hit(rand_candidates(inp), gf, args.kmax, first_reward=r0)
            rec["rand_rank"], rec["rand_nevals"] = rank, ne
        if "samp" in gens:
            rank, ne, _ = reroll_until_hit(samp_candidates(inp), gf, args.kmax, first_reward=r0)
            rec["samp_rank"], rec["samp_nevals"] = rank, ne
        recs.append(rec)
        if (i + 1) % 20 == 0:
            def pk(key, k):
                v = [r for r in recs if key in r]
                return sum(1 for r in v if 1 <= r[key] <= k) / max(len(v), 1)
            msg = []
            for g, key in [("mcts", "mcts_rank"), ("rand", "rand_rank"), ("samp", "samp_rank")]:
                if g in gens:
                    msg.append(f"{g}@{args.kmax}={pk(key, args.kmax):.3f}")
            print(f"{i+1}/{len(samples)} orig={sum(r['orig'] for r in recs)/len(recs):.3f} "
                  f"{' '.join(msg)} ({(time.time()-t0)/(i+1):.1f}s/sample)", flush=True)

    n = len(recs)
    KS = [1, 2, 4, 5, 8, 16, 32, args.kmax]
    summary = {"task": args.task, "n": n, "kmax": args.kmax, "seed": args.seed,
               "num_simulations": args.num_simulations, "temperature": args.temperature,
               "orig_acc": sum(r["orig"] for r in recs) / n}
    for g, key in [("mcts", "mcts_rank"), ("rand", "rand_rank"), ("samp", "samp_rank")]:
        if g in gens:
            summary[f"{g}_passk"] = {
                k: sum(1 for r in recs if 1 <= r[key] <= k) / n for k in KS}
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"fair_{args.task}_seed{args.seed}.json")
    json.dump({"summary": summary, "records": recs}, open(out, "w"), indent=1)
    print("SUMMARY", json.dumps(summary, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
