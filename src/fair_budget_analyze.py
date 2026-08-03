"""W8 fair-budget reroll audit: OFFLINE analysis + stats (pure-python, CPU-only).

Turns the raw per-sample first-hit ranks produced by fair_budget_oracle.py (and,
for the MCTS arm, the recorded eval-order paths in results/w3_oracle2/oraclev2_*.json)
into a BUDGET-MATCHED pass@k table with paired significance tests.

Fair-budget principle (the whole point of this audit):
  Every "candidate generator" (MCTS-program / random-program / output-sampling) gets
  the SAME number k of distinct verifier calls, the SAME verifier (gold-answer grade),
  the SAME decoding length, on the SAME samples. Candidate #1 is ALWAYS the full/default
  greedy path, so pass@1 == baseline for every generator by construction.
  pass@k(gen) = P[ any of the first k candidates is graded correct ].

Key columns the reviewer cares about:
  - pass@k(MCTS)   : the "structured search ceiling" (== PoLar's offline program source).
  - pass@k(random) : random layer-program + same verifier (THE control).
  - pass@k(sample) : temperature best-of-k on the fixed full path (output reroll only).
  - k* = the MCTS arm's ACTUAL mean #distinct evals: at that budget, does random >= MCTS?

Stats: McNemar exact (paired, per-sample hit@k) between generators; bootstrap 95% CI
on each pass@k and on the random-minus-MCTS paired difference.

Usage (all optional; computes whatever inputs are present):
  python fair_budget_analyze.py \
      --oraclev2 results/w3_oracle2/oraclev2_arc_challenge.json \
      --fairbudget results/w8_fairbudget/fair_arc_challenge_seed0.json \
                   results/w8_fairbudget/fair_arc_challenge_seed1.json \
      --null results/w4_null/null_arc_challenge.json \
      --ks 1,2,4,5,8,16,32,64,200
This file has NO torch/transformers import -> runs on the dev CPU.
"""
import argparse
import json
from math import comb

import numpy as np


# ----------------------------------------------------------------------------- IO
def _first_hit_rank(reward_seq):
    """1-indexed rank of the first reward>0.5 in eval order, else -1."""
    for i, r in enumerate(reward_seq):
        if r > 0.5:
            return i + 1
    return -1


def ranks_from_oraclev2(path):
    """MCTS arm from the recording full-sweep oracle (oracle_v2.py output).
    Returns (ranks, ndistinct, orig) as int arrays; candidate order == MCTS eval order,
    whose element 0 is the full/default path (so rank==1 <=> baseline-correct)."""
    d = json.load(open(path))
    recs = d["records"]
    ranks, ndist, orig = [], [], []
    for r in recs:
        seq = [p["r"] for p in r["paths"]]
        ranks.append(_first_hit_rank(seq))
        ndist.append(len(seq))
        orig.append(int(r["orig"] > 0.5))
    return np.array(ranks), np.array(ndist), np.array(orig)


def ranks_from_fairbudget(paths, gen):
    """Pool one generator's first-hit ranks across >=1 seed files from
    fair_budget_oracle.py. gen in {mcts,rand,samp}. Returns (ranks, ndistinct, orig)."""
    R, N, O = [], [], []
    for p in paths:
        d = json.load(open(p))
        for r in d["records"]:
            R.append(int(r[f"{gen}_rank"]))
            N.append(int(r.get(f"{gen}_ndistinct", r.get("kmax", -1))))
            O.append(int(r["orig"] > 0.5))
    return np.array(R), np.array(N), np.array(O)


# --------------------------------------------------------------------------- stats
def hits_at_k(ranks, k):
    """Per-sample 0/1 hit vector at budget k: correct within first k candidates."""
    return ((ranks >= 1) & (ranks <= k)).astype(int)


def passk(ranks, k):
    return float(hits_at_k(ranks, k).mean())


def bootstrap_ci(vec, B=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for the mean of a 0/1 (or real) per-sample vector."""
    rng = np.random.default_rng(seed)
    n = len(vec)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(B, n))
    means = vec[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def mcnemar_exact(a, b):
    """Exact two-sided McNemar on paired 0/1 hit vectors a,b (same samples).
    Returns (b01, b10, p) where b01 = a-wrong & b-right, b10 = a-right & b-wrong."""
    a = np.asarray(a); b = np.asarray(b)
    b10 = int(((a == 1) & (b == 0)).sum())
    b01 = int(((a == 0) & (b == 1)).sum())
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    m = min(b01, b10)
    # exact binomial two-sided p at q=0.5
    tail = sum(comb(n, i) for i in range(0, m + 1)) / (2 ** n)
    return b01, b10, float(min(1.0, 2 * tail))


# ------------------------------------------------------- SELECTION column (Goal A)
# Pure-python per-sample selectors over a generator's stored candidate list. They turn
# the fair_budget_oracle_v2.py per-candidate (answers, correct, conf) arrays into the
# DEPLOYABLE "selection" column of the capability ledger, alongside the oracle ceiling.
# No numpy / torch / mathruler here -> CPU-testable in isolation (--selftest). Voting is
# on the PRE-EXTRACTED answer strings stored on the GPU side; correctness is looked up
# from stored per-candidate grades (never re-graded on CPU -- mathruler is GPU-only).
#   oracle@k     : any-correct ceiling (== existing rank<=k). Needs a verifier.
#   majority@k   : self-consistency vote -- most-common ANSWER STRING among the first k
#                  candidates (ties -> earliest appearance), graded by that answer's
#                  stored correctness (identical strings share a grade). Verifier-free.
#   confidence@k : BoN-by-confidence -- grade the single highest mean-logprob candidate
#                  among the first k (ties -> earliest). Verifier-free; matches the
#                  path_sc.py / reroll_sc.py convention (mean transition logprob).
#   selection gap = oracle@k - majority@k : correct answers reachable in the first k
#                  candidates that a deployable selector fails to recover.
def oracle_hit(correct, k):
    """1 iff any of the first k candidates is graded correct."""
    return int(any(float(c) > 0.5 for c in correct[:k]))


def majority_hit(answers, correct, k):
    """Self-consistency @k. Most-common NON-EMPTY answer string among the first k
    candidates; ties broken by earliest appearance. Hit = stored correctness of that
    answer (looked up, not re-graded).

    Empty/None answers = the candidate FAILED to produce a parseable answer, so they
    ABSTAIN (cast no vote) rather than voting for a spurious empty bucket. Otherwise
    degenerate search paths (e.g. MCTS probes that skip too many layers and emit nothing)
    would dominate the vote and crush majority to ~0 — an extraction artifact, not a real
    'majority is toxic' finding. If ALL k candidates are empty, the vote fails -> 0."""
    a, c = answers[:k], correct[:k]
    counts, first = {}, {}
    for idx, ans in enumerate(a):
        if ans is None or ans == "":
            continue
        counts[ans] = counts.get(ans, 0) + 1
        first.setdefault(ans, idx)
    if not counts:
        return 0
    winner = min(counts, key=lambda x: (-counts[x], first[x]))
    return int(float(c[first[winner]]) > 0.5)


def confidence_hit(correct, conf, k):
    """BoN-by-confidence @k. Grade the single highest mean-logprob candidate among the
    first k; ties broken by earliest appearance. Verifier-free selector."""
    c, f = correct[:k], conf[:k]
    if not c:
        return 0
    j = max(range(len(f)), key=lambda i: (f[i], -i))
    return int(float(c[j]) > 0.5)


def selection_arms_from_fairbudget(paths, gen):
    """Per-sample {answers,correct,conf} lists for one generator pooled across seed files.
    Returns [] if the files predate the v2 selection fields (v1 rank-only output)."""
    rows, has = [], False
    for p in paths:
        d = json.load(open(p))
        for r in d["records"]:
            if f"{gen}_answers" in r:
                has = True
                rows.append({"answers": r[f"{gen}_answers"],
                             "correct": r[f"{gen}_correct"],
                             "conf": r[f"{gen}_conf"]})
    return rows if has else []


def _selftest():
    """Hand-worked verification of the selector math (CPU-only, no I/O)."""
    # ---- Case A: the HEADLINE selection gap. gold=B, k=3 candidates. ----
    #   answers=[B, A, A]  correct=[1,0,0]  conf=[-0.5,-0.1,-0.2]
    #   oracle finds B (rank 1); majority picks A (2 votes > 1) -> WRONG; confidence
    #   picks the max-logprob candidate idx1 (A, -0.1) -> WRONG. The reachable correct
    #   answer B is NOT recoverable by a deployable selector.
    A_ans, A_cor, A_cnf = ["B", "A", "A"], [1, 0, 0], [-0.5, -0.1, -0.2]
    assert oracle_hit(A_cor, 3) == 1, "oracle should see B"
    assert majority_hit(A_ans, A_cor, 3) == 0, "majority picks A (wrong)"
    assert confidence_hit(A_cor, A_cnf, 3) == 0, "confidence picks A@-0.1 (wrong)"
    # at k=1 every selector == baseline (candidate #1 = full greedy = B, correct)
    assert (oracle_hit(A_cor, 1), majority_hit(A_ans, A_cor, 1),
            confidence_hit(A_cor, A_cnf, 1)) == (1, 1, 1)
    # selection gap oracle-majority @3 == 1 for this sample
    assert oracle_hit(A_cor, 3) - majority_hit(A_ans, A_cor, 3) == 1
    # ---- Case B: majority RESCUES, confidence FAILS. gold=B. ----
    B_ans, B_cor, B_cnf = ["A", "B", "B"], [0, 1, 1], [-0.1, -0.9, -0.8]
    assert oracle_hit(B_cor, 3) == 1
    assert majority_hit(B_ans, B_cor, 3) == 1, "B has 2 votes -> correct"
    assert confidence_hit(B_cor, B_cnf, 3) == 0, "max-logprob idx0 (A,-0.1) wrong"
    # ---- Case C: majority tie -> earliest appearance wins (A over B). ----
    assert majority_hit(["A", "B"], [1, 0], 2) == 1
    # ---- Case D: confidence tie -> earliest wins (idx0). ----
    assert confidence_hit([1, 0], [-0.3, -0.3], 2) == 1
    # ---- Case E: all-unparsed answers -> empty-string bucket, always wrong. ----
    assert majority_hit(["", "", ""], [0, 0, 0], 3) == 0
    # ---- Aggregate over {A,B}: oracle=100%, majority=50%, confidence=0%, gap=50pp. ----
    rows = [{"answers": A_ans, "correct": A_cor, "conf": A_cnf},
            {"answers": B_ans, "correct": B_cor, "conf": B_cnf}]
    orac = sum(oracle_hit(r["correct"], 3) for r in rows) / 2
    majo = sum(majority_hit(r["answers"], r["correct"], 3) for r in rows) / 2
    conf = sum(confidence_hit(r["correct"], r["conf"], 3) for r in rows) / 2
    assert (orac, majo, conf) == (1.0, 0.5, 0.0), (orac, majo, conf)
    assert orac - majo == 0.5, "aggregate selection gap = 50pp"
    print("SELFTEST OK (oracle/majority/confidence/gap; hand-worked A/B/C/D/E + aggregate)")


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="")
    ap.add_argument("--oraclev2", default="", help="MCTS arm: oracle_v2 recording JSON")
    ap.add_argument("--fairbudget", nargs="*", default=[],
                    help="fair_budget_oracle.py JSONs (>=1 seed) -> rand/samp/mcts arms")
    ap.add_argument("--null", default="", help="legacy null_oracle JSON (pass@200 point only)")
    ap.add_argument("--ks", default="1,2,4,5,8,16,32,64,200")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only check of the selector math (majority/confidence/gap)")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    KS = [int(x) for x in args.ks.split(",")]

    arms = {}   # name -> (ranks, ndistinct, orig)
    if args.oraclev2:
        arms["MCTS(oraclev2)"] = ranks_from_oraclev2(args.oraclev2)
    if args.fairbudget:
        for gen, label in [("mcts", "MCTS"), ("rand", "random-path"), ("samp", "sampling")]:
            try:
                arms[label] = ranks_from_fairbudget(args.fairbudget, gen)
            except KeyError:
                pass  # generator not present in this run

    print(f"# Fair-budget reroll audit  task={args.task or '?'}")
    if arms:
        anyname = next(iter(arms))
        print(f"n={len(arms[anyname][0])}  seeds/files pooled where applicable")
        print()
        # ---- pass@k table ----
        hdr = "| generator | mean #distinct | " + " | ".join(f"pass@{k}" for k in KS) + " |"
        print(hdr); print("|" + "---|" * (len(KS) + 2))
        for name, (ranks, ndist, orig) in arms.items():
            nd = np.mean(ndist[ndist > 0]) if (ndist > 0).any() else float("nan")
            row = " | ".join(f"{passk(ranks, k)*100:.1f}" for k in KS)
            print(f"| {name} | {nd:.1f} | {row} |")
        print()

        # ---- MCTS budget callout + paired random-vs-MCTS at matched k ----
        mcts_name = "MCTS" if "MCTS" in arms else ("MCTS(oraclev2)" if "MCTS(oraclev2)" in arms else None)
        rnd_name = "random-path" if "random-path" in arms else None
        if mcts_name:
            mr, mnd, _ = arms[mcts_name]
            kstar = int(round(np.mean(mnd[mnd > 0])))
            ksat = int(np.max(mr[mr >= 1])) if (mr >= 1).any() else 1
            print(f"MCTS actual budget: mean distinct evals = {np.mean(mnd[mnd>0]):.1f}, "
                  f"but MCTS never rescues past rank {ksat} (pass@{ksat} == pass@inf).")
            print(f"=> the honest budget-matched comparison is at k in {{{ksat}, {kstar}}} "
                  f"(MCTS-saturation and MCTS-mean-distinct), NOT k=200.")
            if rnd_name:
                rr = arms[rnd_name][0]
                for k in sorted({ksat, kstar, 5} | {kk for kk in KS if kk <= max(kstar, ksat)}):
                    ha, hb = hits_at_k(mr, k), hits_at_k(rr, k)
                    b01, b10, p = mcnemar_exact(ha, hb)
                    diff = hb.astype(float) - ha.astype(float)
                    lo, hi = bootstrap_ci(diff, B=args.boot)
                    verdict = "random>=MCTS" if hb.mean() >= ha.mean() else "MCTS>random"
                    print(f"  @k={k:>3}: MCTS={ha.mean()*100:5.1f}  random={hb.mean()*100:5.1f}  "
                          f"Δ(rand−mcts)={(hb.mean()-ha.mean())*100:+5.1f} "
                          f"[95%CI {lo*100:+.1f},{hi*100:+.1f}]  McNemar p={p:.3g}  {verdict}")
        print()

    # ---- SELECTION column (Goal A): oracle ceiling vs DEPLOYABLE selectors ----
    # Present only for fair_budget_oracle_v2.py outputs (v1 files lack per-candidate
    # answers/conf and are silently skipped). This is the capability-ledger column the
    # audit was missing: even the answers that ARE reachable (oracle) are not recovered
    # by a deployable selector (majority vote / max-confidence).
    if args.fairbudget:
        sel = {}
        for gen, label in [("mcts", "MCTS"), ("rand", "random-path"), ("samp", "sampling")]:
            rows = selection_arms_from_fairbudget(args.fairbudget, gen)
            if rows:
                sel[label] = rows
        if sel:
            print("## SELECTION column: oracle ceiling vs deployable selectors")
            print("majority@k = self-consistency vote (verifier-free); "
                  "confidence@k = max mean-logprob (BoN-by-conf, verifier-free);")
            print("oracle@k = any-correct (needs verifier); "
                  "gap = oracle − majority = reachable-but-unrecoverable.\n")
            for label, rows in sel.items():
                n = len(rows)
                hdr = f"| {label} (n={n}) | " + " | ".join(f"@{k}" for k in KS) + " |"
                print(hdr); print("|" + "---|" * (len(KS) + 1))
                O = {k: np.array([oracle_hit(r["correct"], k) for r in rows]) for k in KS}
                M = {k: np.array([majority_hit(r["answers"], r["correct"], k) for r in rows]) for k in KS}
                C = {k: np.array([confidence_hit(r["correct"], r["conf"], k) for r in rows]) for k in KS}
                print("| oracle@k     | " + " | ".join(f"{O[k].mean()*100:.1f}" for k in KS) + " |")
                print("| majority@k   | " + " | ".join(f"{M[k].mean()*100:.1f}" for k in KS) + " |")
                print("| confidence@k | " + " | ".join(f"{C[k].mean()*100:.1f}" for k in KS) + " |")
                print("| gap(orac−maj)| " + " | ".join(f"{(O[k].mean()-M[k].mean())*100:+.1f}" for k in KS) + " |")
                # significance of the selection gap at the largest budget present
                kmax = max(KS)
                gapvec = O[kmax].astype(float) - M[kmax].astype(float)
                lo, hi = bootstrap_ci(gapvec, B=args.boot)
                b01, b10, pval = mcnemar_exact(M[kmax], O[kmax])  # oracle-only wins = b01
                print(f"  selection gap @k={kmax}: {gapvec.mean()*100:+.1f}pp "
                      f"[95%CI {lo*100:+.1f},{hi*100:+.1f}]  "
                      f"(oracle-only-correct={b01}, majority-only-correct={b10}, "
                      f"McNemar p={pval:.3g})\n")

    if args.null:
        d = json.load(open(args.null))["summary"]
        print(f"legacy null_oracle (UNMATCHED, budget={d.get('budget')}): "
              f"orig={d['orig_acc']*100:.1f}  randpath@{d.get('budget')}={d['randpath_oracle']*100:.1f}  "
              f"bestofN@{d.get('budget')}={d['bestofn_oracle']*100:.1f}  "
              f"[these are the OLD unmatched numbers the reviewer attacks]")


if __name__ == "__main__":
    main()
