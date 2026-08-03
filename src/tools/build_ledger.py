"""Build the capability ledger (per-sample fate matrix) -- Figure-1 of the
capability-accounting paper.

Decompose reported capability into per-sample columns on the SAME samples:
  G    direct-generation correct           (greedy gen)                 [production]
  R    recognition correct                 (MC / letter-loglik)         [recognition]
  S_K  >=1 of K temperature samples right   (output reroll reach)       [sampling reach]
  P_K  >=1 of K layer-programs/paths right  (MCTS distinct programs)     [path reach]
  Z    zero-MLP / scar edit repairs it      (mechanistic repair)        [repair]
  ROUT deployed layer-router path is right  (arm12 routed)              [deployable path]
  M_K  majority-vote / self-consistency     -- PER-SAMPLE NOT STORED     [selection]  (see note)

then classify every sample into ONE disjoint "fate" bucket (waterfall), so the
per-(model,task) fractions form a stacked bar.

This is pure data-reorganization from existing result JSONs -- NO GPU, read-only
except the ledger outputs. All joins are index joins on identical, verified sample
orders (see the alignment asserts in build()). Run:

    python tools/build_ledger.py            # from dr-llm/
    python tools/build_ledger.py --selftest # logic check, no files

Sources (all Llama-3.2-3B-Instruct 392a143b unless noted; gemma-4-12B-it for logiqa):
  w15_bench_l3b/arc_challenge_baseline{,_mcll,_zmlp20}.json  -> G,R,Z  (1172, one run)
  w8_fairbudget/fair_arc_challenge_seed0.json                -> orig(G),S_K,P_K (300)
  w8_arm12/arc_challenge_baseline.json + drllm-router-seed1  -> G(==orig), ROUT (300)
  w8_fairbudget/fair_dart_seed0.json                         -> orig(G),S_K,P_K (300)
  w8_arm12/dart_baseline.json + drllm-router-seed1           -> G, ROUT (300)
  w12_zmlp_g412_logiqa/agieval_logiqa_baseline{,_zmlp29}.json-> G,Z  (300)
  w11_gap_g412hard/agieval_logiqa_baseline_mcll.json         -> R    (300)
  lens/reroll_reroll_l3b_dart.json (+gsm8k)                  -> majority@K,oracle@K AGGREGATE ONLY
"""
import argparse
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RES, "ledger")

# reachability budgets at which we materialise the S_K / P_K binary flags
KS = [8, 16, 200]
KMAIN = 200  # budget used for the headline waterfall (full audit budget = ceiling)


def load(path):
    with open(os.path.join(RES, path)) as f:
        return json.load(f)


def accs(recs, field="acc"):
    return [1 if r[field] > 0.5 else 0 for r in recs]


def reach_at(rank, k):
    """rank is 1-indexed rank of first correct candidate, -1 if none within budget."""
    return 1 if (rank is not None and 1 <= rank <= k) else 0


# --------------------------------------------------------------------------- classify
def fate(row, avail):
    """Disjoint waterfall bucket for one sample given which columns are available.
    Priority: deployed > scar-repairable > recognition-gap > sampling-reachable >
    path-reachable-only > unreached. Only uses columns present in `avail`."""
    g = row["G"]
    if g == 1:
        return "deployed_G1"
    # G == 0 below
    if "Z" in avail and row.get("Z_repair") == 1:
        return "scar_repairable"
    if "R" in avail and row.get("R") == 1:
        return "recognition_gap"
    if "S_K" in avail and row.get(f"S@{KMAIN}") == 1:
        return "sampling_reachable"
    if "P_K" in avail and row.get(f"P@{KMAIN}") == 1:
        return "path_reachable_only"
    return "unreached"


BUCKETS = ["deployed_G1", "scar_repairable", "recognition_gap",
           "sampling_reachable", "path_reachable_only", "unreached"]

# --------------------------------------------------------------------------- cases


def build_arc_l3b():
    """Llama-3.2-3B / arc_challenge -- flagship. The first-300 AUDIT INTERSECTION where
    ALL columns coexist (G,R,Z,S_K,P_K,ROUT) on verified-identical samples. This is the
    coherent all-column Figure-1 bar. (build_arc_l3b_full gives G/R/Z on all 1172.)"""
    g15 = load("w15_bench_l3b/arc_challenge_baseline.json")["records"]
    r15 = load("w15_bench_l3b/arc_challenge_baseline_mcll.json")["records"]
    z15 = load("w15_bench_l3b/arc_challenge_baseline_zmlp20.json")["records"]
    fb = load("w8_fairbudget/fair_arc_challenge_seed0.json")["records"]
    a12 = load("w8_arm12/arc_challenge_baseline.json")["records"]
    rt = load("w8_arm12/drllm-router-seed1/arc_challenge_routed_p-0.5.json")["records"]
    assert len(g15) == len(r15) == len(z15) == 1172
    assert len(fb) == len(a12) == len(rt) == 300
    assert [x["correct"] for x in g15] == [x["correct"] for x in r15] == [x["correct"] for x in z15]
    assert accs(a12) == [1 if r["orig"] > 0.5 else 0 for r in fb]  # orig==arm12 baseline 300/300
    R, Z = accs(r15), accs(z15)
    rows = []
    for i in range(300):
        f = fb[i]
        g = 1 if f["orig"] > 0.5 else 0  # audit greedy 'orig' so S_K/P_K/ROUT nest over G
        row = {"model": "Llama-3.2-3B", "task": "arc_challenge", "idx": i,
               "gold": g15[i]["correct"], "G": g, "G_gen_w15": accs(g15)[i],
               "R": R[i], "Z_acc": Z[i], "Z_repair": 1 if (g == 0 and Z[i] == 1) else 0}
        for arm, key in [("samp", "S"), ("mcts", "P"), ("rand", "Prand")]:
            rank = f[f"{arm}_rank"]
            row[f"{key}_rank"] = rank
            for k in KS:
                row[f"{key}@{k}"] = reach_at(rank, k)
        row["ROUT"] = 1 if rt[i]["acc"] > 0.5 else 0
        row["M_K"] = None  # PER-SAMPLE MISSING (no reroll for l3b/arc, no per-cand answers)
        row["fate"] = fate(row, {"G", "R", "Z", "S_K", "P_K", "ROUT"})
        rows.append(row)
    return rows, [{"G", "R", "Z", "S_K", "P_K", "ROUT"}] * 300


def build_arc_l3b_full():
    """Llama-3.2-3B / arc_challenge, ALL 1172 samples, G/R/Z only (one w15 run) --
    robustness view of the production/recognition/repair separation at full n."""
    g15 = load("w15_bench_l3b/arc_challenge_baseline.json")["records"]
    r15 = load("w15_bench_l3b/arc_challenge_baseline_mcll.json")["records"]
    z15 = load("w15_bench_l3b/arc_challenge_baseline_zmlp20.json")["records"]
    G, R, Z = accs(g15), accs(r15), accs(z15)
    rows = []
    for i in range(len(g15)):
        row = {"model": "Llama-3.2-3B", "task": "arc_challenge_full1172", "idx": i,
               "gold": g15[i]["correct"], "G": G[i], "R": R[i], "Z_acc": Z[i],
               "Z_repair": 1 if (G[i] == 0 and Z[i] == 1) else 0, "M_K": None}
        row["fate"] = fate(row, {"G", "R", "Z"})
        rows.append(row)
    return rows, [{"G", "R", "Z"}] * len(g15)


def build_dart_l3b():
    """Llama-3.2-3B / dart -- path-reachability audit. Generative task: no R, no Z."""
    fb = load("w8_fairbudget/fair_dart_seed0.json")["records"]
    a12 = load("w8_arm12/dart_baseline.json")["records"]
    rt = load("w8_arm12/drllm-router-seed1/dart_routed_p-0.5.json")["records"]
    assert len(fb) == len(a12) == len(rt) == 300
    assert accs(a12) == [1 if r["orig"] > 0.5 else 0 for r in fb]  # same 300 samples
    rows = []
    for i in range(300):
        f = fb[i]
        g = 1 if f["orig"] > 0.5 else 0
        row = {"model": "Llama-3.2-3B", "task": "dart", "idx": i,
               "level": f.get("level"), "G": g, "R": None, "Z_acc": None,
               "Z_repair": None, "M_K": None}
        for arm, key in [("samp", "S"), ("mcts", "P"), ("rand", "Prand")]:
            rank = f[f"{arm}_rank"]
            row[f"{key}_rank"] = rank
            for k in KS:
                row[f"{key}@{k}"] = reach_at(rank, k)
        row["ROUT"] = 1 if rt[i]["acc"] > 0.5 else 0
        row["fate"] = fate(row, {"G", "S_K", "P_K", "ROUT"})
        rows.append(row)
    avail = {"G", "S_K", "P_K", "ROUT"}
    return rows, [avail] * 300


def build_logiqa_g412():
    """gemma-4-12B / agieval_logiqa -- scar case: G,R,Z on 300 (verified same samples)."""
    gb = load("w12_zmlp_g412_logiqa/agieval_logiqa_baseline.json")["records"]
    zb = load("w12_zmlp_g412_logiqa/agieval_logiqa_baseline_zmlp29.json")["records"]
    rb = load("w11_gap_g412hard/agieval_logiqa_baseline_mcll.json")["records"]
    gb2 = load("w11_gap_g412hard/agieval_logiqa_baseline.json")["records"]
    assert len(gb) == len(zb) == len(rb) == 300
    # verified: w12 baseline == w11 baseline (identical gold + per-sample acc)
    assert [x["correct"] for x in gb] == [x["correct"] for x in gb2] == [x["correct"] for x in rb]
    assert accs(gb) == accs(gb2)
    G, R, Z = accs(gb), accs(rb), accs(zb)
    rows = []
    for i in range(300):
        row = {"model": "gemma-4-12B", "task": "agieval_logiqa", "idx": i,
               "gold": gb[i]["correct"], "G": G[i], "R": R[i], "Z_acc": Z[i],
               "Z_repair": 1 if (G[i] == 0 and Z[i] == 1) else 0, "M_K": None}
        row["fate"] = fate(row, {"G", "R", "Z"})
        rows.append(row)
    return rows, [{"G", "R", "Z"}] * 300


# --------------------------------------------------------------------------- aggregate
def frac_table(rows):
    """Per (model,task): bucket fractions + key column rates + cross-tab gaps."""
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["task"])].append(r)
    out = []
    for (m, t), rs in by.items():
        n = len(rs)
        rec = {"model": m, "task": t, "n": n}
        for b in BUCKETS:
            rec[b] = round(sum(1 for r in rs if r["fate"] == b) / n, 4)
        # column marginals (only over samples where the column exists)
        def rate(pred, cond=lambda r: True):
            sub = [r for r in rs if cond(r)]
            return round(sum(1 for r in sub if pred(r)) / len(sub), 4) if sub else None
        rec["G_rate"] = rate(lambda r: r["G"] == 1)
        rec["R_rate"] = rate(lambda r: r.get("R") == 1, lambda r: r.get("R") is not None)
        rec["Z_repair_rate"] = rate(lambda r: r.get("Z_repair") == 1,
                                     lambda r: r.get("Z_repair") is not None)
        has_s = any(f"S@{KMAIN}" in r for r in rs)
        has_p = any(f"P@{KMAIN}" in r for r in rs)
        for k in KS:
            if has_s:
                rec[f"S@{k}_rate"] = rate(lambda r, k=k: r.get(f"S@{k}") == 1,
                                          lambda r, k=k: f"S@{k}" in r)
            if has_p:
                rec[f"Pmcts@{k}_rate"] = rate(lambda r, k=k: r.get(f"P@{k}") == 1,
                                              lambda r, k=k: f"P@{k}" in r)
                rec[f"Prand@{k}_rate"] = rate(lambda r, k=k: r.get(f"Prand@{k}") == 1,
                                              lambda r, k=k: f"Prand@{k}" in r)
        # ---- EXACT task categories (cross-tabs; may overlap, NOT the disjoint bar) ----
        # cat1 G=1 == G_rate (deployed). cat2 G0R1, cat3 G0R1Z1, cat4 G0R0S1,
        # cat5 S1&M0 (M_K missing), cat6 P1&router0.
        rec["cat_G0_R1"] = rate(lambda r: r["G"] == 0 and r.get("R") == 1,
                                lambda r: r.get("R") is not None)             # recognition-realiz
        rec["cat_G0_R1_Z1"] = rate(
            lambda r: r["G"] == 0 and r.get("R") == 1 and r.get("Z_repair") == 1,
            lambda r: r.get("R") is not None and r.get("Z_repair") is not None)  # scar within recog
        if has_s:
            rec["cat_G0_R0_S1"] = rate(
                lambda r: r["G"] == 0 and (r.get("R") in (0, None)) and r.get(f"S@{KMAIN}") == 1,
                lambda r: f"S@{KMAIN}" in r)                                  # reachable-not-recog
        rec["cat_S1_M0"] = "MISSING (M_K per-sample not stored)"
        rec["recog_realiz_gap"] = rec["cat_G0_R1"]  # alias
        if any("ROUT" in r for r in rs):
            rec["ROUT_rate"] = rate(lambda r: r.get("ROUT") == 1, lambda r: "ROUT" in r)
            # path reachable (mcts) but deployed router wrong
            rec[f"path_reach_router_fail@{KMAIN}"] = rate(
                lambda r: r.get(f"P@{KMAIN}") == 1 and r.get("ROUT") == 0,
                lambda r: f"P@{KMAIN}" in r and "ROUT" in r)
            # sampling reachable but router (single deploy) wrong
            rec[f"samp_reach_deploy_fail@{KMAIN}"] = rate(
                lambda r: r.get(f"S@{KMAIN}") == 1 and r["G"] == 0,
                lambda r: f"S@{KMAIN}" in r)
        rec["M_K_per_sample"] = "MISSING (needs rerun)"
        out.append(rec)
    return out


def availability_matrix():
    cols = ["G", "R", "S_K", "P_K", "Z", "ROUT", "M_K"]
    cases = [
        ("Llama-3.2-3B", "arc_challenge",
         {"G": "y", "R": "y", "S_K": "y(n=300)", "P_K": "y(n=300)", "Z": "y",
          "ROUT": "y(n=300)", "M_K": "MISSING"}),
        ("Llama-3.2-3B", "dart",
         {"G": "y", "R": "n/a(gen)", "S_K": "y", "P_K": "y", "Z": "MISSING",
          "ROUT": "y", "M_K": "MISSING(agg-only)"}),
        ("gemma-4-12B", "agieval_logiqa",
         {"G": "y", "R": "y", "S_K": "MISSING", "P_K": "MISSING", "Z": "y",
          "ROUT": "MISSING", "M_K": "MISSING"}),
    ]
    return cols, cases


# --------------------------------------------------------------------------- reroll ctx
def reroll_context():
    """Aggregate majority/oracle selection gaps (NOT per-sample joinable to ledger:
    different n, budget K, temperature, and decoding than the fairbudget audit)."""
    import glob
    ctx = []
    for p in sorted(glob.glob(os.path.join(RES, "lens", "reroll_reroll_l3b_*.json"))):
        d = json.load(open(p))
        c, a = d["config"], d["acc"]
        ctx.append({"file": os.path.basename(p), "model": "Llama-3.2-3B",
                    "task": c["task"], "K": c["K"], "temp": c["temp"], "n": c["n"],
                    "single": a["single"], "random@K": a["random@K"],
                    "majority@K": a["majority@K"], "oracle@K": a["oracle@K"],
                    "oracle_minus_majority": round(a["oracle@K"] - a["majority@K"], 3),
                    "majority_minus_single": round(a["majority@K"] - a["single"], 3)})
    return ctx


# --------------------------------------------------------------------------- selftest
def _selftest():
    assert reach_at(1, 8) == 1 and reach_at(8, 8) == 1 and reach_at(9, 8) == 0
    assert reach_at(-1, 200) == 0 and reach_at(None, 8) == 0
    r = {"G": 1}
    assert fate(r, {"G"}) == "deployed_G1"
    r = {"G": 0, "Z_repair": 1}
    assert fate(r, {"G", "Z"}) == "scar_repairable"
    r = {"G": 0, "Z_repair": 0, "R": 1}
    assert fate(r, {"G", "Z", "R"}) == "recognition_gap"
    r = {"G": 0, "Z_repair": 0, "R": 0, f"S@{KMAIN}": 1}
    assert fate(r, {"G", "Z", "R", "S_K"}) == "sampling_reachable"
    r = {"G": 0, "Z_repair": 0, "R": 0, f"S@{KMAIN}": 0, f"P@{KMAIN}": 1}
    assert fate(r, {"G", "Z", "R", "S_K", "P_K"}) == "path_reachable_only"
    r = {"G": 0, "Z_repair": 0, "R": 0, f"S@{KMAIN}": 0, f"P@{KMAIN}": 0}
    assert fate(r, {"G", "Z", "R", "S_K", "P_K"}) == "unreached"
    # a recognition_gap sample missing S_K col stays recognition_gap
    r = {"G": 0, "R": 1}
    assert fate(r, {"G", "R"}) == "recognition_gap"
    print("SELFTEST OK (reach_at / fate waterfall / missing-column handling)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    for builder in (build_arc_l3b, build_arc_l3b_full, build_dart_l3b, build_logiqa_g412):
        rows, _ = builder()
        all_rows.extend(rows)

    # per-sample jsonl
    with open(os.path.join(OUT, "capability_ledger.jsonl"), "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # aggregated stacked-bar table
    table = frac_table(all_rows)
    with open(os.path.join(OUT, "ledger_categories.csv"), "w", newline="") as f:
        keys = ["model", "task", "n"] + BUCKETS
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in table:
            w.writerow(r)

    # availability matrix
    cols, cases = availability_matrix()
    with open(os.path.join(OUT, "column_availability.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "task"] + cols)
        for m, t, d in cases:
            w.writerow([m, t] + [d[c] for c in cols])

    ctx = reroll_context()
    summary = {"buckets_def": {
        "deployed_G1": "G==1 (direct greedy generation correct = production)",
        "scar_repairable": "G==0 & zero-MLP/scar edit makes it correct (mechanistic repair)",
        "recognition_gap": "G==0 & not scar-repaired & recognition(MC/loglik) correct",
        "sampling_reachable": "G==0 & not above & >=1 of K temperature samples correct",
        "path_reachable_only": "G==0 & not above & >=1 of K layer-programs correct",
        "unreached": "none of the available probes recover it within budget"},
        "KMAIN": KMAIN, "KS": KS,
        "categories": table, "availability": {"cols": cols,
        "cases": [{"model": m, "task": t, **d} for m, t, d in cases]},
        "selection_gap_reroll_aggregate_ONLY": ctx,
        "notes": [
            "M_K (majority@K / self-consistency) is NOT stored per-sample anywhere: "
            "fair-budget audit stored only first-hit RANKS (not per-candidate answers); "
            "reroll_sc stored only AGGREGATE majority@K/oracle@K. Per-sample M_K on the "
            "ledger samples needs a small re-run (store per-candidate answers, then vote).",
            "reroll aggregates use a DIFFERENT run (n, K, temp 0.8, decoding) than the "
            "fair-budget audit -- shown as context, NOT joined into the per-sample ledger.",
            "arc S_K/P_K/ROUT exist only for the first-300 audit intersection (verified "
            "identical samples to w15 first-300; audit orig==arm12 baseline 300/300).",
            "G for arc/dart audit rows = audit greedy 'orig' so S_K/P_K/router nest over G."]}
    with open(os.path.join(OUT, "ledger_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- console report
    print(f"\nwrote {len(all_rows)} ledger rows -> {OUT}/capability_ledger.jsonl")
    print("=" * 92)
    print("STACKED-BAR FRACTIONS (Figure 1), per model x task  [disjoint fate buckets]")
    print("=" * 92)
    hdr = f"{'model/task':<34}" + "".join(f"{b[:11]:>12}" for b in BUCKETS)
    print(hdr)
    for r in table:
        line = f"{r['model']+'/'+r['task']:<34}" + "".join(f"{r[b]:>12.3f}" for b in BUCKETS)
        print(line)
    print("\nColumn rates (over samples where the column exists):")
    for r in table:
        print(f"  {r['model']}/{r['task']} (n={r['n']}): "
              f"G={r['G_rate']} R={r['R_rate']} "
              f"S@200={r.get('S@200_rate')} Pmcts@200={r.get('Pmcts@200_rate')} "
              f"Prand@200={r.get('Prand@200_rate')} Zrepair={r['Z_repair_rate']} "
              f"ROUT={r.get('ROUT_rate')}")
        print(f"      [exact task cats] G0R1={r.get('cat_G0_R1')} "
              f"G0R1Z1(scar)={r.get('cat_G0_R1_Z1')} G0R0S1(reach-not-recog)={r.get('cat_G0_R0_S1')} "
              f"S1&M0={r.get('cat_S1_M0')}")
        print(f"      [deploy gaps] P@200&router0={r.get('path_reach_router_fail@200')} "
              f"S@200&deploy0={r.get('samp_reach_deploy_fail@200')}")
    print("\nSELECTION GAP (majority) -- AGGREGATE-ONLY reroll, NOT per-sample joinable:")
    for c in ctx:
        print(f"  {c['task']:>7} K={c['K']:>2} n={c['n']:>3}: single={c['single']} "
              f"majority@K={c['majority@K']} oracle@K={c['oracle@K']} "
              f"(oracle-maj=+{c['oracle_minus_majority']}, maj-single=+{c['majority_minus_single']})")
    print(f"\nwrote: ledger_categories.csv  column_availability.csv  ledger_summary.json")


if __name__ == "__main__":
    main()
