"""Build the capability ledger (per-sample fate matrix) for Llama-3.1-8B-Instruct /
arc_challenge -- the SECOND model panel of Figure 1 (companion to tools/build_ledger.py,
which stays untouched and covers Llama-3.2-3B + gemma-4-12B).

Same column semantics and the same disjoint waterfall fate buckets as build_ledger.py:
  G     direct-generation correct (greedy)                       [production]
  R     recognition correct (MC letter-loglik, mcll acc)         [recognition]
  S_K   >=1 of K temperature samples correct (samp arm)          [sampling reach]
  P_K   >=1 of K layer-programs correct -- RAND arm (control)    [path reach]
  Pmcts MCTS-arm path reach, kept as a separate column           [path reach, search]
  M_K   majority-vote/self-consistency @K, PER-SAMPLE            [selection]
  C_K   BoN-by-confidence @K, PER-SAMPLE                         [selection]
  Z     scar repair: ALL ZERO BY CONSTRUCTION for this model -- Llama-3.1-8B has no
        identified scar layer and no zmlp edit run. The scar_repairable bucket is
        therefore structurally empty: the absence of the repair layer is itself the
        honest reading for a scar-free model.
  ROUT  N/A -- no layer-router was ever trained for Llama-3.1-8B; not fabricated.

!! BUDGET NOTE (updated 2026-07-10): the DEFAULT source is now the kmax=200 rerun
   w8_fairbudget_v2/l318bk200/fairv2_arc_challenge_seed0.json, so K in {8,16,64,200}
   and the headline waterfall uses KMAIN=200 -- the SAME audit budget as the 3B panel
   (samp/rand store 200 candidates each; mcts terminates early, <=200, kept as-is).
   Fate P arms still differ across panels (3B fate used mcts; 8B fate uses rand per
   spec). The old kmax=64 file (w8_fairbudget_v2/l318b/...) remains selectable via
   --fairbudget; its greedy 'orig' agrees with the k200 rerun 300/300 (checked when
   the legacy file is present).

Sources (all Llama-3.1-8B-Instruct 0e9e39f2, n=300, same sample order -- verified by
the per-sample alignment assert in build(), which hard-fails below 90% agreement):
  w11_gap_prl318bs0m/arc_challenge_baseline.json       -> G_gen_w11 (unpruned greedy)
  w11_gap_prl318bs0m/arc_challenge_baseline_mcll.json  -> R (acc = letter loglik),
                                                          R_len (acc_len = option text)
  w8_fairbudget_v2/l318bk200/fairv2_arc_challenge_seed0.json
        -> orig (audit greedy; used as fate G so S/P/M/C nest over G by construction,
           candidate #1 == greedy in every arm), S_K (samp), P_K (rand), Pmcts (mcts),
           M_K / C_K per arm from the stored per-candidate answers/correct/conf arrays.

Majority uses the FIXED abstain rule (empty answer = no vote), imported straight from
fair_budget_analyze.majority_hit -- NOT the old logic that let empty strings vote.

Pure data-reorganization, read-only except the ledger outputs. Run from dr-llm/:
    python tools/build_ledger_8b.py             # main path
    python tools/build_ledger_8b.py --selftest  # pure logic, reads no result files

Outputs (results/ledger/ preferred; falls back to logs2/ledger8b on PermissionError):
    ledger_8b_arc_challenge.csv   per-sample fate matrix (300 rows)
    ledger_8b_summary.json        bucket fractions + column rates + selection gaps
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
OUT_PRIMARY = os.path.join(RES, "ledger")
OUT_FALLBACK = "outputs/ledger8b"

sys.path.insert(0, ROOT)
from fair_budget_analyze import confidence_hit, majority_hit, oracle_hit  # noqa: E402

MODEL = "Llama-3.1-8B"
TASK = "arc_challenge"
KS = [8, 16, 64, 200]  # audit budgets materialised as binary flags (64 kept for
                       # continuity with the earlier kmax=64 audit)
KMAIN = 200            # kmax of the k200 fairv2 rerun == the 3B panel budget
ALIGN_MIN = 0.90       # hard floor for the per-sample greedy alignment across runs

F_BASE = "w11_gap_prl318bs0m/arc_challenge_baseline.json"
F_MCLL = "w11_gap_prl318bs0m/arc_challenge_baseline_mcll.json"
F_FAIR = "w8_fairbudget_v2/l318bk200/fairv2_arc_challenge_seed0.json"
F_FAIR_K64 = "w8_fairbudget_v2/l318b/fairv2_arc_challenge_seed0.json"  # legacy audit

BUCKETS = ["deployed_G1", "scar_repairable", "recognition_gap",
           "sampling_reachable", "path_reachable_only", "unreached"]

ARMS = [("samp", "S"), ("rand", "P"), ("mcts", "Pmcts")]  # fairv2 key -> ledger prefix


def load(path):
    with open(os.path.join(RES, path)) as f:
        return json.load(f)


def reach_at(rank, k):
    """rank is 1-indexed rank of first correct candidate, -1/None if none."""
    return 1 if (rank is not None and 1 <= rank <= k) else 0


# --------------------------------------------------------------------------- classify
def fate(row, avail):
    """Disjoint waterfall bucket (identical priority to build_ledger.py):
    deployed > scar-repairable > recognition-gap > sampling-reachable >
    path-reachable-only > unreached. Only consults columns named in `avail`."""
    if row["G"] == 1:
        return "deployed_G1"
    if "Z" in avail and row.get("Z_repair") == 1:
        return "scar_repairable"
    if "R" in avail and row.get("R") == 1:
        return "recognition_gap"
    if "S_K" in avail and row.get(f"S@{KMAIN}") == 1:
        return "sampling_reachable"
    if "P_K" in avail and row.get(f"P@{KMAIN}") == 1:
        return "path_reachable_only"
    return "unreached"


# --------------------------------------------------------------------------- alignment
def check_alignment(fb_recs, bl_recs, min_rate=ALIGN_MIN, verbose=True):
    """Per-sample compare fairv2 audit greedy (orig) vs w11_gap baseline greedy (acc).
    Both are greedy on the same prompts -> should be near-100%; below `min_rate` we
    refuse to index-join and exit after dumping the first 10 mismatches (both sides)."""
    n = len(fb_recs)
    assert n == len(bl_recs), f"length mismatch fairv2={n} baseline={len(bl_recs)}"
    mism = [i for i in range(n)
            if (fb_recs[i]["orig"] > 0.5) != (bl_recs[i]["acc"] > 0.5)]
    rate = (n - len(mism)) / n
    detail = [{"idx": i,
               "fairv2": {"orig": fb_recs[i]["orig"],
                          "orig_answer": fb_recs[i].get("orig_answer"),
                          "orig_conf": fb_recs[i].get("orig_conf")},
               "w11_baseline": {"acc": bl_recs[i]["acc"],
                                "acc_len": bl_recs[i].get("acc_len"),
                                "response": bl_recs[i].get("response"),
                                "gold": bl_recs[i].get("correct")}}
              for i in mism[:10]]
    if verbose:
        print(f"[align] fairv2.orig vs w11_gap.acc: {n - len(mism)}/{n} agree "
              f"= {rate:.4f} (floor {min_rate})")
        for d in detail:
            print(f"    mismatch idx={d['idx']}: fairv2={d['fairv2']} "
                  f"| w11={d['w11_baseline']}")
    if rate < min_rate:
        raise SystemExit(
            f"ALIGNMENT FAILURE: greedy agreement {rate:.4f} < {min_rate} -- the two "
            f"runs do not look like the same samples/model; refusing to index-join. "
            f"First mismatches: {json.dumps(detail, ensure_ascii=False)}")
    return {"n": n, "agree": n - len(mism), "rate": round(rate, 6),
            "n_mismatch": len(mism), "mismatches_first10": detail}


# --------------------------------------------------------------------------- build
def build(fair_path=F_FAIR):
    bl = load(F_BASE)
    mc = load(F_MCLL)
    fb = load(fair_path)
    blr, mcr, fbr = bl["records"], mc["records"], fb["records"]
    assert bl["summary"]["n"] == mc["summary"]["n"] == fb["summary"]["n"] == 300
    assert len(blr) == len(mcr) == len(fbr) == 300
    assert fb["summary"]["kmax"] == KMAIN, \
        f"expected kmax={KMAIN}, file says {fb['summary']['kmax']} -- this script's " \
        f"K grid/waterfall is pinned to KMAIN={KMAIN}; pass a kmax={KMAIN} fairv2 file"
    assert mc["summary"].get("mc_loglik") is True, "mcll file lacks mc_loglik marker"
    # gold labels must be literally identical between the two w11 runs
    assert [r["correct"] for r in blr] == [r["correct"] for r in mcr], \
        "gold mismatch between baseline and mcll -- not the same sample order"

    align = check_alignment(fbr, blr)

    # provenance probe: legacy kmax=64 audit (if present) must be the same greedy run
    if os.path.exists(os.path.join(RES, F_FAIR_K64)) and fair_path != F_FAIR_K64:
        old = load(F_FAIR_K64)["records"]
        same = sum(1 for i in range(300)
                   if (old[i]["orig"] > 0.5) == (fbr[i]["orig"] > 0.5))
        same_ans = sum(1 for i in range(300)
                       if old[i].get("orig_answer") == fbr[i].get("orig_answer"))
        print(f"[align] k200 rerun vs legacy k64 audit greedy: {same}/300 flags, "
              f"{same_ans}/300 identical answers")
        align["k200_vs_k64_orig"] = {"agree_flags": same, "agree_answers": same_ans}

    # candidate #1 == greedy in every arm (fair-budget construction) -> S/P nest over G
    for arm, _ in ARMS:
        bad = [i for i, r in enumerate(fbr)
               if (float(r[f"{arm}_correct"][0]) > 0.5) != (r["orig"] > 0.5)]
        assert not bad, f"{arm} candidate#1 != greedy at idx {bad[:5]}"

    rows = []
    for i in range(300):
        f, b, m = fbr[i], blr[i], mcr[i]
        g = 1 if f["orig"] > 0.5 else 0   # audit greedy = fate G (nesting; see header)
        row = {"model": MODEL, "task": TASK, "idx": i, "gold": b["correct"],
               "G": g,
               "G_gen_w11": 1 if b["acc"] > 0.5 else 0,
               "R": 1 if m["acc"] > 0.5 else 0,          # letter loglik (headline R)
               "R_len": 1 if m["acc_len"] > 0.5 else 0,  # option-text loglik (aux)
               # scar-free model: no repair layer exists -> structurally zero (header)
               "Z_acc": None, "Z_repair": 0,
               "ROUT": None,                              # N/A: no 8B router trained
               "orig_answer": f.get("orig_answer"), "orig_conf": f.get("orig_conf")}
        for arm, key in ARMS:
            rank = f[f"{arm}_rank"]
            row[f"{key}_rank"] = rank
            ans, cor, cnf = f[f"{arm}_answers"], f[f"{arm}_correct"], f[f"{arm}_conf"]
            for k in KS:
                row[f"{key}@{k}"] = reach_at(rank, k)
                row[f"M{arm}@{k}"] = majority_hit(ans, cor, k)   # abstain rule
                row[f"C{arm}@{k}"] = confidence_hit(cor, cnf, k)
                # rank-based reach must equal oracle over stored per-candidate grades
                assert row[f"{key}@{k}"] == oracle_hit(cor, k), \
                    f"idx {i} {arm}@{k}: rank({rank}) vs oracle disagree"
        row["fate"] = fate(row, {"G", "R", "Z", "S_K", "P_K"})
        rows.append(row)
    return rows, fb["summary"], align


# --------------------------------------------------------------------------- aggregate
def aggregate(rows):
    n = len(rows)
    frac = {b: round(sum(1 for r in rows if r["fate"] == b) / n, 4) for b in BUCKETS}

    def rate(pred):
        return round(sum(1 for r in rows if pred(r)) / n, 4)

    col = {"G_rate(orig)": rate(lambda r: r["G"] == 1),
           "G_gen_w11_rate": rate(lambda r: r["G_gen_w11"] == 1),
           "R_rate": rate(lambda r: r["R"] == 1),
           "R_len_rate": rate(lambda r: r["R_len"] == 1),
           "Z_repair_rate": 0.0,
           "ROUT_rate": "N/A (no router trained for Llama-3.1-8B)"}
    for _, key in ARMS:
        for k in KS:
            col[f"{key}@{k}_rate"] = rate(lambda r, key=key, k=k: r[f"{key}@{k}"] == 1)
    # cross-tabs (overlapping, NOT the disjoint bar)
    col["cat_G0_R1"] = rate(lambda r: r["G"] == 0 and r["R"] == 1)
    col["cat_G0_R0_S1"] = rate(
        lambda r: r["G"] == 0 and r["R"] == 0 and r[f"S@{KMAIN}"] == 1)
    col[f"cat_S1_M0@{KMAIN}"] = rate(
        lambda r: r[f"S@{KMAIN}"] == 1 and r[f"Msamp@{KMAIN}"] == 0)

    sel = {}
    for arm, key in ARMS:
        sel[arm] = {}
        for k in KS:
            o = rate(lambda r, key=key, k=k: r[f"{key}@{k}"] == 1)
            m = rate(lambda r, arm=arm, k=k: r[f"M{arm}@{k}"] == 1)
            c = rate(lambda r, arm=arm, k=k: r[f"C{arm}@{k}"] == 1)
            sel[arm][k] = {"oracle": o, "majority": m, "confidence": c,
                           "gap_oracle_minus_majority": round(o - m, 4),
                           "gap_oracle_minus_confidence": round(o - c, 4)}
    return frac, col, sel


def crosscheck(sel, fb_summary):
    """Recomputed per-sample aggregates must reproduce the fairv2 file's own summary
    (same majority_hit/confidence_hit code path) -- catches any schema misread.
    K values absent from the stored summary grid (the k200 file stores
    {1,2,4,5,8,16,32,200} -- no 64) are skipped and reported, not silently dropped."""
    checks, skipped, worst = [], [], 0.0
    for arm in ("samp", "rand", "mcts"):
        for k in KS:
            for mine, theirs in [("oracle", "passk"), ("majority", "majority"),
                                 ("confidence", "confidence")]:
                stored = fb_summary[f"{arm}_{theirs}"]
                if str(k) not in stored:
                    skipped.append(f"{arm}_{theirs}@{k}")
                    continue
                got = sel[arm][k][mine]
                want = stored[str(k)]
                d = abs(got - round(want, 4))
                worst = max(worst, d)
                checks.append({"arm": arm, "k": k, "metric": mine,
                               "recomputed": got, "stored": round(want, 6),
                               "abs_diff": round(d, 6)})
                assert d <= 5e-4, f"crosscheck FAIL {arm} {theirs}@{k}: {got} vs {want}"
    return {"n_checks": len(checks), "max_abs_diff": worst, "status": "PASS",
            "skipped_k_not_in_stored_summary": skipped, "detail": checks}


# --------------------------------------------------------------------------- output
def resolve_outdir():
    try:
        os.makedirs(OUT_PRIMARY, exist_ok=True)
        probe = os.path.join(OUT_PRIMARY, ".ledger8b_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return OUT_PRIMARY, False
    except (PermissionError, OSError) as e:
        print(f"[out] cannot write {OUT_PRIMARY} ({e}); falling back to {OUT_FALLBACK}")
        os.makedirs(OUT_FALLBACK, exist_ok=True)
        return OUT_FALLBACK, True


def csv_fieldnames():
    cols = ["model", "task", "idx", "gold", "G", "G_gen_w11", "R", "R_len",
            "Z_acc", "Z_repair", "ROUT", "orig_answer", "orig_conf"]
    for arm, key in ARMS:
        cols.append(f"{key}_rank")
        cols += [f"{key}@{k}" for k in KS]
        cols += [f"M{arm}@{k}" for k in KS]
        cols += [f"C{arm}@{k}" for k in KS]
    cols.append("fate")
    return cols


# --------------------------------------------------------------------------- 3B vs 8B
def compare_with_3b(frac8b):
    """Read the existing 3B summary (if present) and print the two Figure-1 panels
    side by side. Both waterfalls are now at K=200 (same audit budget); the remaining
    asymmetry is the fate P arm (3B used mcts, 8B uses rand per spec)."""
    p3 = os.path.join(OUT_PRIMARY, "ledger_summary.json")
    if not os.path.exists(p3):
        print(f"[3B] {p3} not found -- skipping comparison table")
        return None
    cats = json.load(open(p3))["categories"]
    row3 = next((c for c in cats
                 if c["model"] == "Llama-3.2-3B" and c["task"] == "arc_challenge"), None)
    if row3 is None:
        print("[3B] no Llama-3.2-3B/arc_challenge row in 3B summary -- skipping")
        return None
    print("\n" + "=" * 88)
    print(f"FIGURE-1 PANELS: 3B (K=200, P=mcts) vs 8B (K={KMAIN}, P=rand) "
          f"-- arc_challenge, n=300")
    print("=" * 88)
    print(f"{'bucket':<24}{'Llama-3.2-3B':>16}{'Llama-3.1-8B':>16}{'delta(8B-3B)':>16}")
    comp = {}
    for b in BUCKETS:
        v3, v8 = row3[b], frac8b[b]
        comp[b] = {"3B": v3, "8B": v8, "delta": round(v8 - v3, 4)}
        print(f"{b:<24}{v3:>16.3f}{v8:>16.3f}{v8 - v3:>+16.3f}")
    print("(both audits kmax=200 now; fate P arms still differ: 3B P-arm=mcts, "
          "8B P-arm=rand -- see notes)")
    return comp


# --------------------------------------------------------------------------- selftest
def _selftest():
    """Pure logic -- touches NO result files."""
    # reach_at boundaries
    assert reach_at(1, 8) == 1 and reach_at(8, 8) == 1 and reach_at(9, 8) == 0
    assert reach_at(-1, 64) == 0 and reach_at(None, 8) == 0
    # waterfall branches (incl. structurally-zero Z never firing)
    avail = {"G", "R", "Z", "S_K", "P_K"}
    assert fate({"G": 1}, avail) == "deployed_G1"
    assert fate({"G": 0, "Z_repair": 1}, avail) == "scar_repairable"  # never for 8B
    assert fate({"G": 0, "Z_repair": 0, "R": 1}, avail) == "recognition_gap"
    assert fate({"G": 0, "Z_repair": 0, "R": 0, f"S@{KMAIN}": 1},
                avail) == "sampling_reachable"
    assert fate({"G": 0, "Z_repair": 0, "R": 0, f"S@{KMAIN}": 0, f"P@{KMAIN}": 1},
                avail) == "path_reachable_only"
    assert fate({"G": 0, "Z_repair": 0, "R": 0, f"S@{KMAIN}": 0, f"P@{KMAIN}": 0},
                avail) == "unreached"
    # scar-free model: Z_repair==0 on every row -> bucket empty by construction
    assert all(fate({"G": 0, "Z_repair": 0, "R": r, f"S@{KMAIN}": 0, f"P@{KMAIN}": 0},
                    avail) != "scar_repairable" for r in (0, 1))
    # imported majority is the ABSTAIN version (empty strings cast no vote)
    assert majority_hit(["", "", "A"], [0, 0, 1], 3) == 1   # old empty-vote logic -> 0
    assert majority_hit(["", "", ""], [0, 0, 0], 3) == 0    # all abstain -> fail
    assert majority_hit(["B", "A", "A"], [1, 0, 0], 3) == 0  # majority picks wrong A
    assert majority_hit(["B", "A", "A"], [1, 0, 0], 1) == 1  # @1 == greedy
    assert confidence_hit([1, 0], [-0.3, -0.3], 2) == 1      # tie -> earliest
    assert oracle_hit([0, 0, 1], 2) == 0 and oracle_hit([0, 0, 1], 3) == 1
    # alignment checker: pass at 100%, hard-exit below the 90% floor
    ok_fb = [{"orig": 1.0, "orig_answer": "A", "orig_conf": -0.1}] * 10
    ok_bl = [{"acc": 1.0, "acc_len": 1.0, "response": "A", "correct": "A"}] * 10
    a = check_alignment(ok_fb, ok_bl, verbose=False)
    assert a["rate"] == 1.0 and a["n_mismatch"] == 0
    bad_bl = [dict(b, acc=0.0) for b in ok_bl[:5]] + list(ok_bl[:5])
    try:
        check_alignment(ok_fb, bad_bl, verbose=False)
        raise AssertionError("alignment checker failed to hard-exit at 50%")
    except SystemExit as e:
        assert "ALIGNMENT FAILURE" in str(e)
    # aggregation: disjoint fractions sum to 1
    rows = [{"fate": "deployed_G1", "G": 1, "G_gen_w11": 1, "R": 1, "R_len": 1},
            {"fate": "unreached", "G": 0, "G_gen_w11": 0, "R": 0, "R_len": 0}]
    for r in rows:
        for _, key in ARMS:
            for k in KS:
                r[f"{key}@{k}"] = r["G"]
                r[f"M{ {'S':'samp','P':'rand','Pmcts':'mcts'}[key] }@{k}"] = r["G"]
                r[f"C{ {'S':'samp','P':'rand','Pmcts':'mcts'}[key] }@{k}"] = r["G"]
    frac, col, sel = aggregate(rows)
    assert abs(sum(frac.values()) - 1.0) < 1e-9
    assert sel["samp"][64]["gap_oracle_minus_majority"] == 0.0
    print("SELFTEST OK (reach_at / waterfall+empty-scar / abstain-majority / "
          "confidence / alignment gate / aggregation)")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fairbudget", default=F_FAIR,
                    help=f"fairv2 source (relative to results/), must be kmax={KMAIN}; "
                         f"default {F_FAIR}")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    rows, fb_summary, align = build(args.fairbudget)
    frac, col, sel = aggregate(rows)
    xchk = crosscheck(sel, fb_summary)
    outdir, fell_back = resolve_outdir()

    csv_path = os.path.join(outdir, f"ledger_8b_{TASK}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fieldnames(), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    comp = compare_with_3b(frac)
    mK = sel["samp"][KMAIN]
    summary = {
        "model": MODEL, "task": TASK, "n": len(rows),
        "model_snapshot": fb_summary.get("model_path"),
        "KS": KS, "KMAIN": KMAIN,
        "kmax_note": f"8B fairv2 k200 rerun stores kmax={KMAIN} candidates/arm for "
                     "samp and rand (mcts terminates early, <=200), matching the 3B "
                     "w8_fairbudget audit budget. K=64 flags kept for continuity with "
                     "the earlier kmax=64 audit, whose greedy agrees with this rerun "
                     "300/300 (see alignment block).",
        "sources": {"G_gen_w11+gold": F_BASE, "R(mcll)": F_MCLL,
                    "orig/S/P/Pmcts/M/C": args.fairbudget},
        "alignment_fairv2_orig_vs_w11_acc": align,
        "buckets_def": {
            "deployed_G1": "G==1 (audit greedy generation correct = production)",
            "scar_repairable": "G==0 & scar edit repairs it -- STRUCTURALLY EMPTY: "
                               "Llama-3.1-8B has no identified scar layer / no zmlp "
                               "repair run; Z_repair==0 for all samples. The absence "
                               "of the repair layer is itself the honest reading for "
                               "a scar-free model.",
            "recognition_gap": "G==0 & recognition (MC letter-loglik) correct",
            "sampling_reachable": f"G==0 & not above & >=1 of {KMAIN} temperature "
                                  "samples correct (samp arm)",
            "path_reachable_only": f"G==0 & not above & >=1 of {KMAIN} RAND-arm layer "
                                   "programs correct (mcts kept as separate column)",
            "unreached": "none of the available probes recover it within budget"},
        "fate_fractions": frac,
        "column_rates": col,
        "selection_per_arm": sel,
        "headline_selection_gap": {
            "arm": "samp", "K": KMAIN,
            f"oracle@{KMAIN}": mK["oracle"], f"majority@{KMAIN}": mK["majority"],
            f"confidence@{KMAIN}": mK["confidence"],
            "oracle_minus_majority": mK["gap_oracle_minus_majority"]},
        "crosscheck_vs_fairv2_summary": {k: xchk[k] for k in
                                         ("n_checks", "max_abs_diff", "status",
                                          "skipped_k_not_in_stored_summary")},
        "availability": {"G": "y", "R": "y", "S_K": "y(kmax=200)",
                         "P_K": "y(rand,kmax=200)",
                         "Pmcts_K": "y(search-terminated, <=200 cands)",
                         "M_K": "y(per-sample, abstain rule)",
                         "C_K": "y(per-sample)",
                         "Z": "structurally 0 (scar-free model, no zmlp run)",
                         "ROUT": "N/A (no router trained for Llama-3.1-8B; "
                                 "not fabricated)"},
        "comparison_3b_vs_8b": comp,
        "notes": [
            "Z (scar repair) is all-zero BY CONSTRUCTION: no scar case exists for "
            "Llama-3.1-8B, so no repair run was performed and the scar_repairable "
            "bucket is empty. The absence of the repair layer is itself the honest "
            "reading for a scar-free model -- do not read the empty bucket as a "
            "measured zero repair rate.",
            "ROUT is N/A (no 8B router checkpoint exists); the column is left null "
            "rather than fabricated, and the fate waterfall never consults it.",
            "Fate G = fairv2 audit greedy 'orig' so S/P/M/C nest over G by "
            "construction (candidate #1 == greedy in every arm, asserted). The "
            "independent w11_gap greedy run is kept as column G_gen_w11; per-sample "
            "agreement between the two greedy runs is reported under 'alignment'.",
            "P_K (fate path reach) uses the RAND arm per spec (random layer-program "
            "control); the MCTS arm is materialised separately as Pmcts@k.",
            "M_K majority uses fair_budget_analyze.majority_hit: empty answers "
            "ABSTAIN (no vote); the pre-fix logic that let empty strings vote is "
            "NOT used anywhere here.",
            "R uses mcll 'acc' (letter loglik, headline recognition); 'acc_len' "
            "(option-text mean loglik) kept as R_len.",
            "3B and 8B waterfalls are now at the SAME audit budget (kmax=200); the "
            "remaining asymmetry is the fate P arm (3B mcts vs 8B rand) -- flagged "
            "in the table. Crosscheck at K=64 vs the stored summary is impossible "
            "(k200 file's summary grid lacks 64) and is reported as skipped; the "
            "per-sample rank-vs-oracle assert still covers K=64."]}
    json_path = os.path.join(outdir, f"ledger_8b_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- console report
    print("\n" + "=" * 88)
    print(f"8B LEDGER -- {MODEL} / {TASK} (n={len(rows)}, KMAIN={KMAIN})")
    print("=" * 88)
    print("fate buckets (disjoint waterfall):")
    for b in BUCKETS:
        extra = "  <- structurally empty (scar-free model)" if b == "scar_repairable" else ""
        print(f"  {b:<22}{frac[b]:>8.4f}  ({frac[b]*100:5.1f}%){extra}")
    print(f"\ncolumn rates: G(orig)={col['G_rate(orig)']} G_w11={col['G_gen_w11_rate']} "
          f"R={col['R_rate']} R_len={col['R_len_rate']}")
    for _, key in ARMS:
        print(f"  {key:<6}" + "  ".join(f"@{k}={col[f'{key}@{k}_rate']}" for k in KS))
    print(f"  cross-tabs: G0R1={col['cat_G0_R1']} G0R0S1={col['cat_G0_R0_S1']} "
          f"S1&M0@{KMAIN}={col[f'cat_S1_M0@{KMAIN}']}")
    print("\nselection (oracle / majority / confidence) per arm:")
    for arm in ("samp", "rand", "mcts"):
        for k in KS:
            s = sel[arm][k]
            print(f"  {arm:>5}@{k:<3} oracle={s['oracle']:.4f} maj={s['majority']:.4f} "
                  f"conf={s['confidence']:.4f} (oracle-maj={s['gap_oracle_minus_majority']:+.4f})")
    print(f"\nHEADLINE M_K@{KMAIN} vs oracle@{KMAIN} (samp arm): "
          f"majority={mK['majority']:.4f} vs oracle={mK['oracle']:.4f} "
          f"-> gap {mK['gap_oracle_minus_majority']:+.4f}")
    print(f"crosscheck vs fairv2 stored summary: {xchk['status']} "
          f"({xchk['n_checks']} checks, max|diff|={xchk['max_abs_diff']:.6f}, "
          f"skipped {len(xchk['skipped_k_not_in_stored_summary'])} K-not-in-grid)")
    print(f"ROUT: N/A -- no router trained for {MODEL} (not fabricated)")
    print(f"\nwrote: {csv_path}")
    print(f"wrote: {json_path}" + ("  [FALLBACK dir]" if fell_back else ""))


if __name__ == "__main__":
    main()
