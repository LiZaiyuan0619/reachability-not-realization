#!/usr/bin/env python3
"""Anti-cherry-picking corpus builder. Scans every gap-screen result dir and reports
the realization gap with a FORMATTING-ROBUST metric (gap_len = mcll - gen_LENIENT),
which auto-corrects the strict-grading artifacts that plague verbose small models, and
flags generation-collapse artifacts (thinking-not-closed / fp16 overflow) that produce
fake gaps. Run from dr-llm/results.

Classes:
  REAL SCAR         gap_len>0.05                        (model knows via loglik, cannot say it lenient)
  FORMAT-artifact   strict gap>0.10 but gap_len<0.05    (only strict grading penalizes verbosity)
  COLLAPSE-artifact gen_lenient<0.10 while mcll>0.30    (generation broke -> not a realization scar)
  zero-gap          otherwise                           (clean control; prior-work gains here = reroll)
"""
import json, glob, os, collections, sys

def summ(f):
    try: return json.load(open(f))["summary"]
    except Exception: return None

def build(root="."):
    rows = collections.defaultdict(dict)
    dirs = sorted(set(glob.glob(f"{root}/w11_gap_*") + glob.glob(f"{root}/w15_bench_*")
                      + glob.glob(f"{root}/w13_stage2_*")))
    for d in dirs:
        model = os.path.basename(d)
        for gf in glob.glob(d + "/*_baseline.json"):
            t = os.path.basename(gf).replace("_baseline.json", "")
            ms = summ(gf.replace("_baseline.json", "_baseline_mcll.json")); gs = summ(gf)
            if not gs or not ms: continue
            gstrict, mcll = gs.get("acc_micro"), ms.get("acc_micro")
            glen = gs.get("acc_lenient"); glen = gstrict if glen is None else glen
            if gstrict is None or mcll is None: continue
            rows[model][t] = dict(gs=gstrict, gl=glen, m=mcll)
    return rows

def classify_cell(gs, gl, m):
    """Single-cell (one model,task) verdict from gen-strict gs, gen-lenient gl,
    mcll(letter-loglik) m. THE canonical anti-artifact thresholds -- reused by
    build_ledger_census.py so both tools share one source of truth.
      COLLAPSE gen broke (gl<0.10 while m>0.30)          -> fake gap, drop
      FORMAT   strict-only gap (gap_str>0.10, gap_len<0.05) -> grader artifact, drop
      REAL     format-robust recognition gap (gap_len>0.05)  -> genuine tier separation
      zero     otherwise                                     -> flat / no separation
    Returns (cls, gap_len, gap_str)."""
    gap_len, gap_str = m - gl, m - gs
    if gl < 0.10 and m > 0.30:              cls = "COLLAPSE"
    elif gap_str > 0.10 and gap_len < 0.05: cls = "FORMAT"
    elif gap_len > 0.05:                    cls = "REAL"
    else:                                   cls = "zero"
    return cls, gap_len, gap_str

def classify(ts):
    best = max(ts, key=lambda t: ts[t]["m"] - ts[t]["gl"])
    gs, gl, m = ts[best]["gs"], ts[best]["gl"], ts[best]["m"]
    cls, gap_len, gap_str = classify_cell(gs, gl, m)
    return best, gs, gl, m, gap_len, gap_str, cls

if __name__ == "__main__":
    rows = build(sys.argv[1] if len(sys.argv) > 1 else ".")
    buckets = collections.defaultdict(list)
    for model in sorted(rows):
        if not rows[model]: continue
        best, gs, gl, m, gapL, gapS, cls = classify(rows[model])
        buckets[cls].append((model, best, gapL, gapS))
        print(f"{model:30s}{best:>18}{gs:>8.3f}{gl:>7.3f}{m:>7.3f}{gapL:>+8.3f}{gapS:>+8.3f}  {cls}")
    print(f"\nREAL SCAR ({len(buckets['REAL'])}) | FORMAT-artifact ({len(buckets['FORMAT'])}) | "
          f"COLLAPSE-artifact ({len(buckets['COLLAPSE'])}) | zero-gap controls ({len(buckets['zero'])})")
