#!/usr/bin/env python3
"""Zero-GPU consolidation for §5 (07-15). Runs on a worker via rjob (CPU-only).
Produces two writing-ready artifacts from existing reroll records:
  (1) DOSE-RESPONSE table: sharpening (Δrand@K) vs contraction (Δoracle, Δreachset)
      across all RL/instruct pairs -> turns 'SimpleRL doesn't contract' into the
      mild end of a monotone axis (contraction scales with RL aggressiveness).
  (2) PROPENSITY LAW 2nd family (SimpleRL): base latent items binned by per-item
      sample-hit-rate -> fraction monetized (latent->solid). Tests whether RL only
      redistributes already-high-frequency base solutions (as in the Olmo main case).
Defensive: skips any pair whose files are absent (re-run after P1/P2 land).
Output: printed + written to results/lens/consolidate_0715_report.txt
"""
import json, os, glob
from collections import Counter, defaultdict
L = "./results/lens"
OUT = []
def say(*a): line=" ".join(str(x) for x in a); print(line); OUT.append(line)

def load(tag):
    p = f"{L}/reroll_{tag}_t08.json"
    return json.load(open(p)) if os.path.exists(p) else None

def bucket(rec, K):
    reach = any(a == rec["gold"] for a in rec["sample_answers"][:K])
    return "solid" if rec["single_hit"] else ("latent" if reach else "unreach")

# ------------------------------------------------------------------ (1) dose-response
# (base_tag, rl_tag, label, kind)
PAIRS = [
    ("q3535b", "q3535i", "Qwen3.5-35B-A3B", "base->instruct (full pipeline)"),
    ("fateHi_srbase", "fateHi_srzero", "SimpleRL-7B", "from-base RL-Zero (mild)"),
    ("fateHi_q32base", "fateHi_dapo32", "DAPO-32B", "from-base RL-Zero (aggressive)"),
    ("fateHi_srbase", "fateHi_or1m7", "OR1-7B(vs srbase K64)", "RL-on-distill (harden 3072)"),
    ("r1d7", "or1big", "OR1-7B(vs r1d7, 8k)", "RL-on-distill (P1 clean)"),
    ("srbase", "r1d7", "R1-Distill-7B", "distill"),
    ("qwenlad_q32base", "r1d32", "R1-Distill-32B", "distill"),
]
# try to discover an Olmo pair (main propensity family)
for cand in [("olmo2base","rlzm"),("olmobase","rlzm"),("base_olmo","rlzm")]:
    if load(f"{cand[0]}_gsm8k") and load(f"{cand[1]}_gsm8k"):
        PAIRS.append((cand[0], cand[1], "Olmo-2-7B", "from-base RL-Zero (main case)")); break

def dose(base_tag, rl_tag, task):
    b = load(f"{base_tag}_{task}"); r = load(f"{rl_tag}_{task}")
    if not b or not r: return None
    K = min(b["config"]["K"], r["config"]["K"])
    ab, ar = b["acc"], r["acc"]
    rb = {x["qid"]: x for x in b["records"]}; rr = {x["qid"]: x for x in r["records"]}
    common = sorted(set(rb) & set(rr))
    c = Counter()
    for q in common: c[(bucket(rb[q], K), bucket(rr[q], K))] += 1
    reach_b = sum(v for (x, _), v in c.items() if x in ("solid", "latent"))
    reach_r = sum(v for (_, y), v in c.items() if y in ("solid", "latent"))
    lat = sum(v for (x, _), v in c.items() if x == "latent")
    l2s = c.get(("latent", "solid"), 0)
    return dict(K=K, n=len(common), dsingle=ar["single"]-ab["single"],
                drand=ar["random@K"]-b["acc"]["random@K"], doracle=ar["oracle@K"]-ab["oracle@K"],
                dreach=reach_r-reach_b, monetize=l2s/max(lat,1))

say("="*96); say(" (1) DOSE-RESPONSE: sharpening (Δrand@K) vs contraction (Δoracle / Δreachset)"); say("="*96)
say(f"{'pair':22s} {'kind':30s} {'task':5s} {'K':>4} {'Δsingle':>8} {'Δrand@K':>8} {'Δoracle':>8} {'Δreach':>7} {'monet':>6}")
rows=[]
for bt, rt, lbl, kind in PAIRS:
    for task in ("gsm8k","dart"):
        d = dose(bt, rt, task)
        if d is None: continue
        rows.append((lbl,kind,task,d))
        say(f"{lbl:22s} {kind:30s} {task:5s} {d['K']:>4} {d['dsingle']:>+8.3f} {d['drand']:>+8.3f} {d['doracle']:>+8.3f} {d['dreach']:>+7d} {d['monetize']:>6.1%}")
say("\n  read: as Δrand@K (RL sharpening) rises, Δoracle/Δreach (contraction) becomes more negative.")
say("  SimpleRL = mild sharpen, no contract; DAPO = hard sharpen, strong contract; instruct = near-no-op.")

# ------------------------------------------------------------------ (2) propensity law 2nd family
def propensity(base_tag, rl_tag, task, label):
    b = load(f"{base_tag}_{task}"); r = load(f"{rl_tag}_{task}")
    if not b or not r: return
    K = min(b["config"]["K"], r["config"]["K"])
    rb = {x["qid"]: x for x in b["records"]}; rr = {x["qid"]: x for x in r["records"]}
    common = sorted(set(rb) & set(rr))
    bins = [(0,0.02),(0.02,0.05),(0.05,0.15),(0.15,0.35),(0.35,1.01)]
    agg = {bn: [0,0] for bn in bins}  # [monetized, total] among base-latent items
    for q in common:
        rec = rb[q]
        if bucket(rec, K) != "latent": continue          # base latent = sampling-reachable, greedy-wrong
        rate = sum(1 for a in rec["sample_answers"][:K] if a == rec["gold"]) / K  # per-item hit-rate @matched K
        became_solid = rr[q]["single_hit"]
        for lo,hi in bins:
            if lo <= rate < hi: agg[(lo,hi)][0]+=int(became_solid); agg[(lo,hi)][1]+=1; break
    say(f"\n  {label} [{task}] (K={K}) — base-latent items, monetized(latent->solid) by base sample-hit-rate:")
    for (lo,hi),(m,t) in agg.items():
        say(f"    hit {lo:.0%}-{hi:.0%}: {m}/{t} = {m/max(t,1):.0%}" + ("" if t else "  (empty)"))

say("\n"+"="*96); say(" (2) PROPENSITY LAW — 2nd family (SimpleRL); expect monotone rising (redistribute high-freq)"); say("="*96)
propensity("fateHi_srbase","fateHi_srzero","dart","SimpleRL-7B")
propensity("fateHi_srbase","fateHi_srzero","gsm8k","SimpleRL-7B")
propensity("fateHi_q32base","fateHi_dapo32","dart","DAPO-32B")

open(f"{L}/consolidate_0715_report.txt","w").write("\n".join(OUT)+"\n")
say(f"\n[written] {L}/consolidate_0715_report.txt")
