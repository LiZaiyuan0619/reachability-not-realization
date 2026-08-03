#!/usr/bin/env python3
"""Clean-pair fate matrices for §5 (RLVR redistribute). Zero-GPU: uses existing
reroll records (oracle_hit + sample_answers). K matched via sample_answers truncation.
Pairs = RL-Zero (from-base) + distill, all Qwen chat (same-format, no Olmo confound)."""
import json
from collections import Counter
L = "./results/lens"

def load(tag, task, tt="t08"):
    return json.load(open(f"{L}/reroll_{tag}_{task}_{tt}.json"))

def bucket(rec, K):
    gold = rec["gold"]
    reach = any(a == gold for a in rec["sample_answers"][:K])
    return "solid" if rec["single_hit"] else ("latent" if reach else "unreach")

def fate(base_tag, rl_tag, task, label, Kover=None):
    b = load(base_tag, task); r = load(rl_tag, task)
    Kb, Kr = b["config"]["K"], r["config"]["K"]
    K = Kover or min(Kb, Kr)
    rb = {x["qid"]: x for x in b["records"]}; rr = {x["qid"]: x for x in r["records"]}
    common = sorted(set(rb) & set(rr))
    c = Counter()
    for q in common:
        c[(bucket(rb[q], K), bucket(rr[q], K))] += 1
    n = len(common)
    lat = sum(v for (x, _), v in c.items() if x == "latent")
    l2s = c.get(("latent", "solid"), 0)
    sol = sum(v for (x, _), v in c.items() if x == "solid")
    sol_lost = sum(v for (x, y), v in c.items() if x == "solid" and y != "solid")
    l2u = c.get(("latent", "unreach"), 0)
    unr = sum(v for (x, _), v in c.items() if x == "unreach")
    # reachable-set change: |solid+latent| base vs rl
    reach_b = sum(v for (x, _), v in c.items() if x in ("solid", "latent"))
    reach_r = sum(v for (_, y), v in c.items() if y in ("solid", "latent"))
    print(f"\n{label}  [{task}] (n={n}, K={K})")
    print(f"  monetized latent->solid : {l2s}/{lat} = {l2s/max(lat,1):.1%}")
    print(f"  solid lost (solid->!solid): {sol_lost}/{sol}")
    print(f"  contraction latent->unreach: {l2u}/{lat}")
    print(f"  reachable set: base {reach_b} -> rl {reach_r} ({reach_r-reach_b:+d})  "
          f"[expand>0 / contract<0]")
    # creation candidates: base unreach -> rl reachable
    cre = c.get(("unreach","solid"),0)+c.get(("unreach","latent"),0)
    print(f"  creation-candidate unreach->reachable: {cre}/{unr}")

print("="*70); print("CLEAN-PAIR FATE MATRICES (§5, same-format Qwen, t08)"); print("="*70)
fate("srbase", "srzero", "gsm8k", "SimpleRL: Qwen2.5-Math-7B base -> RL-Zero")
fate("srbase", "srzero", "dart",  "SimpleRL: Qwen2.5-Math-7B base -> RL-Zero")
fate("qwenlad_q32base", "dapo32", "gsm8k", "DAPO: Qwen2.5-32B base -> RL-Zero @32B")
fate("qwenlad_q32base", "dapo32", "dart",  "DAPO: Qwen2.5-32B base -> RL-Zero @32B")
fate("srbase", "r1d7", "gsm8k", "R1-Distill @7B: Qwen2.5-Math-7B base -> distill")
fate("srbase", "r1d7", "dart",  "R1-Distill @7B: Qwen2.5-Math-7B base -> distill")
fate("qwenlad_q32base", "r1d32", "gsm8k", "R1-Distill @32B: Qwen2.5-32B base -> distill")
fate("qwenlad_q32base", "r1d32", "dart",  "R1-Distill @32B: Qwen2.5-32B base -> distill")
