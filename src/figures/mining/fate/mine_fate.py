#!/usr/bin/env python3
"""Mine Fig 5 and Fig 6 inputs from the paired reroll fate files.

Every item is read on both sides of a training step at the same budget, format
and temperature, and joined on its question id.  An item's state is:
  solid   the greedy answer is correct, which is what a user receives
  latent  greedy is wrong but at least one of K samples is correct
  absent  no sample in the budget is correct

fate.csv       per-pair counts for every state transition
propensity.csv for base-latent items, the share that become solid, binned by how
               often the base model already hit them
"""
import json, os, csv
from collections import Counter, defaultdict

LENS = "workspace/myPaper/drllm/docs/data/snapshot/lens"
OUT  = os.path.dirname(os.path.abspath(__file__))

# (label, base file stem, trained file stem, how the training step is described)
PAIRS = [
    ("SimpleRL", "srbase",  "srzero",  "from-base RLVR, mild"),
    ("DAPO",     "q32base", "dapo32",  "from-base RLVR, aggressive"),
    ("OR1",      "srbase",  "or1m7",   "RLVR on a distilled model"),
]
TASKS = ["dart", "gsm8k"]
BINS  = [(0.0, 0.02, "0-2%"), (0.02, 0.05, "2-5%"), (0.05, 0.15, "5-15%"),
         (0.15, 0.35, "15-35%"), (0.35, 1.01, ">35%")]


def load(stem, task):
    p = os.path.join(LENS, f"reroll_fateHi_{stem}_{task}_t08.json")
    return json.load(open(p)) if os.path.exists(p) else None


def state(rec):
    if rec["single_hit"]:
        return "solid"
    return "latent" if rec["oracle_hit"] else "absent"


frows, prows = [], []
print(f"{'pair':10s} {'task':6s} {'n':>4s} {'K':>4s} "
      f"{'base depl':>9s} {'base orac':>9s} {'trn depl':>8s} {'trn orac':>8s} "
      f"{'d_rch':>6s}")
for label, bstem, tstem, kind in PAIRS:
    for task in TASKS:
        jb, jt = load(bstem, task), load(tstem, task)
        if jb is None or jt is None:
            print(f"  MISS {label}/{task}"); continue
        # Join on question id, keep only items present on both sides.
        B = {r["qid"]: r for r in jb["records"]}
        T = {r["qid"]: r for r in jt["records"]}
        common = [q for q in B if q in T]
        if not common:
            print(f"  NO OVERLAP {label}/{task}"); continue
        Kb, Kt = jb["config"]["K"], jt["config"]["K"]
        flow = Counter()
        for q in common:
            flow[(state(B[q]), state(T[q]))] += 1
        n = len(common)
        b_dep = sum(1 for q in common if B[q]["single_hit"]) / n
        t_dep = sum(1 for q in common if T[q]["single_hit"]) / n
        b_orc = sum(1 for q in common if B[q]["oracle_hit"]) / n
        t_orc = sum(1 for q in common if T[q]["oracle_hit"]) / n
        d_rch = sum(1 for q in common if T[q]["oracle_hit"]) - \
                sum(1 for q in common if B[q]["oracle_hit"])
        print(f"{label:10s} {task:6s} {n:4d} {Kb:4d} {b_dep:9.3f} {b_orc:9.3f} "
              f"{t_dep:8.3f} {t_orc:8.3f} {d_rch:+6d}")
        for (s0, s1), c in sorted(flow.items()):
            frows.append(dict(pair=label, kind=kind, task=task, n=n,
                              K_base=Kb, K_trained=Kt, base_state=s0,
                              trained_state=s1, count=c,
                              share=round(c / n, 4),
                              base_deployed=round(b_dep, 4),
                              trained_deployed=round(t_dep, 4),
                              base_oracle=round(b_orc, 4),
                              trained_oracle=round(t_orc, 4),
                              delta_reachable=d_rch))

        # Propensity: among items the base could reach but not produce, which
        # ones does training turn into produced answers?
        buckets = defaultdict(lambda: [0, 0])
        for q in common:
            if state(B[q]) != "latent":
                continue
            hit = B[q]["n_sample_hits"] / Kb
            for lo, hi, name in BINS:
                if lo <= hit < hi:
                    buckets[name][1] += 1
                    if T[q]["single_hit"]:
                        buckets[name][0] += 1
                    break
        for lo, hi, name in BINS:
            got, tot = buckets[name]
            if tot:
                prows.append(dict(pair=label, task=task, bucket=name,
                                  lo=lo, hi=hi, monetized=got, total=tot,
                                  rate=round(got / tot, 4)))

for name, rows, fields in [
        ("fate.csv", frows, ["pair", "kind", "task", "n", "K_base", "K_trained",
                             "base_state", "trained_state", "count", "share",
                             "base_deployed", "trained_deployed", "base_oracle",
                             "trained_oracle", "delta_reachable"]),
        ("propensity.csv", prows, ["pair", "task", "bucket", "lo", "hi",
                                   "monetized", "total", "rate"])]:
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"\n{name}: {len(rows)} rows")

print("\n=== where the deployed gain comes from ===")
for label, _, _, _ in PAIRS:
    for task in TASKS:
        rs = [r for r in frows if r["pair"] == label and r["task"] == task]
        if not rs:
            continue
        n = rs[0]["n"]
        gained = sum(r["count"] for r in rs if r["trained_state"] == "solid"
                     and r["base_state"] != "solid")
        from_latent = sum(r["count"] for r in rs if r["base_state"] == "latent"
                          and r["trained_state"] == "solid")
        from_absent = sum(r["count"] for r in rs if r["base_state"] == "absent"
                          and r["trained_state"] == "solid")
        lost = sum(r["count"] for r in rs if r["base_state"] == "solid"
                   and r["trained_state"] != "solid")
        share = f"{from_latent / gained * 100:.0f}%" if gained else "n/a"
        print(f"{label:10s} {task:6s} new solid={gained:3d} "
              f"(from latent {from_latent:3d} = {share:>4s}, "
              f"from absent {from_absent:2d}), lost={lost:3d}, "
              f"net={gained - lost:+3d} of {n}")

print("\n=== propensity by base hit rate ===")
for label, _, _, _ in PAIRS:
    for task in TASKS:
        rs = [r for r in prows if r["pair"] == label and r["task"] == task]
        if not rs:
            continue
        cells = "  ".join(f"{r['bucket']}: {r['monetized']}/{r['total']}"
                          f"={r['rate']*100:.0f}%" for r in rs)
        print(f"{label:10s} {task:6s} {cells}")
