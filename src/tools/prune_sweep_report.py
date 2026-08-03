#!/usr/bin/env python3
"""Prune-sweep gap-emergence report. Reads the per-sparsity eval dirs produced by the
pruning bridge (rjob/prune_sweep_run.sh writes results/w11_gap_pr<model>s<NN><m>/ with
<task>_baseline.json + <task>_baseline_mcll.json, same schema as the census) and prints
the realization gap  gap_len = mcll(R) - gen_lenient(G)  as a function of sparsity, per
task. Flags the FIRST sparsity at which a gap EMERGES (gap_len > --thr, default 0.05) --
the headline result: a healthy zero-gap model acquires a recognition->production scar as
the sparsity knob is turned up. Reuses the harvest_gaps.py gap metric verbatim.

Usage
  python tools/prune_sweep_report.py --root results --tag prq257            # wanda arm
  python tools/prune_sweep_report.py --root results --tag prq257 --method m # magnitude arm
  python tools/prune_sweep_report.py --selftest                            # no I/O check
"""
import argparse
import collections
import glob
import json
import os
import re


def summ(f):
    try:
        return json.load(open(f))["summary"]
    except Exception:
        return None


def gap_of(gen_json):
    """(gen_strict, gen_lenient, mcll, gap_len) from a *_baseline.json path, or None."""
    gs = summ(gen_json)
    ms = summ(gen_json.replace("_baseline.json", "_baseline_mcll.json"))
    if not gs or not ms:
        return None
    gstrict, mcll = gs.get("acc_micro"), ms.get("acc_micro")
    glen = gs.get("acc_lenient")
    glen = gstrict if glen is None else glen
    if gstrict is None or mcll is None:
        return None
    return gstrict, glen, mcll, mcll - glen


def parse_sparsity(dirname):
    """w11_gap_prq257s50w -> 50 ; returns int percent or None."""
    m = re.search(r"s(\d+)[a-z]?$", dirname)
    return int(m.group(1)) if m else None


def collect(root, tag, method):
    """-> {sparsity_pct: {task: (gs, gl, m, gap)}}. method suffix: 'w' wanda / 'm' magnitude."""
    rows = collections.defaultdict(dict)
    pat = os.path.join(root, f"w11_gap_{tag}s*{method}")
    for d in sorted(glob.glob(pat)):
        sp = parse_sparsity(os.path.basename(d))
        if sp is None:
            continue
        for gf in glob.glob(os.path.join(d, "*_baseline.json")):
            t = os.path.basename(gf).replace("_baseline.json", "")
            g = gap_of(gf)
            if g:
                rows[sp][t] = g
    return rows


def render(rows, base_row=None, thr=0.05):
    """Pretty sparsity x task gap table + emergence flags. base_row: {task:(gs,gl,m,gap)} at s=0."""
    lines = []
    tasks = sorted({t for sp in rows for t in rows[sp]}
                   | (set(base_row) if base_row else set()))
    header = "sparsity%".ljust(10) + "".join(f"{t[:16]:>18}" for t in tasks)
    lines.append(header)
    grid = dict(rows)
    if base_row is not None:
        grid = {0: base_row, **{k: v for k, v in rows.items() if k != 0}}
    for sp in sorted(grid):
        cells = []
        for t in tasks:
            g = grid[sp].get(t)
            cells.append(f"{g[3]:+.3f}({g[1]:.2f}->{g[2]:.2f})".rjust(18) if g else "-".rjust(18))
        lines.append(f"{sp:<10}" + "".join(cells))
    # emergence: first sparsity with gap_len > thr per task
    lines.append("")
    for t in tasks:
        emerged = [sp for sp in sorted(grid) if grid[sp].get(t) and grid[sp][t][3] > thr]
        first = emerged[0] if emerged else None
        peak = max((grid[sp][t][3] for sp in grid if grid[sp].get(t)), default=None)
        lines.append(f"  {t:22} gap>{thr}: emerges at sparsity="
                     f"{'%d%%' % first if first is not None else 'NONE'}"
                     f"  peak_gap={('%+.3f' % peak) if peak is not None else 'NA'}")
    return "\n".join(lines)


def selftest():
    ok = []

    def chk(n, c):
        ok.append(bool(c))
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")

    chk("parse_sparsity w11_gap_prq257s50w -> 50", parse_sparsity("w11_gap_prq257s50w") == 50)
    chk("parse_sparsity w11_gap_prq257s0w -> 0", parse_sparsity("w11_gap_prq257s0w") == 0)
    chk("parse_sparsity no-match -> None", parse_sparsity("w11_gap_q257") is None)

    import tempfile
    with tempfile.TemporaryDirectory() as root:
        def put(dirn, task, gen_micro, gen_len, mcll_micro):
            d = os.path.join(root, dirn)
            os.makedirs(d, exist_ok=True)
            json.dump({"summary": {"acc_micro": gen_micro, "acc_lenient": gen_len}},
                      open(os.path.join(d, f"{task}_baseline.json"), "w"))
            json.dump({"summary": {"acc_micro": mcll_micro}},
                      open(os.path.join(d, f"{task}_baseline_mcll.json"), "w"))
        # healthy at 25%, scar emerges at 60% (gap 0.31), grows at 70%
        put("w11_gap_prq257s25w", "piqa", 0.85, 0.85, 0.86)
        put("w11_gap_prq257s60w", "piqa", 0.40, 0.42, 0.73)
        put("w11_gap_prq257s70w", "piqa", 0.20, 0.22, 0.66)
        rows = collect(root, "prq257", "w")
        chk("collect finds 3 sparsity levels", set(rows) == {25, 60, 70})
        chk("gap at 25% ~ 0.01 (healthy)", abs(rows[25]["piqa"][3] - 0.01) < 1e-6)
        chk("gap at 60% ~ 0.31 (scar)", abs(rows[60]["piqa"][3] - 0.31) < 1e-6)
        txt = render(rows, thr=0.05)
        chk("render flags emergence at 60%", "emerges at sparsity=60%" in txt)
        chk("render reports peak gap +0.310", "+0.310" in txt)
        # magnitude arm isolation: 'm' method must not pick up 'w' dirs
        chk("method filter: magnitude arm empty here", collect(root, "prq257", "m") == {})

    print(f"\nSELFTEST: {sum(ok)}/{len(ok)} checks passed -> "
          f"{'ALL PASS' if all(ok) else 'FAILURES PRESENT'}")
    return all(ok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--tag", default="prq257", help="model tag between 'w11_gap_' and 's<NN>'")
    ap.add_argument("--method", default="w", choices=["w", "m"], help="w=wanda m=magnitude")
    ap.add_argument("--base_dir", default=None,
                    help="optional s=0 census dir (e.g. w11_gap_q257) for the healthy row")
    ap.add_argument("--thr", type=float, default=0.05)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        import sys
        sys.exit(0 if selftest() else 1)
    rows = collect(args.root, args.tag, args.method)
    base_row = None
    if args.base_dir:
        bd = os.path.join(args.root, args.base_dir)
        base_row = {}
        for gf in glob.glob(os.path.join(bd, "*_baseline.json")):
            t = os.path.basename(gf).replace("_baseline.json", "")
            g = gap_of(gf)
            if g:
                base_row[t] = g
    if not rows:
        print(f"no pruned sweep dirs matched results/w11_gap_{args.tag}s*{args.method}")
    else:
        print(render(rows, base_row, args.thr))
