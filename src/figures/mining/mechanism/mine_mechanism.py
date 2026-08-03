#!/usr/bin/env python3
"""Mine Fig 4 inputs from the primary battery JSONs and the checkpoint census.

necessity.csv    per-case repair rate for each intervention (which part is the locus)
selectivity.csv  repair rate on sick items vs break rate on healthy items
minimality.csv   top-k vs random-k neuron ablation curves (is the write local)
census.csv       generation vs recognition for every clean checkpoint cell
"""
import json, os, csv

ROOT = "workspace/myPaper/drllm/docs/data/snapshot"
LENS = os.path.join(ROOT, "lens")
LEDG = os.path.join(ROOT, "ledger")
OUT  = os.path.dirname(os.path.abspath(__file__))

# The six cases of the locus table, with the file that carries each one.
# battery_hin_* are the full-n runs; q2505 has no hin run so its base file is used.
CASES = [
    ("prl318", "battery_hin_prl318.json",  "Llama-3.1-8B pruned, L20"),
    ("g431",   "battery_hin_g431.json",    "gemma-4-31B, L39"),
    ("g4qat",  "battery_hin_g4qat.json",   "gemma-4-12B QAT, L29"),
    ("g412",   "battery_hin_g412.json",    "gemma-4-12B, L29"),
    ("q2505",  "battery_q2505.json",       "Qwen2.5-0.5B, L10"),
    ("l3b",    "battery_hin_l3b.json",     "Llama-3.2-3B, L20"),
]

INTERV = [("zero_mlp", "silence MLP"), ("zero_attn", "silence attention"),
          ("zero_layer", "silence whole layer"), ("project_v", "remove direction"),
          ("top32", "silence top-32 neurons")]


def load(fn):
    p = os.path.join(LENS, fn)
    return json.load(open(p)) if os.path.exists(p) else None


# ══════════════ necessity + selectivity + minimality ════════════════════════
nec, sel, mini = [], [], []
for code, fn, label in CASES:
    j = load(fn)
    if j is None:
        print("  MISS", fn); continue
    cfg, pA = j["config"], j["panelA_necessity"]
    n_sick = cfg["n_R1G0_sick"]
    for key, name in INTERV:
        d = pA.get(key)
        if d is None:
            continue
        nec.append(dict(case=code, label=label, n_sick=n_sick,
                        intervention=name, key=key,
                        repair=round(d["correct"], 4), n_used=d.get("n", "")))

    buckets = j["panelC_selectivity"]["buckets"]
    sick, healthy = buckets.get("R1G0", {}), buckets.get("R1G1", {})
    zs, zh = sick.get("zero_mlp", {}), healthy.get("zero_mlp", {})
    sel.append(dict(case=code, label=label, n_sick=n_sick,
                    repair_sick=zs.get("wrong_to_right_rate"),
                    n_sick_used=zs.get("n"),
                    break_healthy=zh.get("right_to_wrong_rate"),
                    n_healthy=zh.get("n"),
                    preserve_healthy=zh.get("preserve_rate")))

    pB = j["panelB_minimality"]
    for k in pB["ks"]:
        ks = str(k)
        mini.append(dict(case=code, label=label, k=k,
                         topk=pB["topk_correct"].get(ks),
                         randk=pB["randk_correct_mean"].get(ks),
                         randk_sd=pB["randk_correct_sd"].get(ks),
                         full_zero=pB["full_zero_correct"],
                         frac_mass_top32=pB.get("frac_mass_top32")))

for name, rows, fields in [
        ("necessity.csv", nec, ["case", "label", "n_sick", "intervention",
                                "key", "repair", "n_used"]),
        ("selectivity.csv", sel, ["case", "label", "n_sick", "repair_sick",
                                  "n_sick_used", "break_healthy", "n_healthy",
                                  "preserve_healthy"]),
        ("minimality.csv", mini, ["case", "label", "k", "topk", "randk",
                                  "randk_sd", "full_zero", "frac_mass_top32"])]:
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"{name}: {len(rows)} rows")

print("\n=== locus table cross-check (against paper Table 3) ===")
print(f"{'case':8s} {'n':>4s} {'zMLP':>6s} {'zAttn':>6s} {'broken':>7s}")
for s in sel:
    z = [r for r in nec if r["case"] == s["case"]]
    m = next((r["repair"] for r in z if r["key"] == "zero_mlp"), None)
    a = next((r["repair"] for r in z if r["key"] == "zero_attn"), None)
    print(f"{s['case']:8s} {s['n_sick']:4d} {m:6.3f} {a:6.3f} "
          f"{s['break_healthy']:7.4f}")

# ══════════════ census: where the failure appears across checkpoints ════════
cen = []
with open(os.path.join(LEDG, "census_gr_cells.csv")) as f:
    for r in csv.DictReader(f):
        try:
            g, rr = float(r["G_strict"]), float(r["R_letterloglik"])
        except (ValueError, KeyError):
            continue
        cen.append(dict(ckpt=r["ckpt"], family=r["family"], task=r["task"],
                        n=r["n"], chance=r["chance"],
                        is_compression=r["is_compression"],
                        G=g, R=rr, gap=r.get("gap_len", ""),
                        cell_class=r.get("cell_class", ""),
                        genuine=r.get("genuine_real", "")))
with open(os.path.join(OUT, "census.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ckpt", "family", "task", "n", "chance",
                                      "is_compression", "G", "R", "gap",
                                      "cell_class", "genuine"])
    w.writeheader(); w.writerows(cen)
print(f"\ncensus.csv: {len(cen)} cells, "
      f"{len(set(c['ckpt'] for c in cen))} checkpoints, "
      f"{len(set(c['family'] for c in cen))} families")
real = [c for c in cen if c["genuine"] == "True"]
print(f"  cells with a genuine recognized-not-produced gap: {len(real)}")
