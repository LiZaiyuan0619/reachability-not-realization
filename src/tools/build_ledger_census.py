"""MANY-MODEL capability ledger (Figure-1, broad panel) -- the G/R census.

build_ledger.py demonstrates the FULL per-sample fate matrix (G,R,S_K,P_K,M_K,Z)
but only on ~1 model (Llama-3.2-3B) + a couple of scar cases -- reviewers said the
tiers were "shown on only ~1 model, too thin". THIS script extends the two columns
that DO exist across the whole gap census -- G (direct-generation = production) and
R (recognition = MC letter-loglik) -- to EVERY model in results/w11_gap_* so the
"capability tiers exist / production < recognition" claim is shown on MANY models.

For each (model, task) cell we read:
  G_strict  = <task>_baseline.json      summary.acc_micro    (deployed greedy gen)
  G_lenient = <task>_baseline.json      summary.acc_lenient  (format-robust gen)
  R         = <task>_baseline_mcll.json summary.acc_micro    (letter-loglik recog)
  R_opttext = <task>_baseline_mcll.json summary.acc_lenient  (option-text logprob)

The honest realization gap is gap_len = R - G_lenient (format-robust): using lenient
gen as the production floor auto-corrects the strict-grading verbosity artifact that
plagues small/verbose models. Per-cell verdicts REUSE harvest_gaps.classify_cell
(single source of truth) to drop the two false-positive classes:
  COLLAPSE  generation broke (gl<0.10 & m>0.30)          -> excluded artifact
  FORMAT    strict-only gap  (gap_str>0.10 & gap_len<0.05)-> excluded artifact
  REAL      format-robust gap (gap_len>0.05)              -> genuine tier separation
  zero      otherwise                                     -> flat / no separation

A REAL cell only counts as a GENUINE tier if recognition also clears chance by >=0.10
AND generation is not collapsed (G_lenient > 0.5*chance) -- this drops "gaps" where the
recognition is itself only chance-level (e.g. an 80%-pruned model destroyed to chance on
both channels) or the generation harness simply broke. Per-model verdict aggregates cells
of the SAME checkpoint (task-subset reruns like "<code>hard"/"<code>b" merged; quant/prune
variants kept separate as distinct weights):
  separates       >=1 GENUINE recognition>production cell (the positive result)
  separates_weak  REAL gap exists but only at chance-level recognition / collapsed gen
                  -> degenerate (e.g. heavy pruning), NOT a clean tier
  flat_headroom   0 REAL, but >=1 clean zero-gap cell WITH headroom (G_lenient<=0.80)
                  -> genuine evidence the tiers do NOT separate for this model
  flat_saturated  0 REAL, all clean cells near ceiling (G_lenient>0.80)
                  -> UNINFORMATIVE: no headroom for a gap to appear (common-MCQ saturation)
  artifact_only   no clean cell (only FORMAT/COLLAPSE) -> excluded, cannot read

Pure re-analysis of existing census JSONs -- NO GPU, read-only except outputs under
results/ledger/. Model names taken verbatim from the rjob launch scripts. Run:

    python tools/build_ledger_census.py            # from dr-llm/
    python tools/build_ledger_census.py --selftest # logic check, no files
"""
import argparse
import csv
import glob
import json
import os

from harvest_gaps import classify_cell  # single source of truth for thresholds

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RES, "ledger")

HEADROOM = 0.80   # G_lenient above this = near-ceiling: a zero gap is uninformative
REAL_GAP = 0.05   # matches harvest_gaps REAL threshold (format-robust)
R_MARGIN = 0.10   # recognition must clear chance by this to count as GENUINE knowledge
                  # (guards against "gaps" where recognition is itself only chance-level,
                  #  e.g. an 80%-pruned model whose recognition is destroyed to chance too)

# per-task random-chance accuracy = 1/#options (option counts verified from the mcll
# gold-letter sets: 2-way piqa/boolq/winogrande; 5-way csqa/aqua_rat/lsat_*; else 4-way)
CHANCE = {"piqa": 0.5, "boolq": 0.5, "winogrande": 0.5,
          "commonsense_qa": 0.2, "agieval_aqua_rat": 0.2, "agieval_lsat_ar": 0.2,
          "agieval_lsat_lr": 0.2, "agieval_lsat_rc": 0.2}
def chance(task):
    return CHANCE.get(task, 0.25)

# --------------------------------------------------------------------------- registry
# code (w11_gap_<code>) -> (checkpoint-identity, family). Checkpoint-identity MERGES
# task-subset reruns of the SAME weights ("hard"/"b"/"-7b" runs) but keeps quant/prune
# variants SEPARATE (different weights). Names verbatim from rjob/submit_w13*.sh etc.
CKPT = {  # code: (checkpoint display name, family)
    "70b": ("Llama-3.3-70B-Instruct", "Llama"),
    "e4bit": ("gemma-4-E4B-it", "Gemma"),
    "exaone4": ("EXAONE-4.0-32B", "EXAONE"),
    "g22b": ("gemma-2-2b-it", "Gemma"),
    "g29b": ("gemma-2-9b-it", "Gemma"),
    "g312": ("gemma-3-12b-it", "Gemma"),
    "g312qat": ("gemma-3-12b-it (QAT-int4)", "Gemma"),
    "g31b": ("gemma-3-1b-it", "Gemma"),
    "g31bb": ("gemma-3-1b-it", "Gemma"),
    "g327": ("gemma-3-27b-it", "Gemma"),
    "g34b": ("gemma-3-4b-it", "Gemma"),
    "g412": ("gemma-4-12B-it", "Gemma"),
    "g412hard": ("gemma-4-12B-it", "Gemma"),
    "g431": ("gemma-4-31B-it", "Gemma"),
    "g431hard": ("gemma-4-31B-it", "Gemma"),
    "g4qat": ("gemma-4-12B-it (QAT-int4)", "Gemma"),
    "g4qathard": ("gemma-4-12B-it (QAT-int4)", "Gemma"),
    "glm32": ("GLM-4-32B", "GLM"),
    "hy18": ("Hunyuan-1.8B-Instruct", "Hunyuan"),
    "hy4b": ("Hunyuan-4B-Instruct", "Hunyuan"),
    "hy7b": ("Hunyuan-7B-Instruct", "Hunyuan"),
    "l3170b": ("Llama-3.1-70B-Instruct", "Llama"),
    "l318bhard": ("Llama-3.1-8B-Instruct", "Llama"),
    "l32-1b": ("Llama-3.2-1B-Instruct", "Llama"),
    "l38b": ("Meta-Llama-3-8B-Instruct", "Llama"),
    "l3bhard": ("Llama-3.2-3B-Instruct", "Llama"),
    "llama2": ("Llama-2-7b-chat-hf", "Llama"),
    "misnemo": ("Mistral-Nemo-Instruct-2407", "Mistral"),
    "mist03": ("Mistral-7B-Instruct-v0.3", "Mistral"),
    "mixtral": ("Mixtral-8x7B-Instruct-v0.1", "Mistral"),
    "phi3mini": ("Phi-3-mini-4k-instruct", "Phi"),
    "phi4m": ("Phi-4-mini-instruct", "Phi"),
    "prq257s25w": ("Qwen2.5-7B-Instruct (wanda 25%)", "Qwen"),
    "prq257s50m": ("Qwen2.5-7B-Instruct (magnitude 50%)", "Qwen"),
    "prq257s50w": ("Qwen2.5-7B-Instruct (wanda 50%)", "Qwen"),
    "prq257s60w": ("Qwen2.5-7B-Instruct (wanda 60%)", "Qwen"),
    "prq257s70m": ("Qwen2.5-7B-Instruct (magnitude 70%)", "Qwen"),
    "prq257s70w": ("Qwen2.5-7B-Instruct (wanda 70%)", "Qwen"),
    "prq257s80w": ("Qwen2.5-7B-Instruct (wanda 80%)", "Qwen"),
    "q15moe": ("Qwen1.5-MoE-A2.7B-Chat", "Qwen"),
    "q25-7b": ("Qwen2.5-7B-Instruct", "Qwen"),
    "q257": ("Qwen2.5-7B-Instruct", "Qwen"),
    "q257hard": ("Qwen2.5-7B-Instruct", "Qwen"),
    "q2505": ("Qwen2.5-0.5B-Instruct", "Qwen"),
    "q2505b": ("Qwen2.5-0.5B-Instruct", "Qwen"),
    "q2505hard": ("Qwen2.5-0.5B-Instruct", "Qwen"),
    "q2514": ("Qwen2.5-14B-Instruct", "Qwen"),
    "q2514hard": ("Qwen2.5-14B-Instruct", "Qwen"),
    "q2515": ("Qwen2.5-1.5B-Instruct", "Qwen"),
    "q2515b": ("Qwen2.5-1.5B-Instruct", "Qwen"),
    "q2532": ("Qwen2.5-32B-Instruct", "Qwen"),
    "q2572": ("Qwen2.5-72B-Instruct", "Qwen"),
    "q25c32b": ("Qwen2.5-Coder-32B-Instruct", "Qwen"),
    "q314": ("Qwen3-14B", "Qwen"),
    "q330a2507": ("Qwen3-30B-A3B-Instruct-2507", "Qwen"),
    "q332": ("Qwen3-32B", "Qwen"),
    "q34": ("Qwen3-4B", "Qwen"),
    "q3527": ("Qwen3.5-27B", "Qwen"),
    "q352b": ("Qwen3.5-2B", "Qwen"),
    "q359": ("Qwen3.5-9B", "Qwen"),
    "q3627": ("Qwen3.6-27B", "Qwen"),
    "q3635": ("Qwen3.6-35B-A3B", "Qwen"),
    "q38": ("Qwen3-8B", "Qwen"),
    "q3bhard": ("Qwen2.5-3B-Instruct", "Qwen"),
    "q3n4": ("Qwen3-4B-Instruct-2507", "Qwen"),
    "qwen27b": ("Qwen2-7B-Instruct", "Qwen"),
    # sandbagging audit (rjob/sandbag_probe_run.sh): password-locked LoRA merge
    # of FelixHofstaetter/mistral-7b-sandbagging-new vs its untuned v0.1 base;
    # both evaluated with --base_prompt (non-instruct)
    "sandbagbase": ("Mistral-7B-v0.1 (base ctrl)", "Mistral"),
    "sandbaglocked": ("Mistral-7B-v0.1 (password-locked)", "Mistral"),
    "seed36": ("Seed-OSS-36B-Instruct", "Seed"),
    "smol": ("SmolLM3-3B", "SmolLM"),
    "tinyllama": ("TinyLlama-1.1B-Chat-v1.0", "TinyLlama"),
}
# checkpoints whose WEIGHTS are a compression variant of a released model (kept
# separate from the base, flagged so a clean-figure can drop them if desired)
COMPRESSION = {c for c, (n, f) in CKPT.items()
               if any(k in n for k in ("QAT", "wanda", "magnitude", "AWQ", "GPTQ", "FP8"))}


# --------------------------------------------------------------------------- load cells
def load_cells(root=RES):
    """One record per (code, task) w11_gap_* cell that has BOTH gen and mcll."""
    cells = []
    for d in sorted(glob.glob(os.path.join(root, "w11_gap_*"))):
        code = os.path.basename(d)[len("w11_gap_"):]
        ckpt, family = CKPT.get(code, (f"UNKNOWN[{code}]", "?"))
        for gf in sorted(glob.glob(os.path.join(d, "*_baseline.json"))):
            task = os.path.basename(gf)[:-len("_baseline.json")]
            mf = gf[:-len("_baseline.json")] + "_baseline_mcll.json"
            if not os.path.exists(mf):
                continue
            try:
                gs = json.load(open(gf))["summary"]
                ms = json.load(open(mf))["summary"]
            except Exception:
                continue
            G, Glen = gs.get("acc_micro"), gs.get("acc_lenient")
            R, Ropt = ms.get("acc_micro"), ms.get("acc_lenient")
            if G is None or R is None:
                continue
            if Glen is None:
                Glen = G
            cls, gap_len, gap_str = classify_cell(G, Glen, R)
            ch = chance(task)
            # GENUINE = harvest-REAL AND recognition clears chance by R_MARGIN AND
            # generation is at least half-functional vs chance (not a broken/refusing
            # harness). A conservative subset of REAL that drops chance-level / collapsed
            # "gaps"; harvest's cell_class is left untouched above.
            r_above_chance = round(R - ch, 4)
            gen_functional = Glen > 0.5 * ch
            genuine = (cls == "REAL" and r_above_chance >= R_MARGIN and gen_functional)
            cells.append({
                "code": code, "ckpt": ckpt, "family": family,
                "is_compression": code in COMPRESSION, "task": task,
                "n": gs.get("n"), "chance": ch,
                "G_strict": round(G, 4), "G_lenient": round(Glen, 4),
                "R_letterloglik": round(R, 4), "R_opttext": round(Ropt, 4) if Ropt is not None else None,
                "gap_len": round(gap_len, 4), "gap_str": round(gap_str, 4),
                "recog_only_margin": round(max(gap_len, 0.0), 4),
                "R_above_chance": r_above_chance,
                "cell_class": cls, "genuine_real": genuine,
                "has_headroom": Glen <= HEADROOM,
                "saturated": Glen > HEADROOM,
            })
    return cells


# --------------------------------------------------------------------------- per-model
def rollup_models(cells):
    """Aggregate cells by checkpoint identity -> one row per model + verdict."""
    from collections import defaultdict
    by = defaultdict(list)
    for c in cells:
        by[c["ckpt"]].append(c)
    rows = []
    for ckpt, cs in by.items():
        # dedup duplicate tasks (task-subset reruns of same weights): keep max-n cell
        best_by_task = {}
        for c in cs:
            t = c["task"]
            if t not in best_by_task or (c["n"] or 0) > (best_by_task[t]["n"] or 0):
                best_by_task[t] = c
        ucells = list(best_by_task.values())
        clean = [c for c in ucells if c["cell_class"] in ("REAL", "zero")]
        real = [c for c in ucells if c["cell_class"] == "REAL"]
        genuine = [c for c in ucells if c["genuine_real"]]
        n_class = {k: sum(1 for c in ucells if c["cell_class"] == k)
                   for k in ("REAL", "zero", "FORMAT", "COLLAPSE")}
        # verdict (5-way): genuine separation > weak/degenerate REAL > flat(headroom) >
        # flat(saturated) > artifact-only
        if genuine:
            verdict = "separates"                 # >=1 genuine recognition>production tier
        elif real:
            verdict = "separates_weak"            # REAL only at chance-level R / collapsed gen
        elif clean:
            verdict = "flat_headroom" if any(c["has_headroom"] for c in clean) else "flat_saturated"
        else:
            verdict = "artifact_only"
        # display cell: the genuine separation if any, else max-gap clean, else any
        pool = genuine or clean or ucells
        best = max(pool, key=lambda c: c["gap_len"])
        cmean = clean if clean else ucells
        rows.append({
            "ckpt": ckpt, "family": cs[0]["family"],
            "is_compression": cs[0]["is_compression"],
            "codes": ",".join(sorted({c["code"] for c in cs})),
            "n_tasks": len(ucells), "n_clean": len(clean),
            "n_REAL": n_class["REAL"], "n_genuine": len(genuine), "n_zero": n_class["zero"],
            "n_FORMAT": n_class["FORMAT"], "n_COLLAPSE": n_class["COLLAPSE"],
            "verdict": verdict,
            "best_gap_task": best["task"], "best_gap_len": best["gap_len"],
            "best_G_strict": best["G_strict"], "best_G_lenient": best["G_lenient"],
            "best_R": best["R_letterloglik"], "best_R_above_chance": best["R_above_chance"],
            "mean_G_strict": round(sum(c["G_strict"] for c in cmean) / len(cmean), 4),
            "mean_G_lenient": round(sum(c["G_lenient"] for c in cmean) / len(cmean), 4),
            "mean_R": round(sum(c["R_letterloglik"] for c in cmean) / len(cmean), 4),
            "max_recog_only_margin": round(max(c["recog_only_margin"] for c in ucells), 4),
        })
    rows.sort(key=lambda r: (-r["best_gap_len"], r["ckpt"]))
    return rows


# --------------------------------------------------------------------------- write
def write_outputs(cells, models):
    os.makedirs(OUT, exist_ok=True)
    cell_keys = ["code", "ckpt", "family", "is_compression", "task", "n", "chance",
                 "G_strict", "G_lenient", "R_letterloglik", "R_opttext",
                 "gap_len", "gap_str", "recog_only_margin", "R_above_chance",
                 "cell_class", "genuine_real", "has_headroom", "saturated"]
    with open(os.path.join(OUT, "census_gr_cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cell_keys, extrasaction="ignore")
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c["ckpt"], c["task"])):
            w.writerow(c)
    model_keys = ["ckpt", "family", "is_compression", "codes", "n_tasks", "n_clean",
                  "n_REAL", "n_genuine", "n_zero", "n_FORMAT", "n_COLLAPSE", "verdict",
                  "best_gap_task", "best_gap_len", "best_G_strict", "best_G_lenient",
                  "best_R", "best_R_above_chance", "mean_G_strict", "mean_G_lenient",
                  "mean_R", "max_recog_only_margin"]
    with open(os.path.join(OUT, "census_gr_models.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=model_keys, extrasaction="ignore")
        w.writeheader()
        for r in models:
            w.writerow(r)
    return cell_keys, model_keys


def summarize(cells, models):
    from collections import Counter
    verdict_ct = Counter(m["verdict"] for m in models)
    base_models = [m for m in models if not m["is_compression"]]
    comp_models = [m for m in models if m["is_compression"]]
    base_verdict = Counter(m["verdict"] for m in base_models)
    return {
        "scope": "results/w11_gap_* (the gap census); columns G (gen=production) & R "
                 "(mcll letter-loglik=recognition), which exist across many models",
        "n_cells_model_x_task": len(cells),
        "n_distinct_checkpoints": len(models),
        "n_base_checkpoints": len(base_models),
        "n_compression_variants": len(comp_models),
        "n_distinct_tasks": len(set(c["task"] for c in cells)),
        "tasks": sorted(set(c["task"] for c in cells)),
        "thresholds": {"REAL_gap_len": REAL_GAP, "headroom_G_lenient": HEADROOM,
                       "genuine_R_margin_over_chance": R_MARGIN,
                       "genuine_gen_floor": "G_lenient > 0.5*chance",
                       "gap_metric": "gap_len = R(letter-loglik) - G_lenient (format-robust)"},
        "verdict_counts_all_checkpoints": dict(verdict_ct),
        "verdict_counts_base_only": dict(base_verdict),
        "cell_class_counts_harvest": dict(Counter(c["cell_class"] for c in cells)),
        "genuine_real_cell_count": sum(1 for c in cells if c["genuine_real"]),
        "reader_note": (
            "'separates' = >=1 GENUINE recognition>production tier (harvest-REAL gap>0.05 AND "
            "recognition clears chance by >=0.10 AND generation not collapsed); 'separates_weak' "
            "= REAL gap exists but only at chance-level recognition / collapsed generation (e.g. "
            "heavily-pruned models destroyed to chance -- not a clean tier); 'flat_headroom' = "
            "clean zero-gap WITH headroom (genuine no-tier evidence: recognition~=production even "
            "with room); 'flat_saturated' = zero-gap only at ceiling (common-MCQ saturation, "
            "UNINFORMATIVE -- no headroom for a gap); 'artifact_only' = no readable clean cell. "
            "FORMAT/COLLAPSE cells excluded per harvest_gaps (grader / generation-collapse "
            "false positives). cell_class is harvest's; genuine_real is the stricter subset."),
    }


# --------------------------------------------------------------------------- selftest
def _selftest():
    # classify_cell parity with harvest thresholds
    assert classify_cell(0.05, 0.05, 0.40)[0] == "COLLAPSE"   # gen broke
    assert classify_cell(0.20, 0.70, 0.73)[0] == "FORMAT"     # strict-only gap closed by lenient
    assert classify_cell(0.30, 0.30, 0.45)[0] == "REAL"       # +0.15 format-robust gap
    assert classify_cell(0.90, 0.90, 0.90)[0] == "zero"       # flat
    # genuine-real guard: REAL + R clears chance by margin + gen functional
    assert (classify_cell(0.30, 0.30, 0.45)[0] == "REAL")            # gemma-like, R>>chance
    assert not (0.24 - chance("arc_challenge") >= R_MARGIN)          # 80%-pruned: R@chance -> not genuine
    assert (0.56 - chance("piqa") < R_MARGIN)                        # 0.5B/piqa: R barely>chance -> not genuine
    # verdict logic
    def mk(cls, gl, gap, genuine=False, task="t"):
        return {"code": "x", "ckpt": "M", "family": "F", "is_compression": False,
                "task": task, "n": 300, "chance": 0.25, "cell_class": cls, "gap_len": gap,
                "genuine_real": genuine, "G_strict": gl, "G_lenient": gl,
                "R_letterloglik": gl + gap, "R_above_chance": round(gl + gap - 0.25, 4),
                "recog_only_margin": max(gap, 0), "has_headroom": gl <= HEADROOM,
                "saturated": gl > HEADROOM}
    v = lambda cs: rollup_models(cs)[0]["verdict"]
    assert v([mk("REAL", 0.30, 0.20, genuine=True), mk("zero", 0.90, 0.0, task="t2")]) == "separates"
    assert v([mk("REAL", 0.00, 0.24, genuine=False)]) == "separates_weak"  # REAL @chance
    assert v([mk("zero", 0.50, 0.0)]) == "flat_headroom"      # headroom, no gap
    assert v([mk("zero", 0.92, 0.0)]) == "flat_saturated"     # ceiling, no gap
    assert v([mk("FORMAT", 0.05, 0.0), mk("COLLAPSE", 0.05, 0.4, task="t2")]) == "artifact_only"
    print("SELFTEST OK (classify_cell parity + genuine guard + verdict taxonomy)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    cells = load_cells()
    unknown = sorted({c["code"] for c in cells if c["ckpt"].startswith("UNKNOWN")})
    models = rollup_models(cells)
    write_outputs(cells, models)
    summ = summarize(cells, models)
    if unknown:
        summ["UNMAPPED_codes"] = unknown
    with open(os.path.join(OUT, "census_gr_summary.json"), "w") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)

    # ---- console report
    print("=" * 100)
    print("MANY-MODEL G/R CAPABILITY LEDGER  (results/w11_gap_* census)")
    print("=" * 100)
    print(f"cells (model x task) with both G & R : {summ['n_cells_model_x_task']}")
    print(f"distinct checkpoints                 : {summ['n_distinct_checkpoints']} "
          f"({summ['n_base_checkpoints']} base + {summ['n_compression_variants']} compression variants)")
    print(f"distinct tasks                       : {summ['n_distinct_tasks']}  {summ['tasks']}")
    if unknown:
        print(f"!! UNMAPPED codes                    : {unknown}")
    print(f"\nper-checkpoint VERDICTS (all) : {summ['verdict_counts_all_checkpoints']}")
    print(f"per-checkpoint VERDICTS (base): {summ['verdict_counts_base_only']}")
    print(f"per-cell classes (harvest)   : {summ['cell_class_counts_harvest']}  "
          f"| genuine_real cells: {summ['genuine_real_cell_count']}")
    print("\n" + "-" * 104)
    print(f"{'checkpoint':<38}{'fam':<9}{'verdict':<16}{'bestTask':<15}"
          f"{'Gstr':>6}{'Glen':>6}{'R':>6}{'gapL':>7}{'R-chance':>9}")
    print("-" * 104)
    for r in models:
        tag = "*" if r["is_compression"] else " "
        print(f"{tag}{r['ckpt']:<37}{r['family']:<9}{r['verdict']:<16}"
              f"{r['best_gap_task'].replace('agieval_',''):<15}"
              f"{r['best_G_strict']:>6.2f}{r['best_G_lenient']:>6.2f}{r['best_R']:>6.2f}"
              f"{r['best_gap_len']:>+7.2f}{r['best_R_above_chance']:>+9.2f}")
    print("-" * 104)
    print("(* = compression variant; row = model's display cell = its genuine-max-gap task if "
          "any, else max-gap clean task)")
    print(f"\nwrote: {OUT}/census_gr_cells.csv  census_gr_models.csv  census_gr_summary.json")


if __name__ == "__main__":
    main()
