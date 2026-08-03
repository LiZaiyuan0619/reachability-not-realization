#!/usr/bin/env python3
"""MECHANISM-COMPLETENESS BATTERY: run the SAME causal protocol across every scar-positive
case and emit three reviewer-facing panels per case. Reuses mlp_patch/mlp_dissect pass1
(LE x SR bucketing at the answer-decision position; v = mean_out[LE+/SR-] - mean_out[LE+/SR+];
channel score[i] = mean_sick_int[i] * (down_proj[:,i] . v_hat)).

Ledger mapping (R=recognition via letter logprob = LE; G=generation via greedy short-gen = SR):
    R1G0 = LE+/SR-  (recognition-correct, direct-generation-wrong = the SICK bucket)
    R0G0 = LE-/SR-  (neither)
    R1G1 = LE+/SR+  (both = the HEALTHY bucket)
    R0G1 = LE-/SR+  (tracked, not a headline bucket)

PANEL A - Necessity + MLP-vs-attn (reviewer attack 1: "why MLP not attention?").
    On the SICK (R1G0) items, commit/correct under 6 interventions at the target layer:
      baseline | zero_layer | zero_attn | zero_mlp | project_v | top32
    Whole-layer zero == zeroing BOTH sublayer outputs (pre-norm residual block y=x+attn+mlp
    with both forced to 0 gives y=x, i.e. an identity/skipped block). Expected clean pattern:
    zero_mlp ~ zero_layer >> zero_attn (the MLP write is the locus). `mlp_vs_attn_verdict`
    turns the four correct-rates into an advisory clean/partial/muddy label + the raw deltas.

PANEL B - Minimality. top-k channel-ablation CURVE k=1,2,4,8,16,32,64,128 on the sick items
    vs a random-k control (>=20 seeds/k -> mean+/-sd), against the whole-MLP-zero reference.
    Reports the saturation-k (smallest k reaching frac*full_zero repair) and the per-k margin
    of top-k over random-k in sd units. (mlp_dissect did single k; this is the full curve.)

PANEL C - Selectivity by ledger bucket. Apply the intervention (zero_mlp headline + project_v)
    to R1G0 / R0G0 / R1G1 and score per-sample before->after: should REPAIR R1G0 (wrong->right),
    PRESERVE R1G1 (low right->wrong), and NOT fabricate correctness in R0G0 (wrong->right there
    is spurious; compared against 1/n_choices chance, which is 0.5 for the 2-choice piqa cases).

Usage:  python mlp_battery.py --model_path P --tokenizer_path P --layer L --task T \
            --limit 250 [--mm_model] --out X.json
        python mlp_battery.py --selftest        # torch-free CPU check of the pure cores (numpy only)
"""
import argparse, json, re, time
import numpy as np


# ==================== pure, torch-free, unit-testable cores ====================

def bucket_key(le, sr):
    """Map (LE bool, SR bool) -> ledger bucket string. LE=recognition(R), SR=generation(G)."""
    return f"R{int(bool(le))}G{int(bool(sr))}"


SICK, NEITHER, HEALTHY = "R1G0", "R0G0", "R1G1"


def transition_rates(before, after):
    """Per-sample before/after correctness (0/1 lists) -> transition rates. Bucket-agnostic;
    Panel C relabels wrong_to_right as 'repair' (R1G0), 'fabrication' (R0G0) and right_to_wrong
    as 'harm' (R1G1). Denominators are the at-risk sub-populations, not the whole bucket."""
    b = np.asarray(before, dtype=int); a = np.asarray(after, dtype=int)
    n = len(b)
    n_wrong = int((b == 0).sum()); n_right = int((b == 1).sum())
    w2r = int(((b == 0) & (a == 1)).sum())      # wrong -> right (repair / fabrication)
    r2w = int(((b == 1) & (a == 0)).sum())      # right -> wrong (harm)
    return {
        "n": n,
        "base_correct": round(float(b.mean()), 4) if n else None,
        "int_correct": round(float(a.mean()), 4) if n else None,
        "n_wrong_base": n_wrong, "n_right_base": n_right,
        "wrong_to_right": w2r, "right_to_wrong": r2w,
        "wrong_to_right_rate": round(w2r / n_wrong, 4) if n_wrong else None,   # repair / fabrication
        "right_to_wrong_rate": round(r2w / n_right, 4) if n_right else None,   # harm
        "preserve_rate": round(1 - r2w / n_right, 4) if n_right else None,
    }


def randk_summary(per_seed):
    """{k: [correct-rate per seed]} -> {k: mean}, {k: sd(ddof=1)}, n_seeds. Control aggregation."""
    mean, sd = {}, {}
    n_seeds = 0
    for k, vals in per_seed.items():
        a = np.asarray(vals, dtype=np.float64)
        n_seeds = max(n_seeds, len(a))
        mean[k] = round(float(a.mean()), 4) if a.size else None
        sd[k] = round(float(a.std(ddof=1)), 4) if a.size > 1 else 0.0
    return mean, sd, n_seeds


def topk_minus_rand(ks, topk_correct, rand_mean, rand_sd):
    """Per-k margin of top-k over the random-k control, as raw diff and in sd units."""
    out = {}
    for k in ks:
        tk = topk_correct.get(k); rm = rand_mean.get(k); rs = rand_sd.get(k)
        if tk is None or rm is None:
            out[k] = {"diff": None, "sigma": None}; continue
        diff = tk - rm
        out[k] = {"diff": round(diff, 4),
                  "sigma": round(diff / (rs + 1e-9), 3) if rs is not None else None}
    return out


def curve_saturation(ks, topk_correct, full_zero, frac=0.95):
    """Smallest k whose top-k repair reaches frac * full-MLP-zero repair. None => the sparse
    circuit never saturates within max(k) (a 'distributed' / muddy verdict for Panel B)."""
    if full_zero is None or full_zero <= 0:
        return None
    thr = frac * full_zero
    for k in sorted(ks):
        v = topk_correct.get(k)
        if v is not None and v >= thr:
            return k
    return None


def mlp_vs_attn_verdict(base, zero_layer, zero_attn, zero_mlp, min_repair=0.20,
                        dom=1.5, layer_lo=0.6, layer_hi=1.5):
    """Attack-1 adjudicator. Correct-rates in; advisory label out. Repair := correct - base
    (sick baseline is ~0, so repair ~= post-intervention correct-rate).
      mlp_is_locus  := MLP repair is large, dominates attn repair (>= dom x), and explains
                       most of the whole-layer repair (mlp/layer in [layer_lo, layer_hi]).
      label clean   := mlp_is_locus and attn barely repairs.
            partial := MLP still leads attn but separation weak, or layer >> mlp (cross-talk).
            muddy   := attn repairs comparably / MLP repair small.
      attn_confound flagged whenever attn repair alone is non-trivial (>=0.15)."""
    r_layer = zero_layer - base
    r_attn = zero_attn - base
    r_mlp = zero_mlp - base
    ratio_mlp_attn = r_mlp / (abs(r_attn) + 1e-9)
    ratio_mlp_layer = r_mlp / (abs(r_layer) + 1e-9)
    attn_confound = r_attn >= 0.15
    mlp_is_locus = (r_mlp >= min_repair and ratio_mlp_attn >= dom
                    and layer_lo <= ratio_mlp_layer <= layer_hi)
    if mlp_is_locus and not attn_confound and r_mlp >= 0.30:
        label = "clean"
    elif r_mlp >= 0.15 and r_mlp > r_attn:
        label = "partial"
    else:
        label = "muddy"
    return {
        "repair_baseline_correct": round(base, 4),
        "repair_zero_layer": round(r_layer, 4),
        "repair_zero_attn": round(r_attn, 4),
        "repair_zero_mlp": round(r_mlp, 4),
        "ratio_mlp_over_attn": round(ratio_mlp_attn, 3),
        "ratio_mlp_over_layer": round(ratio_mlp_layer, 3),
        "attn_confound": bool(attn_confound),
        "mlp_is_locus": bool(mlp_is_locus),
        "label": label,
    }


# ==================== torch-free CPU self-test ====================

def selftest():
    ok = []
    def chk(name, cond):
        ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # (1) bucket_key covers all 4 ledger cells with the right R/G semantics
    chk("bucket LE+/SR- -> R1G0 (sick)", bucket_key(True, False) == "R1G0")
    chk("bucket LE-/SR- -> R0G0 (neither)", bucket_key(False, False) == "R0G0")
    chk("bucket LE+/SR+ -> R1G1 (healthy)", bucket_key(True, True) == "R1G1")
    chk("bucket LE-/SR+ -> R0G1 (off-diag)", bucket_key(False, True) == "R0G1")

    # (2) transition rates: sick bucket all-wrong at baseline; repair == wrong->right frac
    sick_after = [1, 1, 0, 1, 0]              # 3/5 repaired
    tr = transition_rates([0, 0, 0, 0, 0], sick_after)
    chk("repair rate = 3/5", tr["wrong_to_right_rate"] == 0.6)
    chk("no right->wrong when none right at base", tr["right_to_wrong_rate"] is None)
    chk("int_correct = 0.6", tr["int_correct"] == 0.6)
    # healthy bucket all-right at baseline; harm == right->wrong frac, preserve = 1-harm
    hh = transition_rates([1, 1, 1, 1], [1, 1, 0, 1])
    chk("harm rate = 1/4", hh["right_to_wrong_rate"] == 0.25)
    chk("preserve = 3/4", hh["preserve_rate"] == 0.75)
    # neither bucket: wrong->right here is FABRICATION (spurious correctness)
    fab = transition_rates([0, 0, 0, 0], [1, 0, 1, 0])
    chk("fabrication rate = 2/4 (flagged vs chance elsewhere)", fab["wrong_to_right_rate"] == 0.5)

    # (3) randk_summary mean/sd (ddof=1) matches numpy
    per_seed = {8: [0.1, 0.2, 0.3], 32: [0.5, 0.5, 0.5]}
    mean, sd, ns = randk_summary(per_seed)
    chk("randk mean(k=8)=0.2", abs(mean[8] - 0.2) < 1e-9)
    chk("randk sd(k=8)=sample-sd", abs(sd[8] - float(np.std([0.1, 0.2, 0.3], ddof=1))) < 1e-6)
    chk("randk sd(k=32)=0 (no spread)", sd[32] == 0.0)
    chk("randk n_seeds tracked", ns == 3)

    # (4) topk_minus_rand: sigma margin over the random control
    ks = [8, 32]
    tm = topk_minus_rand(ks, {8: 0.7, 32: 0.8}, {8: 0.1, 32: 0.2}, {8: 0.05, 32: 0.1})
    chk("topk-rand diff(k=8)=0.6", tm[8]["diff"] == 0.6)
    chk("topk-rand sigma(k=8)=12", abs(tm[8]["sigma"] - 12.0) < 1e-2)

    # (5) curve_saturation: finds first k >= 0.95*full_zero; None if never reached
    full = 0.80
    curve_ok = {1: 0.1, 2: 0.2, 4: 0.4, 8: 0.6, 16: 0.7, 32: 0.77, 64: 0.79, 128: 0.80}
    chk("saturates at k=32 (0.77>=0.76)", curve_saturation(list(curve_ok), curve_ok, full, 0.95) == 32)
    curve_bad = {1: 0.0, 2: 0.0, 4: 0.02, 8: 0.02, 16: 0.05, 32: 0.02, 64: 0.10, 128: 0.47}
    chk("g431-like curve never saturates (->None, muddy)",
        curve_saturation(list(curve_bad), curve_bad, 0.956, 0.95) is None)

    # (6) mlp_vs_attn_verdict: CLEAN case (l3b-like: mlp~layer>>attn), attn does not repair
    clean = mlp_vs_attn_verdict(base=0.0, zero_layer=0.74, zero_attn=0.05, zero_mlp=0.74)
    chk("clean: mlp_is_locus", clean["mlp_is_locus"] is True)
    chk("clean: label=clean", clean["label"] == "clean")
    chk("clean: no attn confound", clean["attn_confound"] is False)
    chk("clean: mlp/attn ratio large", clean["ratio_mlp_over_attn"] >= 1.5)
    # MUDDY case: attention ALSO repairs comparably (attack-1 would land)
    muddy = mlp_vs_attn_verdict(base=0.0, zero_layer=0.70, zero_attn=0.45, zero_mlp=0.50)
    chk("muddy: not mlp_locus (attn repairs too)", muddy["mlp_is_locus"] is False)
    chk("muddy: attn_confound flagged", muddy["attn_confound"] is True)
    chk("muddy: label muddy or partial", muddy["label"] in ("muddy", "partial"))
    # PARTIAL case: mlp leads but layer >> mlp (residual cross-talk), weak dominance
    part = mlp_vs_attn_verdict(base=0.0, zero_layer=0.90, zero_attn=0.10, zero_mlp=0.30)
    chk("partial: layer>>mlp -> not sole locus", part["mlp_is_locus"] is False)
    chk("partial: label=partial", part["label"] == "partial")

    print(f"\nSELFTEST: {sum(ok)}/{len(ok)} checks passed -> {'ALL PASS' if all(ok) else 'FAILURES PRESENT'}")
    return all(ok)


# ==================== decoder discovery (matches mlp_dissect/mlp_dla) ====================

def decoder_module(m):
    for path in ("model", "model.language_model", "model.text_model",
                 "language_model.model", "language_model"):
        obj = m
        try:
            for att in path.split("."):
                obj = getattr(obj, att)
            _ = obj.layers
            return obj
        except AttributeError:
            continue
    raise AttributeError("no decoder module")


# ==================== torch battery (GPU) ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path"); ap.add_argument("--tokenizer_path")
    ap.add_argument("--layer", type=int); ap.add_argument("--task")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--mm_model", action="store_true")
    ap.add_argument("--topk_panelA", type=int, default=32, help="representative top-k for Panel A")
    ap.add_argument("--ks", default="1,2,4,8,16,32,64,128", help="Panel B channel-count curve")
    ap.add_argument("--rand_seeds", type=int, default=20, help="Panel B random-k control seeds (>=20)")
    ap.add_argument("--sick_items", type=int, default=40, help="cap on R1G0 items used in Panel A/B")
    ap.add_argument("--bucket_items", type=int, default=40, help="cap per bucket for Panel C")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        import sys; sys.exit(0 if selftest() else 1)
    assert args.model_path and args.layer is not None and args.task and args.out, \
        "need --model_path --layer --task --out (or --selftest)"

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
    import eval_drllm

    ks = [int(x) for x in args.ks.split(",")]
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    Loader = AutoModelForImageTextToText if args.mm_model else AutoModelForCausalLM
    model = Loader.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"})
    model.eval()
    dec = decoder_module(model)
    layer = dec.layers[args.layer]
    mlp = layer.mlp
    attn = layer.self_attn

    # ---- unified hook state driving all interventions ----
    #   P["mlp_mode"] in {None,"zero","project_v"} ; P["attn_zero"] bool ;
    #   P["dp_idx"] channel-mask tensor (top-k / random-k) ; rec/rec_inter for pass1 capture.
    P = {"rec": False, "rec_inter": False, "mlp_mode": None, "attn_zero": False,
         "dp_idx": None, "vhat": None, "delta": None, "inter": None}

    def mlp_out_hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if P["rec"]:
            P["delta"] = o[:, -1, :].detach()
        mode = P["mlp_mode"]
        if mode == "zero":
            o2 = torch.zeros_like(o)
        elif mode == "project_v" and P["vhat"] is not None:
            vh = P["vhat"].to(o.dtype)
            proj = (o * vh).sum(-1, keepdim=True) * vh     # (o . vhat) vhat, per position
            o2 = o - proj
        else:
            return
        return (o2,) + tuple(out[1:]) if isinstance(out, tuple) else o2

    def attn_out_hook(mod, inp, out):
        if not P["attn_zero"]:
            return
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + tuple(out[1:])
        return torch.zeros_like(out)

    def dp_pre(mod, inp):
        x = inp[0]
        if P["rec_inter"]:
            P["inter"] = x[:, -1, :].detach()
        if P["dp_idx"] is not None:
            x = x.clone(); x[..., P["dp_idx"]] = 0.0
            return (x,) + tuple(inp[1:])

    mlp.register_forward_hook(mlp_out_hook)
    attn.register_forward_hook(attn_out_hook)
    mlp.down_proj.register_forward_pre_hook(dp_pre)

    letters = lambda n: [chr(65 + j) for j in range(n)]
    def parse_letter(txt, n):
        opts = "".join(letters(n))
        m = re.search(rf"[Aa]nswer\s*:?\s*\(?\[?([{opts}])\b", txt)
        if m: return m.group(1).upper()
        m = re.search(rf"(?:^|[^A-Za-z])([{opts}])(?:[^A-Za-z]|$)", txt)
        return m.group(1).upper() if m else "?"
    def logprob(pids, cont):
        cid = tok(cont, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            lg = model(torch.cat([pids, cid], 1)).logits
        lp = torch.log_softmax(lg[0, pids.shape[1] - 1:-1].float(), -1)
        return lp.gather(1, cid[0].unsqueeze(1)).sum().item()
    def gen(pids):
        with torch.no_grad():
            g = model.generate(pids, max_new_tokens=12, do_sample=False,
                               pad_token_id=tok.pad_token_id, use_cache=True)
        return tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True)

    def reset():
        P["mlp_mode"] = None; P["attn_zero"] = False; P["dp_idx"] = None

    def set_mode(mode):
        """mode in {baseline, zero_layer, zero_attn, zero_mlp, project_v}"""
        reset()
        if mode == "zero_layer":
            P["mlp_mode"] = "zero"; P["attn_zero"] = True
        elif mode == "zero_attn":
            P["attn_zero"] = True
        elif mode == "zero_mlp":
            P["mlp_mode"] = "zero"
        elif mode == "project_v":
            P["mlp_mode"] = "project_v"

    def eval_items(items, mode=None, dp_idx=None):
        """commit/correct + per-sample correctness list on a set of {pids,gold,n} items."""
        if mode is not None:
            set_mode(mode)
        if dp_idx is not None:
            reset(); P["dp_idx"] = torch.tensor(dp_idx, device=model.device, dtype=torch.long)
        c = cor = 0; per = []
        for it in items:
            g = gen(it["pids"]); pl = parse_letter(g, it["n"])
            ok = int(pl == it["gold"])
            c += int(pl != "?"); cor += ok; per.append(ok)
        reset()
        m = max(len(items), 1)
        return {"n": len(items), "commit": round(c / m, 4), "correct": round(cor / m, 4),
                "per_correct": per}

    # ================= Pass 1: bucket by LE x SR, capture v + sick intermediate =================
    samples = eval_drllm.load_task(args.task, "short", 0, -1)[: args.limit]
    osum = {SICK: None, HEALTHY: None}; isum = {SICK: None}
    cnt = {SICK: 0, NEITHER: 0, HEALTHY: 0, "R0G1": 0}
    items = {SICK: [], NEITHER: [], HEALTHY: []}
    nchoice = {SICK: [], NEITHER: [], HEALTHY: []}
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt = tok.apply_chat_template([{"role": "user", "content": s["input"]}],
                                         tokenize=False, add_generation_prompt=True)
        pids = tok(prompt, return_tensors="pt").to(model.device).input_ids
        gold = s["correct"]; n = len(s["choices"])
        P["rec"] = True; P["rec_inter"] = True
        with torch.no_grad():
            model(pids)
        o = P["delta"][0].float().cpu(); inter = P["inter"][0].float().cpu()
        P["rec"] = False; P["rec_inter"] = False
        lp = {L: logprob(pids, f"Answer: {L}") for L in letters(n)}
        le = max(lp, key=lp.get) == gold
        base = gen(pids); sr = parse_letter(base, n) == gold
        key = bucket_key(le, sr)
        cnt[key] = cnt.get(key, 0) + 1
        if key in (SICK, HEALTHY):
            osum[key] = o if osum[key] is None else osum[key] + o
        if key == SICK and isum[SICK] is not None:
            isum[SICK] = isum[SICK] + inter
        elif key == SICK:
            isum[SICK] = inter
        if key in items and len(items[key]) < max(args.sick_items, args.bucket_items):
            items[key].append({"pids": pids, "gold": gold, "n": n}); nchoice[key].append(n)
        if i % 50 == 0:
            print(f"p1 {i}/{len(samples)} {int(time.time()-t0)}s cnt={cnt}", flush=True)

    if not (cnt[SICK] and cnt[HEALTHY]):
        json.dump({"error": "empty bucket", "cnt": cnt}, open(args.out, "w"))
        print("EMPTY BUCKET", cnt); return

    v = (osum[SICK] / cnt[SICK] - osum[HEALTHY] / cnt[HEALTHY]).to(model.device)   # sick - healthy
    vhat = v / v.norm()
    P["vhat"] = vhat
    mean_sick_int = (isum[SICK] / cnt[SICK]).to(model.device)
    with torch.no_grad():
        Wd = mlp.down_proj.weight.float()                       # [d_model, d_inter]
        chan_align = (vhat.float() @ Wd)                        # down_proj[:,i] . vhat
        chan_score = (mean_sick_int.float() * chan_align).cpu()
    order = torch.argsort(chan_score.abs(), descending=True)
    d_inter = int(chan_score.numel())
    frac32 = round(chan_score.abs()[order[:32]].sum().item() / chan_score.abs().sum().item(), 3)

    sick = items[SICK][: args.sick_items]
    config = {"model": args.model_path, "layer": args.layer, "task": args.task,
              "n_R1G0_sick": cnt[SICK], "n_R0G0": cnt[NEITHER], "n_R1G1_healthy": cnt[HEALTHY],
              "n_R0G1": cnt.get("R0G1", 0), "v_norm": round(v.norm().item(), 3),
              "d_inter": d_inter, "frac_mass_top32": frac32,
              "n_sick_used": len(sick), "rand_seeds": args.rand_seeds, "ks": ks}
    out = {"config": config}

    # ---------------- PANEL A: necessity + MLP-vs-attn (sick items) ----------------
    A = {}
    A["baseline"] = eval_items(sick, mode="baseline")
    A["zero_layer"] = eval_items(sick, mode="zero_layer")
    A["zero_attn"] = eval_items(sick, mode="zero_attn")
    A["zero_mlp"] = eval_items(sick, mode="zero_mlp")
    A["project_v"] = eval_items(sick, mode="project_v")
    A[f"top{args.topk_panelA}"] = eval_items(sick, dp_idx=order[: args.topk_panelA].tolist())
    verdict = mlp_vs_attn_verdict(A["baseline"]["correct"], A["zero_layer"]["correct"],
                                  A["zero_attn"]["correct"], A["zero_mlp"]["correct"])
    out["panelA_necessity"] = {k: {kk: vv for kk, vv in r.items() if kk != "per_correct"}
                               for k, r in A.items()}
    out["panelA_necessity"]["verdict"] = verdict

    # ---------------- PANEL B: minimality (top-k curve vs random-k) ----------------
    full_zero_correct = A["zero_mlp"]["correct"]
    base_correct = A["baseline"]["correct"]
    topk_correct = {}
    for k in ks:
        kk = min(k, d_inter)
        topk_correct[k] = eval_items(sick, dp_idx=order[:kk].tolist())["correct"]
    import random as _pyr
    rand_per_seed = {k: [] for k in ks}
    for seed in range(args.rand_seeds):
        rng = _pyr.Random(1000 + seed)
        for k in ks:
            kk = min(k, d_inter)
            idx = rng.sample(range(d_inter), kk)
            rand_per_seed[k].append(eval_items(sick, dp_idx=idx)["correct"])
    rand_mean, rand_sd, n_seeds = randk_summary(rand_per_seed)
    margin = topk_minus_rand(ks, topk_correct, rand_mean, rand_sd)
    sat_k = curve_saturation(ks, topk_correct, full_zero_correct, frac=0.95)
    out["panelB_minimality"] = {
        "ks": ks, "base_correct": base_correct, "full_zero_correct": full_zero_correct,
        "topk_correct": topk_correct, "randk_correct_mean": rand_mean,
        "randk_correct_sd": rand_sd, "n_rand_seeds": n_seeds,
        "topk_minus_rand": margin, "saturation_k": sat_k, "saturation_frac": 0.95,
        "frac_mass_top32": frac32}

    # ---------------- PANEL C: selectivity by ledger bucket ----------------
    C = {"intervention_primary": "zero_mlp", "intervention_secondary": "project_v",
         "chance_correct": None, "buckets": {}}
    nc_all = [n for lst in nchoice.values() for n in lst]
    if nc_all:
        # modal option count -> chance level for the fabrication comparison (piqa=2 -> 0.5)
        vals, counts = np.unique(np.asarray(nc_all), return_counts=True)
        C["chance_correct"] = round(1.0 / int(vals[int(np.argmax(counts))]), 4)
    for bk in (SICK, NEITHER, HEALTHY):
        bucket_items = items[bk][: args.bucket_items]
        if not bucket_items:
            C["buckets"][bk] = {"n": 0, "note": "empty bucket"}; continue
        b0 = eval_items(bucket_items, mode="baseline")
        bm = eval_items(bucket_items, mode="zero_mlp")
        bp = eval_items(bucket_items, mode="project_v")
        entry = {"n": len(bucket_items),
                 "baseline": {"commit": b0["commit"], "correct": b0["correct"]},
                 "zero_mlp": {"commit": bm["commit"], "correct": bm["correct"],
                              **transition_rates(b0["per_correct"], bm["per_correct"])},
                 "project_v": {"commit": bp["commit"], "correct": bp["correct"],
                               **transition_rates(b0["per_correct"], bp["per_correct"])}}
        # semantic aliases per bucket for the reviewer table
        if bk == SICK:
            entry["repair_rate_zero_mlp"] = entry["zero_mlp"]["wrong_to_right_rate"]
            entry["repair_rate_project_v"] = entry["project_v"]["wrong_to_right_rate"]
        elif bk == NEITHER:
            entry["fabrication_rate_zero_mlp"] = entry["zero_mlp"]["wrong_to_right_rate"]
            entry["fabrication_rate_project_v"] = entry["project_v"]["wrong_to_right_rate"]
        elif bk == HEALTHY:
            entry["harm_rate_zero_mlp"] = entry["zero_mlp"]["right_to_wrong_rate"]
            entry["harm_rate_project_v"] = entry["project_v"]["right_to_wrong_rate"]
        C["buckets"][bk] = entry
    out["panelC_selectivity"] = C

    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)

    # ---------------- readable summary ----------------
    print(f"\n=== BATTERY {args.task} L{args.layer} v_norm={config['v_norm']} "
          f"sick={cnt[SICK]} healthy={cnt[HEALTHY]} neither={cnt[NEITHER]} "
          f"top32_mass={frac32} ===")
    print("PANEL A (sick commit/correct):")
    for k in ("baseline", "zero_layer", "zero_attn", "zero_mlp", "project_v", f"top{args.topk_panelA}"):
        r = out["panelA_necessity"][k]
        print(f"   {k:12}: commit={r['commit']} correct={r['correct']}")
    print(f"   VERDICT: {verdict['label']}  mlp_is_locus={verdict['mlp_is_locus']} "
          f"attn_confound={verdict['attn_confound']} "
          f"(repair mlp={verdict['repair_zero_mlp']} attn={verdict['repair_zero_attn']} "
          f"layer={verdict['repair_zero_layer']})")
    print(f"PANEL B: full_zero={full_zero_correct}  saturation_k(0.95)={sat_k}")
    print("   top-k correct:", topk_correct)
    print("   rand-k mean  :", rand_mean)
    print("   margin sigma :", {k: margin[k]["sigma"] for k in ks})
    print("PANEL C (per bucket, base->zero_mlp correct; key rate):")
    for bk in (SICK, NEITHER, HEALTHY):
        e = C["buckets"].get(bk, {})
        if e.get("n"):
            zm = e["zero_mlp"]
            print(f"   {bk}: n={e['n']} base={e['baseline']['correct']} "
                  f"zero_mlp={zm['correct']} w2r={zm['wrong_to_right_rate']} "
                  f"r2w={zm['right_to_wrong_rate']}")
    print(f"   chance_correct(fabrication ref)={C['chance_correct']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
