#!/usr/bin/env python3
"""BEHAVIOR-TOKEN-SET DLA (grouped direct-logit attribution of the realization scar).

Raw logit-lens on the mid-layer MLP write v returns multilingual junk (single top-1
tokens do NOT capture a mid-layer write; cf. dissect_dissect_g412.json top_promoted_by_v =
'<|image>','다','dise',...). This tool instead does GROUPED attribution: define behavior
token SETS and measure v's (and the top-k channel reconstruction v_top32's) RELATIVE
logit contribution to each GROUP, not to individual tokens. Group comparison is robust to
the absolute logit-lens noise/offset because we (a) center each group's mean logit against
the global vocab mean and z-score by the global std, and (b) calibrate every group against
TWO controls: random token groups (noise floor for group-z) and random directions of the
same norm as v (null distribution for each group's loading).

Pipeline (reuses mlp_patch pass1 + mlp_dissect/mlp_patchscope channel decomposition):
  Pass1: bucket samples by LE x SR at the answer-decision position, accumulate mean MLP
         OUT and mean down_proj INPUT for LE+/SR- (sick) and LE+/SR+ (healthy), and the
         gold-letter distribution of the sick bucket.
    v        = mean_out[sick] - mean_out[healthy]            (full scar write direction)
    int_diff = mean_int[sick] - mean_int[healthy]            (per-neuron activation diff)
    down_proj(int_diff) == v exactly (gated MLP out is linear in intermediate, no bias),
    so v decomposes over channels: score[i]=int_diff[i]*(v_hat . down_proj[:,i]).
    v_top32  = down_proj(mask_top32(int_diff))               (the sparse subcircuit's write)
  DLA: logits(d) = W_U @ final_norm(d) for d in {v, v_top32}; per group g report
       z[g] = (mean_{t in g} logits[t] - global_mean) / global_std, normalized shares among
       behavior groups, the dominant group + sign, and the sign-meaningful phenotype
       CONTRASTS (defer-commit for the gemma hedging phenotype; abias-correct for the llama
       A-bias phenotype), each expressed in sigma above a random-direction null.

Behavior groups (token sets, built robustly against the real tokenizer: leading-space AND
capitalization variants, first non-whitespace subword so multi-token words like
'unfortunately'/'therefore' are captured via their reliably-single-token leading-space form;
pure-whitespace tokens such as the ' ' in ' 1' are skipped):
  commit          : "Answer","The answer","answer",":", option letters A/B/C/D (answer pos)
  defer           : Please, provide, need, cannot, context, statement, unable, clarify, unfortunately
  abias           : the standalone letter "A" (llama phenotype)
  correct_letter  : the gold letters of the sick bucket (built at runtime)
  reasoning(ctrl) : step, first, because, therefore, ... and digits 0-9
  rand_tok_*(ctrl): random vocab token groups (group-z noise floor, must be ~0)

Success = gemma L29: defer promoted &/or commit suppressed (defer-commit contrast >> null);
          llama L20: abias promoted &/or correct_letter suppressed (abias-correct >> null).

Usage:  python mlp_dla.py --model_path P --tokenizer_path P --layer L --task T
          --limit 250 [--mm_model] --out X.json
        python mlp_dla.py --selftest                 # torch-free CPU logic check (numpy + FakeTok)
        python mlp_dla.py --tok_report P [--mm_model] # print the group token-id sets for a real tokenizer
"""
import argparse, json, re, time, random
import numpy as np


# ==================== pure, torch-free, unit-testable pieces ====================

def lead_token_id(tok, s):
    """First NON-whitespace token id of s (add_special_tokens=False), else None.
    Skips a pure-whitespace leading piece so ' 1' -> id('1') not id(' ')."""
    ids = tok(s, add_special_tokens=False).input_ids
    for i in ids:
        if tok.decode([int(i)]).strip() != "":
            return int(i)
    return None


def build_group_ids(tok, surfaces, add_space=True):
    """Deduped list of lead-token ids for the given surface strings. For each surface we
    also try its leading-space variant (the emission-relevant, reliably-single-token form).
    Case is NOT folded here (put the exact casings you want in `surfaces`)."""
    ids, detail, seen = [], {}, set()
    for s in surfaces:
        variants = [s]
        if add_space and not s.startswith(" "):
            variants.append(" " + s)
        for v in variants:
            t = lead_token_id(tok, v)
            if t is not None and t not in seen:
                seen.add(t); ids.append(t); detail[t] = v
    return ids, detail


# Behavior token sets (exact casings chosen deliberately; A NOT lowercased -> no 'a' article).
BEHAVIOR_SURFACES = {
    "commit":    ["Answer", "answer", "The answer", ":", "A", "B", "C", "D"],
    "defer":     ["Please", "please", "provide", "need", "cannot", "context",
                  "statement", "unable", "clarify", "unfortunately"],
    "abias":     ["A"],
    "reasoning": ["step", "first", "because", "therefore", "second", "then", "since"]
                 + [str(d) for d in range(10)],
}
BEHAVIOR_NAMES = ["commit", "defer", "abias", "correct_letter", "reasoning"]


def build_all_groups(tok, gold_letters, vocab_size, n_random_groups=8, random_group_size=40, seed=0):
    """Returns (groups: name->list[int], details: name->{id:surface}). Adds the dynamic
    correct_letter group (from sick-bucket gold letters) and n_random_groups control groups."""
    groups, details = {}, {}
    for name, surf in BEHAVIOR_SURFACES.items():
        ids, det = build_group_ids(tok, surf, add_space=True)
        groups[name], details[name] = ids, det
    cl_surf = sorted(set(gold_letters))
    ids, det = build_group_ids(tok, cl_surf, add_space=True)
    groups["correct_letter"], details["correct_letter"] = ids, det

    # random token-id control groups: avoid special ids and pure-whitespace decodes
    rng = random.Random(seed)
    special = set(getattr(tok, "all_special_ids", []) or [])
    pool, tries, want = [], 0, n_random_groups * random_group_size
    while len(pool) < want and tries < want * 40:
        tid = rng.randrange(vocab_size); tries += 1
        if tid in special:
            continue
        if tok.decode([tid]).strip() == "":
            continue
        pool.append(tid)
    for gi in range(n_random_groups):
        chunk = pool[gi * random_group_size:(gi + 1) * random_group_size]
        if chunk:
            groups[f"rand_tok_{gi}"] = chunk
    return groups, details


def group_stats(logits, groups):
    """Per-group DLA stats for a vocab logit vector. z = (group_mean - global_mean)/global_std.
    Centering+z-scoring removes the logit-lens global offset/scale (affine-invariant)."""
    logits = np.asarray(logits, dtype=np.float64)
    gm = float(logits.mean()); gs = float(logits.std()) + 1e-9
    out = {"_global_mean": round(gm, 4), "_global_std": round(gs, 4)}
    for name, ids in groups.items():
        if not ids:
            out[name] = {"n": 0, "raw_mean": None, "raw_sum": None, "centered": None, "z": None}
            continue
        vals = logits[np.asarray(ids, dtype=np.int64)]
        rm = float(vals.mean())
        out[name] = {"n": len(ids), "raw_mean": round(rm, 4), "raw_sum": round(float(vals.sum()), 4),
                     "centered": round(rm - gm, 4), "z": round((rm - gm) / gs, 4)}
    return out


def behavior_shares(stats, behavior_names=BEHAVIOR_NAMES):
    """Normalized |centered| share among behavior groups (which group holds the most mass)."""
    mags = {n: abs(stats[n]["centered"]) for n in behavior_names
            if stats.get(n, {}).get("centered") is not None}
    tot = sum(mags.values()) + 1e-12
    return {n: round(mags[n] / tot, 4) for n in mags}


def dominant_behavior(stats, behavior_names=BEHAVIOR_NAMES):
    """Behavior group with the largest |z|, and whether v promotes or suppresses it."""
    cand = {n: stats[n]["z"] for n in behavior_names if stats.get(n, {}).get("z") is not None}
    if not cand:
        return None
    dom = max(cand, key=lambda n: abs(cand[n]))
    return {"group": dom, "z": cand[dom], "sign": "promoted" if cand[dom] > 0 else "suppressed"}


def rand_tok_floor(stats):
    """Noise floor: mean |z| over the random token control groups (should be small)."""
    zs = [abs(v["z"]) for k, v in stats.items()
          if k.startswith("rand_tok_") and isinstance(v, dict) and v.get("z") is not None]
    return round(float(np.mean(zs)), 4) if zs else None


def null_z_summary(list_of_stats, names):
    """Aggregate per-group z across many random directions -> mean/std (the null)."""
    out = {}
    for n in names:
        zs = [s[n]["z"] for s in list_of_stats if s.get(n, {}).get("z") is not None]
        if zs:
            a = np.asarray(zs, dtype=np.float64)
            out[n] = {"null_z_mean": round(float(a.mean()), 4), "null_z_std": round(float(a.std()) + 1e-9, 4)}
    return out


def contrast_vs_null(z_a, z_b, null_list, a, b):
    """Sign-meaningful contrast (z_a - z_b) expressed in sigma above a random-direction null.
    Common-mode logit offset cancels in the difference, so this is the robust phenotype test."""
    obs = z_a - z_b
    diffs = [s[a]["z"] - s[b]["z"] for s in null_list
             if s.get(a, {}).get("z") is not None and s.get(b, {}).get("z") is not None]
    if not diffs:
        return {"contrast": round(obs, 4), "null_mean": None, "null_std": None, "sigma": None}
    arr = np.asarray(diffs, dtype=np.float64)
    mu, sd = float(arr.mean()), float(arr.std()) + 1e-9
    return {"contrast": round(obs, 4), "null_mean": round(mu, 4), "null_std": round(sd, 4),
            "sigma": round((obs - mu) / sd, 3)}


# ==================== torch-free CPU self-test ====================

def selftest():
    """Verify the pure token-set + grouping/normalization logic with a FakeTok that
    reproduces real tokenizer quirks (leading-space single token, no-space multi token,
    space+digit split) and dummy numpy logits. No torch, no model, no transformers."""
    ok = []
    def chk(name, cond):
        ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # FakeTok: explicit id<->surface table mirroring the real tokenizers probed on dev.
    DEC = {0: " ", 99: "",  # 0=whitespace piece, 99=special (empty decode)
           1: "Answer", 2: " Answer", 3: "answer", 4: " answer", 5: ":",
           6: "A", 7: " A", 8: "B", 9: " B", 10: "C", 11: " C", 12: "D", 13: " D",
           14: "un", 15: "fortunately", 16: " unfortunately",
           17: "Please", 18: " Please", 19: "please", 20: " please",
           21: "provide", 22: " provide", 23: "context", 24: " context",
           25: "step", 26: " step", 27: "first", 28: " first", 29: "1", 30: "2", 31: "3"}
    # per-string encode table (only surfaces the test uses); mirrors observed splits
    ENC = {" ": [0], "Answer": [1], " Answer": [2], "answer": [3], " answer": [4],
           ":": [5], " :": [0, 5], "A": [6], " A": [7], "B": [8], " B": [9],
           "C": [10], " C": [11], "D": [12], " D": [13],
           "unfortunately": [14, 15], " unfortunately": [16],   # multi vs single (space form)
           "Please": [17], " Please": [18], "please": [19], " please": [20],
           "provide": [21], " provide": [22], "context": [23], " context": [24],
           "statement": [23], " statement": [24], "need": [21], " need": [22],
           "cannot": [21], " cannot": [22], "unable": [21], " unable": [22],
           "clarify": [21], " clarify": [22], "The answer": [1], " The answer": [0, 1],
           "step": [25], " step": [26], "first": [27], " first": [28],
           "because": [27], " because": [28], "therefore": [27], " therefore": [28],
           "second": [27], " second": [28], "then": [27], " then": [28], "since": [27], " since": [28],
           "1": [29], " 1": [0, 29], "2": [30], " 2": [0, 30], "3": [31], " 3": [0, 31],
           "0": [29], " 0": [0, 29], "4": [30], " 4": [0, 30], "5": [31], " 5": [0, 31],
           "6": [29], " 6": [0, 29], "7": [30], " 7": [0, 30], "8": [31], " 8": [0, 31], "9": [29], " 9": [0, 29]}

    class FakeTok:
        vocab_size = 200
        all_special_ids = [99]
        def __call__(self, text, add_special_tokens=False):
            ids = ENC.get(text)
            if ids is None:  # random-pool decode only; encode of unknowns not needed by test
                ids = [100 + (abs(hash(text)) % 90)]
            return type("T", (), {"input_ids": list(ids)})
        def decode(self, ids):
            return "".join(DEC.get(int(i), f"t{int(i)}") for i in ids)
    ft = FakeTok()

    # (1) lead_token_id skips pure-whitespace piece
    chk("lead(' 1') -> digit id 29 (skips space)", lead_token_id(ft, " 1") == 29)
    chk("lead(' Answer') -> single space-merged id 2", lead_token_id(ft, " Answer") == 2)
    chk("lead(' unfortunately') single tok 16", lead_token_id(ft, " unfortunately") == 16)
    chk("lead('unfortunately') first subword 14", lead_token_id(ft, "unfortunately") == 14)

    # (2) build_group_ids: dedups, includes leading-space variant, captures multi->single
    ids, det = build_group_ids(ft, ["Answer", "unfortunately", ":"], add_space=True)
    chk("group has no-space + space Answer (1,2)", 1 in ids and 2 in ids)
    chk("group captures single-token ' unfortunately' (16)", 16 in ids)
    chk("group ids unique", len(ids) == len(set(ids)))

    # (3) abias must NOT contain lowercase 'a'; only capital A ids
    ab, _ = build_group_ids(ft, BEHAVIOR_SURFACES["abias"], add_space=True)
    chk("abias = {A, ' A'} ids {6,7}", set(ab) == {6, 7})

    # (4) full group build incl dynamic correct_letter + random controls
    groups, details = build_all_groups(ft, gold_letters=["C", "B"], vocab_size=ft.vocab_size,
                                       n_random_groups=6, random_group_size=8, seed=0)
    chk("correct_letter from golds C,B -> {8,9,10,11}", set(groups["correct_letter"]) == {8, 9, 10, 11})
    chk("has >=5 random control groups", sum(k.startswith("rand_tok_") for k in groups) >= 5)
    chk("random groups avoid special id 99", all(99 not in v for k, v in groups.items() if k.startswith("rand_tok_")))

    # (5) group_stats: crafted logits so 'defer' high, 'commit' low, random ~ mean
    vocab = 200
    logits = np.random.RandomState(0).randn(vocab) * 0.5
    for i in groups["defer"]:
        logits[i] += 6.0
    for i in groups["commit"]:
        logits[i] -= 6.0
    st = group_stats(logits, groups)
    chk("z(defer) > 0 (promoted)", st["defer"]["z"] > 1.0)
    chk("z(commit) < 0 (suppressed)", st["commit"]["z"] < -1.0)
    chk("random-token noise floor small (<0.8)", rand_tok_floor(st) < 0.8)

    # (6) affine invariance of z: scale*logits + offset leaves every group z unchanged
    st2 = group_stats(logits * 7.3 - 11.0, groups)
    chk("z invariant to affine rescale", abs(st2["defer"]["z"] - st["defer"]["z"]) < 1e-6
        and abs(st2["commit"]["z"] - st["commit"]["z"]) < 1e-6)

    # (7) shares sum to 1 and dominant group is the extreme one
    sh = behavior_shares(st)
    chk("behavior shares sum to 1 (within 4-dp rounding)", abs(sum(sh.values()) - 1.0) < 1e-3)
    dom = dominant_behavior(st)
    chk("dominant behavior is defer or commit", dom["group"] in ("defer", "commit"))

    # (8) contrast-vs-null: build a null of random directions' stats; observed defer-commit >> null
    rng = np.random.RandomState(1)
    null = [group_stats(rng.randn(vocab) * 0.5, groups) for _ in range(24)]
    ns = null_z_summary(null, ["defer", "commit"])
    ct = contrast_vs_null(st["defer"]["z"], st["commit"]["z"], null, "defer", "commit")
    chk("null defer/commit z ~ 0", abs(ns["defer"]["null_z_mean"]) < 0.6 and abs(ns["commit"]["null_z_mean"]) < 0.6)
    chk("observed defer-commit contrast > 5 sigma over null", ct["sigma"] > 5.0)

    print(f"\nSELFTEST: {sum(ok)}/{len(ok)} checks passed -> {'ALL PASS' if all(ok) else 'FAILURES PRESENT'}")
    return all(ok)


# ==================== real-tokenizer report (no torch model load) ====================

def tok_report(model_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    vs = int(getattr(tok, "vocab_size", len(tok)))
    groups, details = build_all_groups(tok, gold_letters=["A", "B", "C", "D"], vocab_size=vs)
    print(f"# tokenizer={type(tok).__name__} vocab_size={vs}")
    for name in BEHAVIOR_NAMES:
        ids = groups.get(name, [])
        toks = [tok.decode([i]) for i in ids]
        print(f"{name:15} n={len(ids):2}  ids={ids}")
        print(f"{'':15}      toks={toks}")
    nr = sum(k.startswith('rand_tok_') for k in groups)
    print(f"random control groups: {nr}")
    return True


# ==================== decoder discovery (matches mlp_dissect) ====================

def decoder_module(m):
    for path in ("model", "model.language_model", "model.text_model", "language_model.model", "language_model"):
        obj = m
        try:
            for att in path.split("."):
                obj = getattr(obj, att)
            _ = obj.layers
            return obj
        except AttributeError:
            continue
    raise AttributeError("no decoder module")


def channel_scores(int_diff, vhat, Wd):
    """chan_align[i]=v_hat . down_proj[:,i]; score[i]=int_diff[i]*chan_align[i] (sum==|v|)."""
    chan_align = vhat.float() @ Wd.float()
    return chan_align, int_diff.float() * chan_align


def recon_from_channels(int_diff, Wd, idx):
    """down_proj restricted to channel set idx -> partial reconstruction of v in d_model."""
    import torch
    masked = torch.zeros_like(int_diff.float())
    masked[idx] = int_diff.float()[idx]
    return Wd.float() @ masked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path"); ap.add_argument("--tokenizer_path")
    ap.add_argument("--layer", type=int); ap.add_argument("--task")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--mm_model", action="store_true")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--n_null_dirs", type=int, default=32, help="random directions for the null")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tok_report")
    args = ap.parse_args()

    if args.selftest:
        import sys; sys.exit(0 if selftest() else 1)
    if args.tok_report:
        import sys; sys.exit(0 if tok_report(args.tok_report) else 1)

    assert args.model_path and args.layer is not None and args.task and args.out, \
        "need --model_path --layer --task --out (or --selftest / --tok_report)"

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
    import eval_drllm

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    Loader = AutoModelForImageTextToText if args.mm_model else AutoModelForCausalLM
    model = Loader.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"})
    model.eval()
    dec = decoder_module(model); mlp = dec.layers[args.layer].mlp
    W_U = model.get_output_embeddings().weight            # [vocab, d_model]
    final_norm = dec.norm

    # ---- capture hooks: mean MLP out (v) + mean down_proj input (int_diff) ----
    REC = {"on": False, "out": None, "inter": None}
    def cap_out(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if REC["on"]: REC["out"] = o[:, -1, :].detach()
    def cap_in(mod, inp):
        if REC["on"]: REC["inter"] = inp[0][:, -1, :].detach()
    mlp.register_forward_hook(cap_out)
    mlp.down_proj.register_forward_pre_hook(cap_in)

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

    # ================= Pass 1 (as mlp_patch/mlp_dissect): bucket by LE x SR =================
    samples = eval_drllm.load_task(args.task, "short", 0, -1)[: args.limit]
    osum = {"LE+SR-": None, "LE+SR+": None}; isum = {"LE+SR-": None, "LE+SR+": None}
    cnt = {"LE+SR-": 0, "LE+SR+": 0}; gold_ctr = {}
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt = tok.apply_chat_template([{"role": "user", "content": s["input"]}],
                                         tokenize=False, add_generation_prompt=True)
        pids = tok(prompt, return_tensors="pt").to(model.device).input_ids
        gold = s["correct"]; n = len(s["choices"])
        REC["on"] = True
        with torch.no_grad():
            model(pids)
        o = REC["out"][0].float().cpu(); it = REC["inter"][0].float().cpu()
        REC["on"] = False
        lp = {L: logprob(pids, f"Answer: {L}") for L in letters(n)}
        le = max(lp, key=lp.get) == gold
        base = gen(pids); sr = parse_letter(base, n) == gold
        key = "LE+SR-" if (le and not sr) else ("LE+SR+" if (le and sr) else None)
        if key:
            osum[key] = o if osum[key] is None else osum[key] + o
            isum[key] = it if isum[key] is None else isum[key] + it
            cnt[key] += 1
            if key == "LE+SR-":
                gold_ctr[gold] = gold_ctr.get(gold, 0) + 1
        if i % 50 == 0:
            print(f"p1 {i}/{len(samples)} {int(time.time()-t0)}s cnt={cnt}", flush=True)

    if not (cnt["LE+SR-"] and cnt["LE+SR+"]):
        json.dump({"error": "empty bucket", "cnt": cnt}, open(args.out, "w"))
        print("EMPTY BUCKET", cnt); return

    v = (osum["LE+SR-"] / cnt["LE+SR-"] - osum["LE+SR+"] / cnt["LE+SR+"]).to(model.device)
    healthy_write = (osum["LE+SR+"] / cnt["LE+SR+"]).to(model.device)   # control direction
    int_diff = (isum["LE+SR-"] / cnt["LE+SR-"] - isum["LE+SR+"] / cnt["LE+SR+"]).to(model.device)
    vhat = v / v.norm()
    Wd = mlp.down_proj.weight.detach()
    d_inter = int(Wd.shape[1])

    # channel decomposition -> v_top32 (exact partial reconstruction of v)
    recon_full = recon_from_channels(int_diff, Wd, torch.arange(d_inter, device=Wd.device))
    decomp_relerr = ((recon_full.to(v.dtype) - v).norm() / v.norm()).item()
    _, score = channel_scores(int_diff, vhat, Wd)
    order = torch.argsort(score.abs(), descending=True)
    idx_top = order[: args.topk]
    v_top = recon_from_channels(int_diff, Wd, idx_top).to(v.dtype)
    frac_top = round(score.abs()[idx_top].sum().item() / score.abs().sum().item(), 3)

    # ---- build behavior + control token groups against the real tokenizer ----
    gold_letters = sorted(gold_ctr, key=lambda k: -gold_ctr[k])
    vocab_size = int(W_U.shape[0])
    groups, details = build_all_groups(tok, gold_letters, vocab_size,
                                       n_random_groups=8, random_group_size=40, seed=0)

    W_U_f = W_U.float()   # hoist the [vocab, d_model] float32 cast out of the null loop
    def logits_of(d):
        with torch.no_grad():
            return (W_U_f @ final_norm(d.to(W_U.dtype)).float()).cpu().numpy()

    lg_v = logits_of(v); lg_top = logits_of(v_top); lg_heal = logits_of(healthy_write)
    st_v = group_stats(lg_v, groups)
    st_top = group_stats(lg_top, groups)
    st_heal = group_stats(lg_heal, groups)

    # random-direction null (same norm as v), for per-group and contrast significance
    torch.manual_seed(0)
    null_stats = []
    for _ in range(args.n_null_dirs):
        r = torch.randn_like(v); r = r / r.norm() * v.norm()
        null_stats.append(group_stats(logits_of(r), groups))
    null_summary = null_z_summary(null_stats, BEHAVIOR_NAMES)

    out = {"config": {"model": args.model_path, "layer": args.layer, "task": args.task,
                      "n_sick": cnt["LE+SR-"], "n_healthy": cnt["LE+SR+"],
                      "v_norm": round(v.norm().item(), 3), "d_inter": d_inter,
                      "topk": args.topk, "frac_mass_topk": frac_top,
                      "decomp_relerr": round(decomp_relerr, 6),
                      "cos_vtop_v": round(torch.cosine_similarity(v_top.float(), v.float(), 0).item(), 3),
                      "sick_gold_letters": gold_ctr, "n_null_dirs": args.n_null_dirs},
           "group_token_ids": {k: groups[k] for k in BEHAVIOR_NAMES},
           "group_tokens": {k: [tok.decode([i]) for i in groups[k]] for k in BEHAVIOR_NAMES},
           "dla_v_full": {k: st_v[k] for k in ["_global_mean", "_global_std"] + BEHAVIOR_NAMES},
           "dla_v_top{}".format(args.topk): {k: st_top[k] for k in ["_global_mean", "_global_std"] + BEHAVIOR_NAMES},
           "dla_healthy_write_ctrl": {k: st_heal[k] for k in BEHAVIOR_NAMES},
           "behavior_shares_v": behavior_shares(st_v),
           "behavior_shares_vtop": behavior_shares(st_top),
           "dominant_v": dominant_behavior(st_v),
           "dominant_vtop": dominant_behavior(st_top),
           "random_token_floor": {"v": rand_tok_floor(st_v), "vtop": rand_tok_floor(st_top)},
           "random_dir_null_z": null_summary,
           "phenotype_contrasts": {
               "gemma_defer_minus_commit_v": contrast_vs_null(st_v["defer"]["z"], st_v["commit"]["z"], null_stats, "defer", "commit"),
               "gemma_defer_minus_commit_vtop": contrast_vs_null(st_top["defer"]["z"], st_top["commit"]["z"], null_stats, "defer", "commit"),
               "llama_abias_minus_correct_v": contrast_vs_null(st_v["abias"]["z"], st_v["correct_letter"]["z"], null_stats, "abias", "correct_letter"),
               "llama_abias_minus_correct_vtop": contrast_vs_null(st_top["abias"]["z"], st_top["correct_letter"]["z"], null_stats, "abias", "correct_letter"),
           }}
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)

    # ---------- readable summary ----------
    print(f"\n=== DLA {args.task} L{args.layer} v_norm={v.norm().item():.3f} "
          f"sick={cnt['LE+SR-']} healthy={cnt['LE+SR+']} decomp_relerr={decomp_relerr:.1e} "
          f"cos(vtop,v)={out['config']['cos_vtop_v']} topk_mass={frac_top} ===")
    print("gold letters (sick):", gold_ctr, "-> correct_letter group ids:", groups.get("correct_letter"))
    print("group z (v_full):   ", {k: st_v[k]["z"] for k in BEHAVIOR_NAMES})
    print(f"group z (v_top{args.topk}):  ", {k: st_top[k]["z"] for k in BEHAVIOR_NAMES})
    print("healthy-write ctrl z:", {k: st_heal[k]["z"] for k in BEHAVIOR_NAMES})
    print("random-token floor  :", out["random_token_floor"])
    print("random-dir null z   :", {k: (null_summary[k]["null_z_mean"], null_summary[k]["null_z_std"]) for k in null_summary})
    print("dominant (v_full)   :", out["dominant_v"], "| shares:", out["behavior_shares_v"])
    print("PHENOTYPE CONTRASTS (sigma over random-direction null):")
    for k, c in out["phenotype_contrasts"].items():
        print(f"   {k:38}: contrast={c['contrast']:+.3f}  sigma={c['sigma']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
