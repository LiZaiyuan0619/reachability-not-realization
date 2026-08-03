#!/usr/bin/env python3
"""P1-DEEP capstone v2: PATCHSCOPES semantic closure of the realization-scar subcircuit,
   FIXED. This is a v2 of mlp_patchscope.py that repairs the two things that made v1 fail.

v1's fatal flaws (see patchscope_g412.json / patchscope_l3b.json):
  (A) the placeholder/filler token was literally "x"  (filler_id = tok("x")). Fluent models
      have an overwhelming "x = a math variable / unknown quantity" prior, so the DESCRIBE and
      MEANING targets just DEFINED x ("a symbol used to represent the unknown quantity",
      "In the equation 2x + 5 = 11 ...") and drowned the injected scar signal.
  (B) the targets asked the model to DEFINE A WORD ("The word \"X\" means", "X: <gloss>").
      That frame elicits dictionary glosses, not the behavior/action the scar encodes.
  (plus gemma writes at a mid layer that is not vocab-readable, giving punctuation/digit spam.)

v2 fixes:
  (1) NEUTRAL FILLER SWEEP. filler is configurable and defaults to the masked-slot box "□"
      (single-token in gemma/llama/qwen, no lexical-definition prior). --fillers runs the
      no-inject baseline for a whole set {□, ?, ??, •, _, …} so we EMPIRICALLY verify the
      chosen filler's baseline is neutral (not already garbage/definitional) before trusting
      any injected readout. --filler picks the one used for the sweep + closure.
  (2) BEHAVIOR/ACTION TARGETS instead of definition targets: the prompts ask "what is the
      writer/model doing / about to do / intending" (gerund/verb answers), not "define this
      word". An echo "identity" target is kept purely as a control.
  (3) FULL CONTROL SET in closure_subcircuit: v_full / v_top32 / v_rand32 / same-layer-random
      direction (rand_dir, |v|-norm) / healthy-baseline write (v_healthy = mean_out[LE+/SR+]).
      Closure result: if v_top32 verbalizes the SAME behavior as v_full while v_rand32 /
      rand_dir / v_healthy do not, the scar's semantics are carried by the ~32-neuron subcircuit.

Source representations (all in d_model; gated-MLP out is a LINEAR map of its intermediate, no
bias, so down_proj(int_diff) == v exactly; v decomposes over down_proj channels = MLP neurons):
  v        = mean_out[LE+/SR-] - mean_out[LE+/SR+]
  int_diff = mean_int[LE+/SR-] - mean_int[LE+/SR+]         (down_proj(int_diff) == v)
  v_topk   = down_proj(mask_topk(int_diff))                (sparse subcircuit write, k~32)
  v_randk  = down_proj(mask_rand(int_diff))                (channel control)
  rand_dir = random Gaussian dir scaled to |v|             (same-layer random-direction control)
  v_healthy= mean_out[LE+/SR+]                             (healthy-baseline write control)

Injection (Patchscopes, Ghandeharioun et al. 2024): patch the SOURCE into the residual stream
at the placeholder position of a raw target prompt at an EARLY/MID target layer (so depth
remains to verbalize), norm-preserving REPLACE by default. Prefill-only (decode steps skip).

Works on mm (gemma: model.language_model.layers) and text (llama/qwen) alike.

Usage:  python mlp_patchscope2.py --model_path P --tokenizer_path P --layer L --task T
          --limit 250 [--mm_model] [--filler □] [--fillers "□,?,??,•,_,…"] --out X.json
        python mlp_patchscope2.py --selftest      # CPU logic check (needs torch, no model/GPU)
"""
import argparse, json, re, time, random, torch


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


# ---------- pure, unit-testable pieces (no model / no CUDA) ----------
def channel_scores(int_diff, vhat, Wd):
    """chan_align[i]=v_hat . down_proj[:,i]; chan_score[i]=int_diff[i]*chan_align[i] (sum==|v|)."""
    chan_align = vhat.float() @ Wd.float()
    return chan_align, int_diff.float() * chan_align


def recon_from_channels(int_diff, Wd, idx):
    """down_proj restricted to channel set idx: Wd @ (mask_idx * int_diff) -> [d_model]."""
    masked = torch.zeros_like(int_diff.float())
    masked[idx] = int_diff.float()[idx]
    return Wd.float() @ masked


def inject_at(o, pos, source, mode, scale):
    """Patch residual tensor o:[B,T,d] at token position `pos` with `source`:[d]."""
    o = o.clone()
    orig = o[:, pos, :]
    sh = source.to(o.dtype)
    unit = sh / (sh.norm() + 1e-8)
    if mode == "replace_norm":                          # source direction, placeholder magnitude
        o[:, pos, :] = orig.norm(dim=-1, keepdim=True) * unit
    elif mode == "replace_raw":                         # paper-literal replace
        o[:, pos, :] = sh
    elif mode == "add":                                 # additive steer by a scaled unit
        o[:, pos, :] = orig + float(scale) * unit
    else:
        raise ValueError(mode)
    return o


def filler_token_id(tok, filler):
    """Deterministic filler id: last subword of the filler string (our default fillers are all
    single-token; falls back gracefully for a multi-piece string)."""
    ids = tok(filler, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"filler {filler!r} tokenizes to nothing")
    return int(ids[-1])


def build_target_ids(tok, left_text, right_text, filler_id):
    """Deterministic placeholder placement: concatenate token-id lists so the placeholder
    position is exact regardless of BPE merges. Returns (ids:list, placeholder_pos:int).
    right_text="" places the placeholder at the very end (open continuation)."""
    left_ids = tok(left_text, add_special_tokens=True).input_ids
    right_ids = tok(right_text, add_special_tokens=False).input_ids
    ids = list(left_ids) + [int(filler_id)] + list(right_ids)
    return ids, len(left_ids)


# BEHAVIOR/ACTION target prompts (NOT definition prompts). (left, right); placeholder between.
# The frame asks what the writer/model is DOING; the injected source biases the completion.
TEMPLATES = {
    # control: token-identity echo ("A A / B B / <X>") -> model echoes the concept X encodes
    "identity": ("cat cat\nsun sun\nhouse house\nwater water\n", ""),
    # behavior: few-shot token->gerund action, read what the injected state is DOING
    "behavior": ('What is the writer doing?\n"thanks" -> thanking\n"help!" -> requesting\n"', '" ->'),
    # intent: one-word intended action of the injected state
    "intent": ('Q: In one word, what is the intended action?\nToken: "', '"\nA:'),
    # commit_or_hedge: does the injected state commit to an answer or hedge/refuse?
    "commit": ('Does it commit to an answer or hedge/refuse?\nToken: "', '"\nIt will'),
}

DEFAULT_FILLERS = ["□", "?", "??", "•", "_", "…"]   # masked-slot box first; all single-token in g/l/q


def selftest():
    """CPU-only logic verification (needs torch for tensor math, but NO model / NO CUDA):
    decoder discovery, exact MLP-write decomposition, top-k recon, deterministic placeholder
    position for ALL templates + ALL fillers incl placeholder-at-end, injection math,
    prefill-only gating, source-control shapes (rand_dir norm, v_healthy), readout slicing."""
    import torch.nn as nn
    torch.manual_seed(0)
    ok = []

    def chk(name, cond):
        ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # -- (1) decoder_module finds .layers for text- and mm-style nesting --
    class Dec(nn.Module):
        def __init__(self): super().__init__(); self.layers = nn.ModuleList([nn.Identity()]); self.norm = nn.Identity()
    class TextM(nn.Module):
        def __init__(self): super().__init__(); self.model = Dec()
    class MMInner(nn.Module):
        def __init__(self): super().__init__(); self.language_model = Dec()
    class MMM(nn.Module):
        def __init__(self): super().__init__(); self.model = MMInner()
    chk("decoder_module text (model.layers)", decoder_module(TextM()).layers is not None)
    chk("decoder_module mm (model.language_model.layers)", decoder_module(MMM()).layers is not None)

    # -- (2) EXACT decomposition: down_proj(int_diff) == v --
    d_model, d_inter = 16, 40
    down = nn.Linear(d_inter, d_model, bias=False)
    int_s, int_h = torch.randn(d_inter), torch.randn(d_inter)
    v = (down(int_s) - down(int_h)).detach()
    int_diff = (int_s - int_h).detach()
    Wd = down.weight.detach()
    recon_full = recon_from_channels(int_diff, Wd, torch.arange(d_inter))
    chk("down_proj(int_diff) == v (rel-err<1e-4)", ((recon_full - v).norm() / v.norm()).item() < 1e-4)

    # -- (3) channel scores sum to |v|; top-k recon shape & better-than-random alignment --
    vhat = v / v.norm()
    chan_align, score = channel_scores(int_diff, vhat, Wd)
    chk("sum(chan_score) == |v|", abs(score.sum().item() - v.norm().item()) < 1e-3)
    order = torch.argsort(score.abs(), descending=True)
    r_top = recon_from_channels(int_diff, Wd, order[:8])
    r_rnd = recon_from_channels(int_diff, Wd, torch.tensor(random.Random(0).sample(range(d_inter), 8)))
    cos = lambda a, b: torch.cosine_similarity(a, b, 0).item()
    chk("recon shape == d_model", tuple(r_top.shape) == (d_model,))
    chk("cos(top8,v) > cos(rand8,v)", cos(r_top, v) > cos(r_rnd, v))

    # -- (4) source controls: rand_dir has |v| norm; v_healthy is an independent d_model vec --
    healthy = down(int_h).detach()
    rr = torch.randn(d_model); rand_dir = rr / rr.norm() * v.norm()
    chk("rand_dir norm == |v|", abs(rand_dir.norm().item() - v.norm().item()) < 1e-4)
    chk("v_healthy shape == d_model", tuple(healthy.shape) == (d_model,))
    chk("rand_dir NOT aligned to v", abs(cos(rand_dir, v)) < 0.9)

    # -- (5) deterministic placeholder position for ALL templates x MANY fillers --
    class FakeTok:
        # space-splitting fake: each whitespace-separated chunk is one id; supports the box/?/etc
        def __call__(self, text, add_special_tokens=True):
            base = [99] if add_special_tokens else []
            ids = base + [1000 + (abs(hash(w)) % 500) for w in text.split()]
            return type("T", (), {"input_ids": ids})
        def decode(self, ids): return " ".join(f"t{i}" for i in ids)
    ft = FakeTok()
    pos_ok = True; endpos_ok = True; between_ok = True
    for fill in ["□", "?", "??", "•", "_", "…", "x"]:
        fid = 424242
        for name, (L, R) in TEMPLATES.items():
            ids, pos = build_target_ids(ft, L, R, fid)
            left_len = len(ft(L, add_special_tokens=True).input_ids)
            right_len = len(ft(R, add_special_tokens=False).input_ids)
            pos_ok &= (pos == left_len)
            pos_ok &= (ids[pos] == fid)
            pos_ok &= (len(ids) == left_len + 1 + right_len)
            if R == "":
                endpos_ok &= (pos == len(ids) - 1)     # placeholder at very end
            else:
                between_ok &= (pos < len(ids) - 1)      # placeholder strictly inside
    chk("placeholder pos exact for all templates x fillers", pos_ok)
    chk("identity: placeholder at END (open continuation)", endpos_ok)
    chk("behavior/intent/commit: placeholder strictly inside", between_ok)

    # -- (6) filler_token_id: single-token fillers -> one id; multi-piece -> last piece --
    class FT2:
        def __call__(self, s, add_special_tokens=False):
            table = {"□": [700], "?": [30], "??": [7801], "•": [6806], "_": [62], "…": [1981]}
            return type("T", (), {"input_ids": table.get(s, [1, 2, 3])})
    ft2 = FT2()
    chk("filler_token_id(box) single id", filler_token_id(ft2, "□") == 700)
    chk("filler_token_id multi-piece -> last", filler_token_id(ft2, "unknownstr") == 3)

    # -- (7) injection math: replace_norm preserves magnitude; add nudges; others untouched --
    o = torch.randn(1, 6, d_model); p = 3; src = torch.randn(d_model)
    o_rn = inject_at(o, p, src, "replace_norm", None)
    chk("replace_norm keeps |h_pos|", abs(o_rn[0, p].norm().item() - o[0, p].norm().item()) < 1e-4)
    chk("replace_norm aligns h_pos to source dir", cos(o_rn[0, p], src) > 0.999)
    chk("replace_norm leaves other positions", torch.allclose(o_rn[0, 0], o[0, 0]))
    o_add = inject_at(o, p, src, "add", 5.0)
    chk("add moves h_pos toward source", cos(o_add[0, p] - o[0, p], src) > 0.999)

    # -- (8) prefill-only patch gate: patch when T>1 (prefill), skip when T==1 (decode) --
    INJ = {"active": True, "layer": 2, "pos": p, "src": src, "mode": "replace_norm", "scale": None}
    def gate(o, my_layer):
        if not (INJ["active"] and my_layer == INJ["layer"]):
            return o
        if o.shape[1] <= 1:
            return o
        return inject_at(o, INJ["pos"], INJ["src"], INJ["mode"], INJ["scale"])
    pre = gate(o, 2); dec_step = gate(torch.randn(1, 1, d_model), 2); wrong = gate(o, 5)
    chk("gate patches prefill at target layer", not torch.allclose(pre[0, p], o[0, p]))
    chk("gate skips decode step (T==1)", dec_step.shape[1] == 1)
    chk("gate skips non-target layer", torch.allclose(wrong, o))

    # -- (9) readout slice: continuation = decode(gen[prompt_len:]) --
    ids, _ = build_target_ids(ft, TEMPLATES["identity"][0], TEMPLATES["identity"][1], 5)
    prompt_len = len(ids)
    fake_gen = list(range(prompt_len)) + [500, 501, 502]
    chk("readout slice picks continuation", fake_gen[prompt_len:] == [500, 501, 502])

    print(f"\nSELFTEST: {sum(ok)}/{len(ok)} checks passed -> {'ALL PASS' if all(ok) else 'FAILURES PRESENT'}")
    return all(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path"); ap.add_argument("--tokenizer_path")
    ap.add_argument("--layer", type=int); ap.add_argument("--task")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--mm_model", action="store_true")
    ap.add_argument("--gen_tokens", type=int, default=32)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--filler", default="□", help="placeholder used for sweep + closure")
    ap.add_argument("--fillers", default=",".join(DEFAULT_FILLERS),
                    help="comma list of fillers for the no-inject neutrality baseline sweep")
    ap.add_argument("--target_layers", default="", help="comma list; default = ~15/30/50%% depth + scar layer")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        import sys
        sys.exit(0 if selftest() else 1)

    assert args.model_path and args.layer is not None and args.task and args.out, \
        "need --model_path --layer --task --out (or --selftest)"

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
    import eval_drllm

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    Loader = AutoModelForImageTextToText if args.mm_model else AutoModelForCausalLM
    model = Loader.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"})
    model.eval()
    dec = decoder_module(model)
    N = len(dec.layers)
    mlp = dec.layers[args.layer].mlp

    if args.target_layers.strip():
        target_layers = sorted({int(x) for x in args.target_layers.split(",")})
    else:
        cand = [max(1, int(0.15 * N)), int(0.30 * N), int(0.50 * N), args.layer]
        target_layers = sorted({min(max(1, c), N - 1) for c in cand})

    # ---- capture hooks: MLP output (v) + down_proj input (int_diff) at answer position ----
    REC = {"on": False, "out": None, "inter": None}
    def cap_out(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if REC["on"]:
            REC["out"] = o[:, -1, :].detach()
    def cap_in(mod, inp):
        if REC["on"]:
            REC["inter"] = inp[0][:, -1, :].detach()
    mlp.register_forward_hook(cap_out)
    mlp.down_proj.register_forward_pre_hook(cap_in)

    # ---- injection hooks on candidate target layers (patch placeholder residual) ----
    INJ = {"active": False, "layer": None, "pos": None, "src": None, "mode": None, "scale": None}
    def make_inj(idx):
        def hook(mod, inp, out):
            if not (INJ["active"] and INJ["layer"] == idx):
                return
            o = out[0] if isinstance(out, tuple) else out
            if o.shape[1] <= 1:                                  # decode step: leave cache alone
                return
            o2 = inject_at(o, INJ["pos"], INJ["src"], INJ["mode"], INJ["scale"])
            return (o2,) + tuple(out[1:]) if isinstance(out, tuple) else o2
        return hook
    for lt in target_layers:
        dec.layers[lt].register_forward_hook(make_inj(lt))

    letters = lambda n: [chr(65 + j) for j in range(n)]
    def parse_letter(txt, n):
        opts = "".join(letters(n))
        m = re.search(rf"[Aa]nswer\s*:?\s*\(?\[?([{opts}])\b", txt)
        if m:
            return m.group(1).upper()
        m = re.search(rf"(?:^|[^A-Za-z])([{opts}])(?:[^A-Za-z]|$)", txt)
        return m.group(1).upper() if m else "?"
    def logprob(pids, cont):
        cid = tok(cont, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            lg = model(torch.cat([pids, cid], 1)).logits
        lp = torch.log_softmax(lg[0, pids.shape[1] - 1:-1].float(), -1)
        return lp.gather(1, cid[0].unsqueeze(1)).sum().item()
    def gen_ids(ids, mx):
        pids = torch.tensor([ids], device=model.device)
        with torch.no_grad():
            g = model.generate(pids, max_new_tokens=mx, do_sample=False,
                               pad_token_id=tok.pad_token_id, use_cache=True)
        return tok.decode(g[0, len(ids):], skip_special_tokens=True).strip().replace("\n", " ")[:200]

    # ================= Pass 1: bucket by LE x SR, accumulate out & inter =================
    samples = eval_drllm.load_task(args.task, "short", 0, -1)[: args.limit]
    osum = {"LE+SR-": None, "LE+SR+": None}
    isum = {"LE+SR-": None, "LE+SR+": None}
    cnt = {"LE+SR-": 0, "LE+SR+": 0}
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
        with torch.no_grad():
            g = model.generate(pids, max_new_tokens=12, do_sample=False,
                               pad_token_id=tok.pad_token_id, use_cache=True)
        base = tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True)
        sr = parse_letter(base, n) == gold
        key = "LE+SR-" if (le and not sr) else ("LE+SR+" if (le and sr) else None)
        if key:
            osum[key] = o if osum[key] is None else osum[key] + o
            isum[key] = it if isum[key] is None else isum[key] + it
            cnt[key] += 1
        if i % 50 == 0:
            print(f"p1 {i}/{len(samples)} {int(time.time()-t0)}s cnt={cnt}", flush=True)

    if not (cnt["LE+SR-"] and cnt["LE+SR+"]):
        json.dump({"error": "empty bucket", "cnt": cnt}, open(args.out, "w"))
        print("EMPTY BUCKET", cnt); return

    v = (osum["LE+SR-"] / cnt["LE+SR-"] - osum["LE+SR+"] / cnt["LE+SR+"]).to(model.device)
    healthy_write = (osum["LE+SR+"] / cnt["LE+SR+"]).to(model.device)      # healthy-baseline source
    int_diff = (isum["LE+SR-"] / cnt["LE+SR-"] - isum["LE+SR+"] / cnt["LE+SR+"]).to(model.device)
    vhat = v / v.norm()
    Wd = mlp.down_proj.weight.detach()
    d_inter = int(Wd.shape[1])

    recon_full = recon_from_channels(int_diff, Wd, torch.arange(d_inter, device=Wd.device))
    decomp_relerr = ((recon_full.to(v.dtype) - v).norm() / v.norm()).item()
    chan_align, score = channel_scores(int_diff, vhat, Wd)
    order = torch.argsort(score.abs(), descending=True)
    idx_top = order[: args.topk]
    rnd = random.Random(0).sample(range(d_inter), min(args.topk, d_inter))
    idx_rnd = torch.tensor(rnd, device=Wd.device)

    def recon(idx): return recon_from_channels(int_diff, Wd, idx).to(v.dtype)
    torch.manual_seed(0)
    rr = torch.randn_like(v); rand_dir = (rr / rr.norm() * v.norm()).to(v.dtype)   # same-layer random dir
    sources = {
        "v_full": v,
        f"v_top{args.topk}": recon(idx_top),
        f"v_rand{args.topk}": recon(idx_rnd),
        "rand_dir": rand_dir,
        "v_healthy": healthy_write,
    }
    cosv = lambda a: torch.cosine_similarity(a.float(), v.float(), 0).item()
    frac = lambda idx: round(score.abs()[idx].sum().item() / score.abs().sum().item(), 3)

    out = {"config": {"model": args.model_path, "layer": args.layer, "task": args.task,
                      "n_sick": cnt["LE+SR-"], "n_healthy": cnt["LE+SR+"],
                      "v_norm": round(v.norm().item(), 3), "num_layers": N, "d_inter": d_inter,
                      "target_layers": target_layers, "topk": args.topk,
                      "filler": args.filler, "fillers_swept": args.fillers.split(","),
                      "decomp_relerr": round(decomp_relerr, 6)},
           "channels": {f"top{args.topk}_idx": idx_top.tolist(),
                        "frac_mass_topk": frac(idx_top)},
           "source_cos_to_v": {k: round(cosv(x), 3) for k, x in sources.items()},
           "source_norm": {k: round(x.norm().item(), 3) for k, x in sources.items()}}

    # filler ids + per-filler target prompts
    fillers = [f for f in args.fillers.split(",") if f != ""]
    if args.filler not in fillers:
        fillers = [args.filler] + fillers
    fid = {f: filler_token_id(tok, f) for f in fillers}
    out["config"]["filler_token_ids"] = fid
    targ_by_filler = {f: {name: build_target_ids(tok, L, R, fid[f]) for name, (L, R) in TEMPLATES.items()}
                      for f in fillers}

    def run_patch(ids, pos, target_layer, src, mode, scale):
        INJ.update(active=True, layer=target_layer, pos=pos, src=src, mode=mode, scale=scale)
        try:
            txt = gen_ids(ids, args.gen_tokens)
        finally:
            INJ.update(active=False, layer=None, src=None)
        return txt

    # ---------- FILLER NEUTRALITY BASELINE: no-inject output for every filler x template ----------
    out["baseline_no_inject"] = {f: {name: gen_ids(ids, args.gen_tokens)
                                     for name, (ids, pos) in targ_by_filler[f].items()}
                                 for f in fillers}

    # primary filler used for the sweep + closure
    targ = targ_by_filler[args.filler]

    # ---------- SWEEP on source=v_full: find where/how the scar verbalizes ----------
    inject_settings = [("replace_norm", None), ("add", 6.0), ("add", 12.0)]
    sweep = []
    for lt in target_layers:
        for mode, scale in inject_settings:
            for tname, (ids, pos) in targ.items():
                sweep.append({"target_layer": lt, "mode": mode, "scale": scale,
                              "template": tname, "gen": run_patch(ids, pos, lt, v, mode, scale)})
    out["sweep_full_v"] = sweep

    # ---------- CLOSURE: fixed strong config, compare the 5 sources ----------
    closure_layers = target_layers[:2] if len(target_layers) >= 2 else target_layers
    closure = []
    for lt in closure_layers:
        for tname, (ids, pos) in targ.items():
            row = {"target_layer": lt, "template": tname, "mode": "replace_norm"}
            for sname, src in sources.items():
                row[sname] = run_patch(ids, pos, lt, src, "replace_norm", None)
            closure.append(row)
    out["closure_subcircuit"] = closure

    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)

    # ---------- readable summary ----------
    print(f"\n=== PATCHSCOPES2 {args.task} L{args.layer} v_norm={v.norm().item():.3f} "
          f"sick={cnt['LE+SR-']} healthy={cnt['LE+SR+']} decomp_relerr={decomp_relerr:.2e} "
          f"filler={args.filler!r} ===")
    print("source cos-to-v:", out["source_cos_to_v"], "| top-k mass frac:", frac(idx_top))
    print("-- FILLER NEUTRALITY (no-inject baseline; want NON-definitional, NON-garbage) --")
    for f in fillers:
        print(f"  filler {f!r}: " + " | ".join(f"{t}={g[:34]!r}" for t, g in out["baseline_no_inject"][f].items()))
    print(f"-- SWEEP full-v (replace_norm rows, filler={args.filler!r}) --")
    for r in sweep:
        if r["mode"] == "replace_norm":
            print(f"  L{r['target_layer']:>2} {r['template']:>9}: {r['gen'][:80]!r}")
    print("-- CLOSURE: full vs top vs rand vs rand_dir vs healthy (does the subcircuit verbalize the SAME?) --")
    for r in closure:
        tk = f"v_top{args.topk}"; rk = f"v_rand{args.topk}"
        print(f"  L{r['target_layer']} {r['template']:>9} | full={r['v_full'][:42]!r}")
        print(f"     {'':>11} | {tk}={r[tk][:42]!r}")
        print(f"     {'':>11} | {rk}={r[rk][:30]!r} rand_dir={r['rand_dir'][:30]!r} healthy={r['v_healthy'][:30]!r}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
