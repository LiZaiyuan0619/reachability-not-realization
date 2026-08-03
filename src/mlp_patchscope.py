#!/usr/bin/env python3
"""P1-DEEP capstone: PATCHSCOPES semantic closure of the realization-scar subcircuit.

The failed alternatives this replaces:
  - logit-lens on the mid-layer MLP write -> garbage (already known).
  - readout.py's "patchscope-lite" (add v/|v|*scale to the LAST token of a chat
    "Repeat the word: hello" prompt at the scar layer) -> also garbage
    (echoes 'hello' at scale 4, quote-spam at scale 8/16; see readout_readout_g412.json).

This tool does Patchscopes properly (Ghandeharioun et al. 2024): take a SOURCE
representation (the scar write) and PATCH it into the residual stream of a PLACEHOLDER
token inside a raw few-shot IDENTITY / DESCRIPTION target prompt, then let the model
*verbalize* the representation in free text. Two knobs the failed lite version lacked:
  (1) target-LAYER sweep -- inject at an EARLY/MID layer so the model has depth left to
      turn the direction into words (injecting at the scar layer L leaves too few layers);
  (2) norm-preserving REPLACE -- put the placeholder fully in the source direction but at
      its own (on-manifold) magnitude, instead of a small-norm additive nudge.

SOURCE representations (all live in d_model = residual stream; exact because a gated MLP's
output is a *linear* map of its intermediate: mlp_out = down_proj(intermediate), no bias):
  v      = mean_out[LE+/SR-] - mean_out[LE+/SR+]                 (the full scar direction)
  int_diff = mean_int[LE+/SR-] - mean_int[LE+/SR+]              (per-neuron activation diff)
  => down_proj(int_diff) == v  (verified, rel-err ~1e-7).  Hence v decomposes EXACTLY over
     down_proj input channels (= MLP neurons); each neuron i contributes
     score[i] = int_diff[i] * (v_hat . down_proj[:,i]),  sum_i score[i] = |v|.
  v_topk = down_proj( mask_topk(int_diff) )   -- the SPARSE SUBCIRCUIT's write (k~32 neurons,
           the same neurons mlp_dissect ablates: top-32 ~ full-zero >> random-128).
  v_randk (control) = down_proj( mask_rand(int_diff) ).

CLOSURE (the money result): if injecting v_top32 verbalizes the SAME thing as the full v
(gemma: deferral "Please provide the context... I cannot"; llama: "choose A"), while v_rand32
verbalizes nothing coherent, the scar's semantics are carried by the ~32-neuron subcircuit.

Works on mm (gemma: model.language_model.layers) and text (llama/qwen) alike.

Usage:  python mlp_patchscope.py --model_path P --tokenizer_path P --layer L --task T
          --limit 250 [--mm_model] --out X.json
        python mlp_patchscope.py --selftest      # CPU logic check, no model load
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
    """Wd: [d_model, d_inter]. Returns (chan_align[d_inter], chan_score[d_inter]).
    chan_align[i] = v_hat . down_proj[:, i];  chan_score[i] = int_diff[i]*chan_align[i]
    (sum_i chan_score[i] == v_hat . down_proj(int_diff) == |v|)."""
    chan_align = vhat.float() @ Wd.float()          # [d_inter]
    return chan_align, int_diff.float() * chan_align


def recon_from_channels(int_diff, Wd, idx):
    """down_proj restricted to channel set idx: Wd @ (mask_idx * int_diff) -> [d_model]."""
    masked = torch.zeros_like(int_diff.float())
    masked[idx] = int_diff.float()[idx]
    return Wd.float() @ masked                        # [d_model]


def inject_at(o, pos, source, mode, scale):
    """Patch residual tensor o:[B,T,d] at token position `pos` with `source`:[d]."""
    o = o.clone()
    orig = o[:, pos, :]                                # [B, d]
    sh = source.to(o.dtype)
    unit = sh / (sh.norm() + 1e-8)
    if mode == "replace_norm":                         # direction of source, magnitude of placeholder
        o[:, pos, :] = orig.norm(dim=-1, keepdim=True) * unit
    elif mode == "replace_raw":                        # paper-literal replace
        o[:, pos, :] = sh
    elif mode == "add":                                # additive steer by a scaled unit
        o[:, pos, :] = orig + float(scale) * unit
    else:
        raise ValueError(mode)
    return o


def build_target_ids(tok, left_text, right_text, filler_id):
    """Deterministic placeholder placement: concatenate token-id lists so the placeholder
    position is exact regardless of BPE merges. Returns (ids:list, placeholder_pos:int)."""
    left_ids = tok(left_text, add_special_tokens=True).input_ids
    right_ids = tok(right_text, add_special_tokens=False).input_ids
    ids = list(left_ids) + [int(filler_id)] + list(right_ids)
    return ids, len(left_ids)


# raw few-shot Patchscopes target prompts. (left, right); placeholder sits between them.
TEMPLATES = {
    # token-identity: "A A / B B / <X>" -> model echoes the token X encodes
    "identity": ("cat cat\nsun sun\nhouse house\nwater water\n", ""),
    # entity/word description: "<X>:" after word:desc few-shot -> model describes X
    "describe": ("happy: a feeling of pleasure or contentment.\n"
                 "Paris: the capital city of France.\n"
                 "run: to move quickly on foot.\n", ":"),
    # open gloss: The word "<X>" means -> free continuation
    "meaning": ("The word \"", "\" means"),
}


def selftest():
    """CPU-only logic verification: decoder discovery, exact MLP-write decomposition,
    top-k reconstruction shapes, deterministic placeholder position, injection math &
    prefill-only patching, and readout slicing. No model / no CUDA."""
    import torch.nn as nn
    torch.manual_seed(0)
    ok = []

    def chk(name, cond):
        ok.append(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

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

    # -- (2) EXACT decomposition: down_proj(int_diff) == v  (mlp_out = down_proj(intermediate)) --
    d_model, d_inter = 16, 40
    down = nn.Linear(d_inter, d_model, bias=False)
    int_s, int_h = torch.randn(d_inter), torch.randn(d_inter)
    v = (down(int_s) - down(int_h)).detach()
    int_diff = (int_s - int_h).detach()
    Wd = down.weight.detach()                                    # [d_model, d_inter]
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
    print(f"       cos(top8,v)={cos(r_top,v):.3f}  cos(rand8,v)={cos(r_rnd,v):.3f}")

    # -- (4) deterministic placeholder position via id-list concat --
    class FakeTok:
        def __call__(self, text, add_special_tokens=True):
            base = [99] if add_special_tokens else []          # fake BOS
            ids = base + [1000 + i for i, _ in enumerate(text.split())]
            return type("T", (), {"input_ids": ids})
    ft = FakeTok()
    ids, pos = build_target_ids(ft, "the word is", "means", filler_id=7)
    left_len = len(ft("the word is", add_special_tokens=True).input_ids)
    chk("placeholder pos == len(left_ids)", pos == left_len)
    chk("filler at placeholder pos", ids[pos] == 7)
    chk("total len == left+1+right", len(ids) == left_len + 1 + len(ft("means", add_special_tokens=False).input_ids))

    # -- (5) injection math: replace_norm preserves magnitude; add nudges; other pos untouched --
    o = torch.randn(1, 6, d_model); p = 3; src = torch.randn(d_model)
    o_rn = inject_at(o, p, src, "replace_norm", None)
    chk("replace_norm keeps |h_pos|", abs(o_rn[0, p].norm().item() - o[0, p].norm().item()) < 1e-4)
    chk("replace_norm aligns h_pos to source dir",
        cos(o_rn[0, p], src) > 0.999)
    chk("replace_norm leaves other positions", torch.allclose(o_rn[0, 0], o[0, 0]))
    o_add = inject_at(o, p, src, "add", 5.0)
    chk("add moves h_pos toward source", cos(o_add[0, p] - o[0, p], src) > 0.999)

    # -- (6) prefill-only patch gate: patch when T>1 (prefill), skip when T==1 (decode) --
    INJ = {"active": True, "layer": 2, "pos": p, "src": src, "mode": "replace_norm", "scale": None}
    def gate(o, my_layer):
        if not (INJ["active"] and my_layer == INJ["layer"]):
            return o
        if o.shape[1] <= 1:                                    # decode step -> no patch
            return o
        return inject_at(o, INJ["pos"], INJ["src"], INJ["mode"], INJ["scale"])
    pre = gate(o, 2)                                            # prefill at target layer
    dec_step = gate(torch.randn(1, 1, d_model), 2)             # decode step
    wrong = gate(o, 5)                                          # non-target layer
    chk("gate patches prefill at target layer", not torch.allclose(pre[0, p], o[0, p]))
    chk("gate skips decode step (T==1)", dec_step.shape[1] == 1)
    chk("gate skips non-target layer", torch.allclose(wrong, o))

    # -- (7) readout slice: continuation = decode(gen[prompt_len:]) --
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
    ap.add_argument("--target_layers", default="", help="comma list; default = ~15/30/50%% depth + scar layer")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        import sys
        sys.exit(0 if selftest() else 1)

    assert args.model_path and args.layer is not None and args.task and args.out, \
        "need --model_path --layer --task --out (or --selftest)"

    import torch as _t
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
    import eval_drllm

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    Loader = AutoModelForImageTextToText if args.mm_model else AutoModelForCausalLM
    model = Loader.from_pretrained(args.model_path, torch_dtype=_t.bfloat16, device_map={"": "cuda:0"})
    model.eval()
    dec = decoder_module(model)
    N = len(dec.layers)
    mlp = dec.layers[args.layer].mlp

    # target injection layers (default: early/mid + scar layer, clamped to [1, N-1])
    if args.target_layers.strip():
        target_layers = sorted({int(x) for x in args.target_layers.split(",")})
    else:
        cand = [max(1, int(0.15 * N)), int(0.30 * N), int(0.50 * N), args.layer]
        target_layers = sorted({min(max(1, c), N - 1) for c in cand})

    # ---- hooks: capture MLP output (v) + down_proj input (int_diff) at answer position ----
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
    int_diff = (isum["LE+SR-"] / cnt["LE+SR-"] - isum["LE+SR+"] / cnt["LE+SR+"]).to(model.device)
    vhat = v / v.norm()
    Wd = mlp.down_proj.weight.detach()                            # [d_model, d_inter]
    d_inter = int(Wd.shape[1])

    # exact-decomposition sanity + channel scores + top-k / random-k sources
    recon_full = recon_from_channels(int_diff, Wd, torch.arange(d_inter, device=Wd.device))
    decomp_relerr = ((recon_full.to(v.dtype) - v).norm() / v.norm()).item()
    chan_align, score = channel_scores(int_diff, vhat, Wd)
    order = torch.argsort(score.abs(), descending=True)
    idx_top = order[: args.topk]
    idx_top8 = order[:8]
    rnd = random.Random(0).sample(range(d_inter), min(args.topk, d_inter))
    idx_rnd = torch.tensor(rnd, device=Wd.device)

    def recon(idx): return recon_from_channels(int_diff, Wd, idx).to(v.dtype)
    sources = {
        "v_full": v,
        f"v_top{args.topk}": recon(idx_top),
        "v_top8": recon(idx_top8),
        f"v_rand{args.topk}": recon(idx_rnd),
    }
    cosv = lambda a: torch.cosine_similarity(a.float(), v.float(), 0).item()
    frac = lambda idx: round(score.abs()[idx].sum().item() / score.abs().sum().item(), 3)

    out = {"config": {"model": args.model_path, "layer": args.layer, "task": args.task,
                      "n_sick": cnt["LE+SR-"], "n_healthy": cnt["LE+SR+"],
                      "v_norm": round(v.norm().item(), 3), "num_layers": N, "d_inter": d_inter,
                      "target_layers": target_layers, "topk": args.topk,
                      "decomp_relerr": round(decomp_relerr, 6)},
           "channels": {"top{}_idx".format(args.topk): idx_top.tolist(),
                        "top8_idx": idx_top8.tolist(),
                        "frac_mass_topk": frac(idx_top), "frac_mass_top8": frac(idx_top8)},
           "source_cos_to_v": {k: round(cosv(x), 3) for k, x in sources.items()}}

    # precompute target-prompt ids + placeholder positions
    filler_id = tok("x", add_special_tokens=False).input_ids[-1]
    targ = {name: build_target_ids(tok, L, R, filler_id) for name, (L, R) in TEMPLATES.items()}

    def run_patch(ids, pos, target_layer, src, mode, scale):
        INJ.update(active=True, layer=target_layer, pos=pos, src=src, mode=mode, scale=scale)
        try:
            txt = gen_ids(ids, args.gen_tokens)
        finally:
            INJ.update(active=False, layer=None, src=None)
        return txt

    # baseline (no injection) per template, to see what the template says unperturbed
    out["baseline_no_inject"] = {name: gen_ids(ids, args.gen_tokens) for name, (ids, pos) in targ.items()}

    # ---------- SWEEP on source=v_full: find where/how the scar verbalizes ----------
    inject_settings = [("replace_norm", None), ("add", 6.0), ("add", 12.0)]
    sweep = []
    for lt in target_layers:
        for mode, scale in inject_settings:
            for tname, (ids, pos) in targ.items():
                txt = run_patch(ids, pos, lt, v, mode, scale)
                sweep.append({"target_layer": lt, "mode": mode, "scale": scale,
                              "template": tname, "gen": txt})
    out["sweep_full_v"] = sweep

    # ---------- CLOSURE: fixed strong config, compare the 4 sources ----------
    # closure layers = the two shallowest target layers (most room to verbalize)
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
    print(f"\n=== PATCHSCOPES {args.task} L{args.layer}  v_norm={v.norm().item():.3f} "
          f"sick={cnt['LE+SR-']} healthy={cnt['LE+SR+']} decomp_relerr={decomp_relerr:.2e} ===")
    print("source cos-to-v:", out["source_cos_to_v"], "| top-k mass frac:", frac(idx_top))
    print("baseline (no inject):", {k: v_[:50] for k, v_ in out["baseline_no_inject"].items()})
    print("-- SWEEP full-v (replace_norm rows) --")
    for r in sweep:
        if r["mode"] == "replace_norm":
            print(f"  L{r['target_layer']:>2} {r['template']:>9}: {r['gen'][:80]!r}")
    print("-- CLOSURE: full vs top vs rand (does the subcircuit verbalize the SAME?) --")
    for r in closure:
        tk = f"v_top{args.topk}"; rk = f"v_rand{args.topk}"
        print(f"  L{r['target_layer']} {r['template']:>9} | full={r['v_full'][:45]!r}")
        print(f"     {'':>13} | {tk}={r[tk][:45]!r}  {rk}={r[rk][:40]!r}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
