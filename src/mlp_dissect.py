#!/usr/bin/env python3
"""P1-DEEP: OPEN THE MLP WRITE (the Go/No-Go mechanism experiment). Pass1 buckets by
LE x SR (as mlp_patch) and captures, at the answer-decision position, the mean MLP OUTPUT
(-> v = mean[LE+/SR-] - mean[LE+/SR+]) and the mean down_proj INPUT (intermediate acts)
on sick examples. Then:
 (1) behavior-token-set DLA: logit-lens the direction v (final-norm + unembed) and read
     RELATIVE logit mass on token GROUPS {commit, defer, letters} + top tokens. Robust to
     absolute logit-lens noise because we compare group contributions, not raw top-1.
 (2) channel decomposition: score each down_proj input channel i by
     mean_sick_int[i] * (down_proj[:,i] . v_hat) = its contribution to the sick write along v.
 (3) top-k vs random-k channel ablation: zero the top-k v-writing channels during
     generation on sick LE+/SR- examples; compare commit/correct to full-zero and random-k.
     If small top-k ~ full-zero >> random-k, the scar is CONCENTRATED in a few channels.
Usage: python mlp_dissect.py --model_path P --tokenizer_path P --layer L --task T
         --limit 250 [--mm_model] --out X.json
"""
import argparse, json, re, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
import eval_drllm


def decoder_module(m):
    for path in ("model", "model.language_model", "model.text_model", "language_model.model", "language_model"):
        obj = m
        try:
            for att in path.split("."): obj = getattr(obj, att)
            _ = obj.layers
            return obj
        except AttributeError: continue
    raise AttributeError("no decoder module")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True); ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--task", required=True)
    ap.add_argument("--limit", type=int, default=250); ap.add_argument("--mm_model", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    Loader = AutoModelForImageTextToText if args.mm_model else AutoModelForCausalLM
    model = Loader.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"})
    model.eval()
    dec = decoder_module(model); mlp = dec.layers[args.layer].mlp
    W_U = model.get_output_embeddings().weight            # [vocab, d_model]
    final_norm = dec.norm

    # MLP output hook: record last-pos output; zero whole MLP
    P = {"rec": False, "zero": False, "delta": None}
    def out_hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if P["rec"]: P["delta"] = o[:, -1, :].detach()
        if P["zero"]:
            o2 = torch.zeros_like(o)
            return (o2,) + tuple(out[1:]) if isinstance(out, tuple) else o2
    mlp.register_forward_hook(out_hook)
    # down_proj pre-hook: record last-pos intermediate; zero given channel idx set
    D = {"rec": False, "idx": None, "inter": None}
    def dp_pre(mod, inp):
        x = inp[0]
        if D["rec"]: D["inter"] = x[:, -1, :].detach()
        if D["idx"] is not None:
            x = x.clone(); x[..., D["idx"]] = 0.0
            return (x,) + tuple(inp[1:])
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
        with torch.no_grad(): lg = model(torch.cat([pids, cid], 1)).logits
        lp = torch.log_softmax(lg[0, pids.shape[1] - 1:-1].float(), -1)
        return lp.gather(1, cid[0].unsqueeze(1)).sum().item()
    def gen(pids):
        with torch.no_grad():
            g = model.generate(pids, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        return tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True)

    samples = eval_drllm.load_task(args.task, "short", 0, -1)[: args.limit]
    # ---- Pass 1: bucket, capture mean MLP out (v) + mean sick intermediate ----
    osum = {"LE+SR-": None, "LE+SR+": None}; isum = {"LE+SR-": None}; cnt = {"LE+SR-": 0, "LE+SR+": 0}
    sick_items = []
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt = tok.apply_chat_template([{"role": "user", "content": s["input"]}], tokenize=False, add_generation_prompt=True)
        pids = tok(prompt, return_tensors="pt").to(model.device).input_ids
        gold = s["correct"]; n = len(s["choices"])
        P["rec"] = True; D["rec"] = True
        with torch.no_grad(): model(pids)
        o = P["delta"][0].float().cpu(); inter = D["inter"][0].float().cpu()
        P["rec"] = False; D["rec"] = False
        lp = {L: logprob(pids, f"Answer: {L}") for L in letters(n)}
        le = max(lp, key=lp.get) == gold
        base = gen(pids); sr = parse_letter(base, n) == gold
        key = "LE+SR-" if (le and not sr) else ("LE+SR+" if (le and sr) else None)
        if key:
            osum[key] = o if osum[key] is None else osum[key] + o; cnt[key] += 1
            if key == "LE+SR-":
                isum["LE+SR-"] = inter if isum["LE+SR-"] is None else isum["LE+SR-"] + inter
                if len(sick_items) < 45: sick_items.append({"pids": pids, "gold": gold, "n": n})
        if i % 50 == 0: print(f"p1 {i}/{len(samples)} {int(time.time()-t0)}s", flush=True)
    if not (cnt["LE+SR-"] and cnt["LE+SR+"]):
        json.dump({"error": "empty bucket", "cnt": cnt}, open(args.out, "w")); print("EMPTY", cnt); return
    v = (osum["LE+SR-"] / cnt["LE+SR-"] - osum["LE+SR+"] / cnt["LE+SR+"]).to(model.device)  # sick-healthy
    vhat = v / v.norm()
    mean_sick_int = (isum["LE+SR-"] / cnt["LE+SR-"]).to(model.device)                       # [d_inter]
    out = {"config": {"model": args.model_path, "layer": args.layer, "task": args.task,
                      "n_sick": cnt["LE+SR-"], "n_healthy": cnt["LE+SR+"], "v_norm": round(v.norm().item(), 3)}}

    # ---- (1) behavior-token-set DLA (logit-lens on v) ----
    with torch.no_grad():
        vl = (W_U.float() @ final_norm(v.to(W_U.dtype)).float()).cpu()   # [vocab] logit contribution of v
    def first_id(s):
        ids = tok(s, add_special_tokens=False).input_ids; return ids[0] if ids else None
    groups = {"commit": [" Answer", "Answer", " The answer", " answer"],
              "defer": [" Please", " provide", " need", " cannot", " statement", " context", " clarify", " sorry"],
              "letters": [f" {L}" for L in "ABCDE"] + list("ABCDE")}
    gmass = {}
    for gname, toks in groups.items():
        ids = [first_id(t) for t in toks]; ids = [x for x in ids if x is not None]
        gmass[gname] = round(vl[ids].mean().item(), 3) if ids else None
    top = torch.topk(vl, 15); bot = torch.topk(-vl, 15)
    out["dla"] = {"group_logit_of_v": gmass,
                  "top_promoted_by_v": [tok.decode([t]) for t in top.indices.tolist()],
                  "top_suppressed_by_v": [tok.decode([t]) for t in bot.indices.tolist()]}

    # ---- (2) channel decomposition ----
    with torch.no_grad():
        Wd = mlp.down_proj.weight.float()                 # [d_model, d_inter]
        chan_align = (vhat.float() @ Wd)                  # [d_inter] : down_proj[:,i] . vhat
        chan_score = (mean_sick_int.float() * chan_align).cpu()   # contribution of ch i to sick write along v
    order = torch.argsort(chan_score.abs(), descending=True)
    out["channels"] = {"d_inter": int(chan_score.numel()),
                       "top16_idx": order[:16].tolist(),
                       "top16_score": [round(x, 4) for x in chan_score[order[:16]].tolist()],
                       "frac_mass_top32": round(chan_score[order[:32]].abs().sum().item() / chan_score.abs().sum().item(), 3),
                       "frac_mass_top128": round(chan_score[order[:128]].abs().sum().item() / chan_score.abs().sum().item(), 3)}

    # ---- (3) top-k vs random-k channel ablation on sick items ----
    import random as _r; _r.seed(0)
    d_inter = chan_score.numel()
    def eval_ablate(idx):
        D["idx"] = None if idx is None else torch.tensor(idx, device=model.device, dtype=torch.long)
        c = cor = 0
        for it in sick_items:
            g = gen(it["pids"]); pl = parse_letter(g, it["n"])
            c += int(pl != "?"); cor += int(pl == it["gold"])
        D["idx"] = None
        m = len(sick_items)
        return {"commit": round(c/m, 3), "correct": round(cor/m, 3)}
    abl = {"base": eval_ablate(None)}
    P["zero"] = True; abl["full_zero"] = eval_ablate(None); P["zero"] = False   # whole-MLP zero reference
    for k in (8, 32, 128):
        abl[f"top{k}"] = eval_ablate(order[:k].tolist())
    rand128 = _r.sample(range(d_inter), min(128, d_inter))
    abl["rand128"] = eval_ablate(rand128)
    out["ablation_sick"] = abl

    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n=== v_norm={out['config']['v_norm']} sick={cnt['LE+SR-']} ===")
    print("DLA group logit of v:", gmass)
    print("  v promotes:", out['dla']['top_promoted_by_v'][:8])
    print("  v suppresses:", out['dla']['top_suppressed_by_v'][:8])
    print(f"channels: top32 mass frac={out['channels']['frac_mass_top32']} top128={out['channels']['frac_mass_top128']}")
    print("ablation (sick commit/correct):", {k: (v["commit"], v["correct"]) for k, v in abl.items()})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
