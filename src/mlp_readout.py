#!/usr/bin/env python3
"""P1-DEEP capstone: WHAT does the scar direction MEAN (semantic readout, robust
alternative to the failed logit-lens). Pass1 buckets by LE x SR (as mlp_patch), v =
mean[LE+/SR-] - mean[LE+/SR+]. Then decode FULL free-text (not just the letter) under
directional steering to read the direction's behavioral content:
  on HEALTHY LE+/SR+ prompts:  base vs +k*v   -> does adding v produce DEFERRAL/wrong text?
  on SICK    LE+/SR- prompts:  base vs -k*v   -> does removing v produce COMMITTED answers?
Also a Patchscopes-lite: inject v at layer L into an identity/repeat prompt and decode
the "meaning" token. The qualitative text is the interpretable, quotable evidence:
e.g. "+v turns 'Answer: C' into 'Please provide the statement you are referring to...'".
Usage: python mlp_readout.py --model_path P --tokenizer_path P --layer L --task T
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
    dec = decoder_module(model); mlp = dec.layers[args.layer].mlp; layerL = dec.layers[args.layer]

    P = {"rec": False, "vec": None, "coef": 0.0, "delta": None}
    def mlp_hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if P["rec"]: P["delta"] = o[:, -1, :].detach()
        if P["vec"] is not None and P["coef"] != 0.0:
            o2 = o + P["coef"] * P["vec"].to(o.dtype)
            return (o2,) + tuple(out[1:]) if isinstance(out, tuple) else o2
    mlp.register_forward_hook(mlp_hook)
    # Patchscopes-lite: add a source vector to layer-L residual output at the LAST position
    PS = {"vec": None}
    def layer_hook(mod, inp, out):
        if PS["vec"] is None: return
        o = out[0] if isinstance(out, tuple) else out
        o = o.clone(); o[:, -1, :] = o[:, -1, :] + PS["vec"].to(o.dtype)
        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
    layerL.register_forward_hook(layer_hook)

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
    def gen(pids, mx=40):
        with torch.no_grad():
            g = model.generate(pids, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        return tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True)

    samples = eval_drllm.load_task(args.task, "short", 0, -1)[: args.limit]
    sums = {"LE+SR-": None, "LE+SR+": None}; cnt = {"LE+SR-": 0, "LE+SR+": 0}; items = {"LE+SR-": [], "LE+SR+": []}
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt = tok.apply_chat_template([{"role": "user", "content": s["input"]}], tokenize=False, add_generation_prompt=True)
        pids = tok(prompt, return_tensors="pt").to(model.device).input_ids
        gold = s["correct"]; n = len(s["choices"])
        P["rec"] = True
        with torch.no_grad(): model(pids)
        d = P["delta"][0].float().cpu(); P["rec"] = False
        lp = {L: logprob(pids, f"Answer: {L}") for L in letters(n)}
        le = max(lp, key=lp.get) == gold
        base = gen(pids, 12); sr = parse_letter(base, n) == gold
        key = "LE+SR-" if (le and not sr) else ("LE+SR+" if (le and sr) else None)
        if key:
            sums[key] = d if sums[key] is None else sums[key] + d; cnt[key] += 1
            if len(items[key]) < 12: items[key].append({"pids": pids, "gold": gold, "n": n})
        if i % 50 == 0: print(f"p1 {i}/{len(samples)} {int(time.time()-t0)}s", flush=True)
    if not (cnt["LE+SR-"] and cnt["LE+SR+"]):
        json.dump({"error": "empty bucket", "cnt": cnt}, open(args.out, "w")); print("EMPTY", cnt); return
    v = (sums["LE+SR-"] / cnt["LE+SR-"] - sums["LE+SR+"] / cnt["LE+SR+"]).to(model.device)
    vnorm = v.norm().item()
    out = {"config": {"model": args.model_path, "layer": args.layer, "task": args.task,
                      "n_sick": cnt["LE+SR-"], "n_healthy": cnt["LE+SR+"], "v_norm": round(vnorm, 3)}}

    def readout(pid_list, coefs, mx=40):
        rows = []
        for it in pid_list:
            r = {"gold": it["gold"]}
            for tagc, c in coefs:
                P["vec"] = v; P["coef"] = c
                r[tagc] = gen(it["pids"], mx).strip().replace("\n", " ")[:160]
                P["vec"] = None; P["coef"] = 0.0
            rows.append(r)
        return rows

    # HEALTHY + induce: does adding v produce deferral/wrong TEXT?
    out["induce_text_on_healthy"] = readout(items["LE+SR+"][:8], [("base", 0.0), ("+4v", 4.0), ("+8v", 8.0)])
    # SICK + rescue: does removing v produce committed answer TEXT?
    out["rescue_text_on_sick"] = readout(items["LE+SR-"][:8], [("base", 0.0), ("-4v", -4.0), ("-8v", -8.0)])

    # Patchscopes-lite: inject v into an identity/repeat prompt, decode the "meaning"
    idp = tok.apply_chat_template([{"role": "user", "content": "Repeat the word: hello"}],
                                  tokenize=False, add_generation_prompt=True)
    ipids = tok(idp, return_tensors="pt").to(model.device).input_ids
    ps = {}
    for scale in (4.0, 8.0, 16.0):
        PS["vec"] = (v / vnorm * scale)
        ps[f"inject_{scale}"] = gen(ipids, 24).strip().replace("\n", " ")[:160]
        PS["vec"] = None
    out["patchscope_lite"] = ps

    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n=== v_norm={vnorm:.3f} sick={cnt['LE+SR-']} healthy={cnt['LE+SR+']} ===")
    print("-- INDUCE text on HEALTHY (base -> +v): does +v make it DEFER? --")
    for r in out["induce_text_on_healthy"][:5]: print(f"  base={r['base'][:60]!r} | +8v={r['+8v'][:60]!r}")
    print("-- RESCUE text on SICK (base -> -v): does -v make it COMMIT? --")
    for r in out["rescue_text_on_sick"][:5]: print(f"  base={r['base'][:60]!r} | -8v={r['-8v'][:60]!r}")
    print("-- patchscope-lite inject:", ps)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
