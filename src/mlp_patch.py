#!/usr/bin/env python3
"""v3 CAUSAL SEAL: is the harmful MLP write a DIRECTION that causally controls the
realization failure? Pass1 buckets by LE x SR and captures per-bucket mean MLP write
at the answer-decision position; v = mean[LE+/SR-] - mean[LE+/SR+] (the 'sick-healthy'
write direction). Pass2, via generation (the metric that worked in v2):
  INDUCE  : on healthy LE+/SR+ examples, add +k*v -> does it induce deferral/errors?
  RESCUE  : on sick LE+/SR- examples, add -k*v -> does it commit (like full zero)?
Usage: python mlp_patch.py --model_path P --tokenizer_path P --layer L --task T
         --limit 250 [--mm_model] --out X.json
"""
import argparse, json, re, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
import eval_drllm


def decoder_layers(m):
    for path in ("model", "model.language_model", "model.text_model", "language_model.model", "language_model"):
        obj = m
        try:
            for att in path.split("."): obj = getattr(obj, att)
            return obj.layers
        except AttributeError: continue
    raise AttributeError("no decoder layers")


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
    model.eval(); mlp = decoder_layers(model)[args.layer].mlp

    # hook: records last-pos output (P["rec"]); zeroes (P["zero"]); or adds P["coef"]*P["vec"]
    P = {"rec": False, "zero": False, "vec": None, "coef": 0.0, "delta": None}
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if P["rec"]: P["delta"] = o[:, -1, :].detach()
        if P["zero"]: o2 = torch.zeros_like(o)
        elif P["vec"] is not None and P["coef"] != 0.0: o2 = o + P["coef"] * P["vec"].to(o.dtype)
        else: return
        return (o2,) + tuple(out[1:]) if isinstance(out, tuple) else o2
    mlp.register_forward_hook(hook)

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
    # ---- Pass 1: bucket + capture mean MLP write per bucket ----
    sums = {"LE+SR-": None, "LE+SR+": None}; cnt = {"LE+SR-": 0, "LE+SR+": 0}
    items = {"LE+SR-": [], "LE+SR+": []}
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt = tok.apply_chat_template([{"role": "user", "content": s["input"]}], tokenize=False, add_generation_prompt=True)
        pids = tok(prompt, return_tensors="pt").to(model.device).input_ids
        gold = s["correct"]; n = len(s["choices"])
        P["rec"] = True; P["zero"] = False; P["vec"] = None
        with torch.no_grad(): model(pids)          # capture delta at answer-decision pos
        d = P["delta"][0].float().cpu()
        P["rec"] = False
        lp = {L: logprob(pids, f"Answer: {L}") for L in letters(n)}
        le = max(lp, key=lp.get) == gold
        base = gen(pids); sr = parse_letter(base, n) == gold
        key = "LE+SR-" if (le and not sr) else ("LE+SR+" if (le and sr) else None)
        if key:
            sums[key] = d if sums[key] is None else sums[key] + d
            cnt[key] += 1
            if len(items[key]) < 60: items[key].append({"pids": pids, "gold": gold, "n": n, "base": base})
        if i % 50 == 0: print(f"p1 {i}/{len(samples)} {int(time.time()-t0)}s", flush=True)
    P["rec"] = False
    if not (cnt["LE+SR-"] and cnt["LE+SR+"]):
        json.dump({"error": "empty bucket", "cnt": cnt}, open(args.out, "w")); print("EMPTY BUCKET", cnt); return
    mean_sick = sums["LE+SR-"] / cnt["LE+SR-"]; mean_healthy = sums["LE+SR+"] / cnt["LE+SR+"]
    v = (mean_sick - mean_healthy).to(model.device)   # sick - healthy write direction
    vnorm = v.norm().item()

    def commit_correct(item_list, coef=0.0, zero=False):
        P["zero"] = zero; P["vec"] = None if zero else v; P["coef"] = coef
        c = cor = n = 0; ex = []
        for it in item_list:
            g = gen(it["pids"]); pl = parse_letter(g, it["n"]); n += 1
            c += int(pl != "?"); cor += int(pl == it["gold"])
            if len(ex) < 6: ex.append({"gold": it["gold"], "gen": g[:80]})
        P["zero"] = False; P["vec"] = None; P["coef"] = 0.0
        return {"n": n, "commit": round(c/n, 3) if n else None, "correct": round(cor/n, 3) if n else None, "ex": ex}

    out = {"config": {"model": args.model_path, "layer": args.layer, "task": args.task,
                      "n_sick": cnt["LE+SR-"], "n_healthy": cnt["LE+SR+"], "v_norm": round(vnorm, 3)}}
    # INDUCE on healthy: base vs +k*v  (does the sick direction CAUSE deferral/errors?)
    heal = items["LE+SR+"][:40]
    out["induce_healthy"] = {"base": commit_correct(heal, 0.0)}
    for k in (2.0, 4.0, 8.0):
        out["induce_healthy"][f"+{k}v"] = commit_correct(heal, k)
    # RESCUE on sick: base vs zero vs -k*v  (is the harmful write a low-rank direction?)
    sick = items["LE+SR-"][:40]
    out["rescue_sick"] = {"base": commit_correct(sick, 0.0), "zero": commit_correct(sick, 0.0, zero=True)}
    for k in (2.0, 4.0, 8.0):
        out["rescue_sick"][f"-{k}v"] = commit_correct(sick, -k)
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n=== v_norm={vnorm:.3f}  sick={cnt['LE+SR-']} healthy={cnt['LE+SR+']} ===")
    print("INDUCE (healthy LE+/SR+): base commit/correct -> +k*v  [want commit DROPS = sick dir induces deferral]")
    for k, r in out["induce_healthy"].items(): print(f"  {k:>6}: commit={r['commit']} correct={r['correct']}")
    print("RESCUE (sick LE+/SR-): base -> zero -> -k*v  [want -k*v approaches zero's commit/correct]")
    for k, r in out["rescue_sick"].items(): print(f"  {k:>6}: commit={r['commit']} correct={r['correct']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
