#!/usr/bin/env python3
"""METHOD test: SURGICAL directional realization repair. Instead of zeroing the whole
MLP (which hurts reasoning), remove ONLY the causally-validated suppression direction
v = mean[LE+/SR-] - mean[LE+/SR+]. Hypothesis: proj-out-v repairs DIRECT-answer (like
zMLP) but PRESERVES reasoning/CoT (unlike zMLP).
  Stage1: compute v on --dir_task.  Stage2: eval accuracy on --eval_specs
  (task:protocol csv) under {base, zmlp, proj}.
Usage: python mlp_dirrepair.py --model_path P --tokenizer_path P --layer L
  --dir_task agieval_logiqa --eval_specs agieval_logiqa:short,arc_challenge:short,arc_challenge:cot
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
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--dir_task", required=True)
    ap.add_argument("--eval_specs", required=True); ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--mm_model", action="store_true"); ap.add_argument("--out", required=True)
    ap.add_argument("--extra_modes", default="",
                    help="comma-list of EXTRA intervention modes appended to the fixed "
                         "7-mode sweep, e.g. 'dose:0.25,dose:0.75' to densify the dose-"
                         "response grid (the dose:A hook already handles any A). Default "
                         "'' = the original sweep, byte-identical to every prior call.")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    Loader = AutoModelForImageTextToText if args.mm_model else AutoModelForCausalLM
    model = Loader.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"})
    model.eval(); mlp = decoder_layers(model)[args.layer].mlp

    # mode: base | zero | proj(rank-1 v) | dose:A (o*A) | steer:K (o - K*v) | subspace (remove top-r)
    P = {"mode": "base", "vhat": None, "v": None, "Q": None, "rec": False, "delta": None}
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if P["rec"]: P["delta"] = o[:, -1, :].detach()
        m = P["mode"]
        if m == "zero": o2 = torch.zeros_like(o)
        elif m == "proj" and P["vhat"] is not None:
            vh = P["vhat"].to(o.dtype); o2 = o - (o * vh).sum(-1, keepdim=True) * vh
        elif m.startswith("dose:"): o2 = o * float(m.split(":")[1])
        elif m.startswith("steer:") and P["v"] is not None:
            o2 = o - float(m.split(":")[1]) * P["v"].to(o.dtype)
        elif m == "subspace" and P["Q"] is not None:
            Q = P["Q"].to(o.dtype); o2 = o - (o @ Q.T) @ Q       # remove top-r subspace
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
        lp = torch.log_softmax(lg[0, pids.shape[1]-1:-1].float(), -1)
        return lp.gather(1, cid[0].unsqueeze(1)).sum().item()
    def chat(inp):
        p = tok.apply_chat_template([{"role": "user", "content": inp}], tokenize=False, add_generation_prompt=True)
        return tok(p, return_tensors="pt").to(model.device).input_ids
    def gen(pids, mx):
        with torch.no_grad():
            g = model.generate(pids, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        return tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True)

    # ---- Stage 1: compute v = mean[LE+/SR-] - mean[LE+/SR+] on dir_task ----
    P["mode"] = "base"; P["rec"] = True
    sums = {"s": None, "h": None}; cnt = {"s": 0, "h": 0}; sick_ds = []
    for i, sm in enumerate(eval_drllm.load_task(args.dir_task, "short", 0, -1)[: args.limit]):
        pids = chat(sm["input"]); gold = sm["correct"]; n = len(sm["choices"])
        with torch.no_grad(): model(pids)
        d = P["delta"][0].float().cpu()
        le = max(letters(n), key=lambda L: logprob(pids, f"Answer: {L}")) == gold
        sr = parse_letter(gen(pids, 12), n) == gold
        k = "s" if (le and not sr) else ("h" if (le and sr) else None)
        if k:
            sums[k] = d if sums[k] is None else sums[k] + d; cnt[k] += 1
            if k == "s": sick_ds.append(d)
        if i % 60 == 0: print(f"stage1 {i} {cnt}", flush=True)
    P["rec"] = False
    if not (cnt["s"] and cnt["h"]):
        json.dump({"error": "empty bucket", "cnt": cnt}, open(args.out, "w")); print("EMPTY", cnt); return
    mean_h = sums["h"] / cnt["h"]
    v = (sums["s"] / cnt["s"] - mean_h)
    P["v"] = v.to(model.device); P["vhat"] = (v / v.norm()).to(model.device)
    X = torch.stack(sick_ds) - mean_h                        # center sick writes on healthy mean
    Vt = torch.linalg.svd(X, full_matrices=False)[2]
    P["Q"] = Vt[:4].to(model.device)                          # top-4 suppression subspace
    print(f"v_norm={v.norm().item():.3f} sick={cnt['s']} healthy={cnt['h']} Q={tuple(P['Q'].shape)}")

    # ---- Stage 2: eval each spec under base / zmlp / proj ----
    sweep_modes = ["base", "zero", "proj", "dose:0.5", "steer:4", "steer:8", "subspace"]
    sweep_modes += [m.strip() for m in args.extra_modes.split(",") if m.strip()]
    out = {"config": {"model": args.model_path, "layer": args.layer, "dir_task": args.dir_task,
                      "v_norm": round(v.norm().item(), 3), "limit": args.limit,
                      "extra_modes": args.extra_modes, "modes": sweep_modes}, "acc": {}}
    for spec in args.eval_specs.split(","):
        task, proto = spec.split(":")
        samples = eval_drllm.load_task(task, proto, 0, -1)[: args.limit]
        mx = 512 if proto == "cot" else 12
        res = {}
        for mode in sweep_modes:
            P["mode"] = mode; cor = n = 0
            for sm in samples:
                pids = chat(sm["input"]); g = gen(pids, mx)
                cor += int(parse_letter(g, len(sm["choices"])) == sm["correct"]); n += 1
            res[mode] = round(cor / n, 3)
        P["mode"] = "base"
        out["acc"][spec] = res
        print(f"  {spec}: " + " ".join(f"{m}={res[m]}" for m in res), flush=True)
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("\n=== intervention sweep: base / zero(crude) / proj / dose:0.5 / steer:k / subspace(top-4) ===")
    print("  [direct: want an intervention REPAIRS like zero>base ; cot: want it PRESERVES like base>zero]")
    for spec, r in out["acc"].items():
        print(f"  {spec:<26} " + " ".join(f"{m}={r[m]}" for m in r))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
