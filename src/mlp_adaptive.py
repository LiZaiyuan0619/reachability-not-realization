#!/usr/bin/env python3
"""Round-2 METHOD: ADAPTIVE never-hurt realization gate. Per example, only intervene
when the model is already failing to realize a known answer; otherwise leave it.
Two detectors:
  gen-trigger (deployable, no options): baseline generation does NOT commit to a letter
    (deferral) -> re-generate with zMLP.
  le-trigger (MCQ-aware): loglik-argmax disagrees with generated answer -> use zMLP gen.
Reports base / always-zMLP / adaptive(gen) / adaptive(le) per task. On direct-answer
scar tasks the gate fires and repairs; on reasoning/CoT tasks the model commits, the
gate rarely fires, so accuracy is preserved (unlike always-zMLP).
Usage: python mlp_adaptive.py --model_path P --tokenizer_path P --layer L
  --eval_specs agieval_logiqa:short,arc_challenge:short,arc_challenge:cot --limit 150 [--mm_model] --out X
"""
import argparse, json, re, torch
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
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--eval_specs", required=True)
    ap.add_argument("--limit", type=int, default=150); ap.add_argument("--mm_model", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    Loader = AutoModelForImageTextToText if args.mm_model else AutoModelForCausalLM
    model = Loader.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"})
    model.eval(); mlp = decoder_layers(model)[args.layer].mlp
    Z = {"on": False}
    def hook(mod, inp, out):
        if not Z["on"]: return
        o = out[0] if isinstance(out, tuple) else out
        z = torch.zeros_like(o)
        return (z,) + tuple(out[1:]) if isinstance(out, tuple) else z
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

    out = {"config": {"model": args.model_path, "layer": args.layer}, "acc": {}}
    for spec in args.eval_specs.split(","):
        task, proto = spec.split(":"); mx = 512 if proto == "cot" else 12
        samples = eval_drllm.load_task(task, proto, 0, -1)[: args.limit]
        c = {"base": 0, "zero": 0, "adg": 0, "adle": 0}; trig_g = trig_l = n = 0
        for sm in samples:
            pids = chat(sm["input"]); gold = sm["correct"]; nopt = len(sm["choices"]); n += 1
            Z["on"] = False
            sr_base = parse_letter(gen(pids, mx), nopt)
            le = max(letters(nopt), key=lambda L: logprob(pids, f"Answer: {L}"))
            Z["on"] = True; sr_zero = parse_letter(gen(pids, mx), nopt); Z["on"] = False
            committed = sr_base != "?"
            tg = not committed; tl = le != sr_base
            trig_g += int(tg); trig_l += int(tl)
            c["base"] += int(sr_base == gold); c["zero"] += int(sr_zero == gold)
            c["adg"] += int((sr_zero if tg else sr_base) == gold)
            c["adle"] += int((sr_zero if tl else sr_base) == gold)
        r = lambda x: round(x / n, 3)
        out["acc"][spec] = {"base": r(c["base"]), "always_zmlp": r(c["zero"]), "adaptive_gen": r(c["adg"]),
                            "adaptive_le": r(c["adle"]), "trig_gen": r(trig_g), "trig_le": r(trig_l)}
        print(f"  {spec}: base={r(c['base'])} zmlp={r(c['zero'])} adG={r(c['adg'])} adLE={r(c['adle'])} "
              f"trig(gen/le)={r(trig_g)}/{r(trig_l)}", flush=True)
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("\n=== adaptive gate: base / always-zMLP / adaptive-gen-trigger / adaptive-le-trigger ===")
    print("  [direct scar tasks: want adaptive ~= always-zMLP > base ; cot: want adaptive ~= base > always-zMLP]")
    for spec, rr in out["acc"].items():
        print(f"  {spec:<24} base={rr['base']} zmlp={rr['always_zmlp']} adG={rr['adaptive_gen']} adLE={rr['adaptive_le']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
