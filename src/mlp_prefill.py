#!/usr/bin/env python3
"""PROMPTING-ARTIFACT control: is the realization scar just "the model won't START
answering", fixable by a prompt prefix? Or a genuine generation-path failure that a
prompt can't fix? For each scar-task example, compare commit/correct under:
  base      : free generation (the scar)
  prefill:A : assistant turn seeded with "Answer:"      (force it to start committing)
  prefill:T : assistant turn seeded with "The answer is" (a softer nudge)
  zmlp      : our single-MLP intervention (no prefill)
If prefill ~= base (still wrong) while zmlp repairs -> the scar is NOT a prompting
artifact; forcing a commit surfaces the wrong answer, the loglik-known answer needs the
MLP intervention. If prefill repairs -> honestly report prompting is a cheap alt.
Usage: python mlp_prefill.py --model_path P --tokenizer_path P --layer L
  --eval_specs agieval_logiqa:short,arc_challenge:short --limit 250 [--mm_model] --out X
"""
import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
import eval_drllm
from mlp_gate import decoder_layers, extract_ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True); ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--eval_specs", required=True)
    ap.add_argument("--limit", type=int, default=250); ap.add_argument("--mm_model", action="store_true")
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

    def ids_for(inp, prefix):
        p = tok.apply_chat_template([{"role": "user", "content": inp}], tokenize=False, add_generation_prompt=True)
        return tok(p + prefix, return_tensors="pt").to(model.device).input_ids
    def gen(pids, mx):
        with torch.no_grad():
            g = model.generate(pids, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        return tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True)

    out = {"config": {"model": args.model_path, "layer": args.layer}, "acc": {}}
    for spec in args.eval_specs.split(","):
        task, proto = spec.split(":"); mx = 512 if proto == "cot" else 12
        samples = eval_drllm.load_task(task, proto, 0, -1)[: args.limit]
        conds = {"base": ("", False), "prefill_A": ("Answer:", False),
                 "prefill_T": ("The answer is", False), "zmlp": ("", True)}
        acc = {k: {"commit": 0, "correct": 0} for k in conds}; n = 0
        for sm in samples:
            gold = sm["correct"]; n += 1
            for k, (prefix, zon) in conds.items():
                pids = ids_for(sm["input"], prefix)
                Z["on"] = zon; resp = prefix + gen(pids, mx); Z["on"] = False
                acc[k]["correct"] += eval_drllm.grade(task, resp, gold, proto)
                acc[k]["commit"] += int(extract_ans(task, resp, proto) is not None)
        r = lambda x: round(x / n, 3)
        out["acc"][spec] = {k: {"commit": r(v["commit"]), "correct": r(v["correct"])} for k, v in acc.items()}
        row = out["acc"][spec]
        print(f"  {spec:<22} " + " | ".join(f"{k}={row[k]['correct']}({row[k]['commit']})" for k in conds), flush=True)
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("\n=== prefill control: if prefill_* correct ~= base < zmlp -> scar is NOT a prompting artifact ===")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
