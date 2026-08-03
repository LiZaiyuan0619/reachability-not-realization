#!/usr/bin/env python3
"""SPURIOUS-ABSTENTION safety probe. Our gate fires zMLP when the model does NOT commit
(a deferral). On truly UNANSWERABLE questions, NOT committing is CORRECT — so does the
gate wrongly force a (false) answer there? Uses AbstentionBench answerable/unanswerable
pairs (gsm8k_abstain: unanswerable = same question with the context stripped).
Reports, per task, stratified by answerability:
  answerable   -> accuracy       (base / always-zMLP / gate=le_or_defer)  [want gate >= base]
  unanswerable -> false-answer %  (committed a specific answer to an unanswerable q)  [want gate ~ base, NOT higher]
If gate false-answer >> base on unanswerable, the deferral gate cannot tell genuine
abstention from a scar deferral -> honest NEGATIVE for the abstention framing.
Usage: python mlp_abstain.py --model_path P --tokenizer_path P --layer L
  --tasks gsm8k_abstain --limit 300 [--mm_model] --out X
"""
import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
import abstention_loader as AB
from mlp_gate import decoder_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True); ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--tasks", required=True)
    ap.add_argument("--limit", type=int, default=300); ap.add_argument("--mm_model", action="store_true")
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
    def chat(inp):
        p = tok.apply_chat_template([{"role": "user", "content": inp}], tokenize=False, add_generation_prompt=True)
        return tok(p, return_tensors="pt").to(model.device).input_ids
    def gen(pids, mx=20):
        with torch.no_grad():
            g = model.generate(pids, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        return tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True)

    out = {"config": {"model": args.model_path, "layer": args.layer}, "results": {}}
    for task in args.tasks.split(","):
        samples = AB.load_task(task)[: args.limit]
        agg = {ans: {m: {"correct": 0, "false": 0, "n": 0} for m in ("base", "zmlp", "gate")}
               for ans in (True, False)}
        for sm in samples:
            pids = chat(sm["input"]); ans = sm["answerable"]
            Z["on"] = False; base_resp = gen(pids)
            Z["on"] = True; zmlp_resp = gen(pids); Z["on"] = False
            committed = AB.extract_committed_answer(task, base_resp) is not None
            gate_resp = zmlp_resp if not committed else base_resp   # le_or_defer == defer (no options)
            for m, resp in (("base", base_resp), ("zmlp", zmlp_resp), ("gate", gate_resp)):
                d = agg[ans][m]; d["n"] += 1
                if ans: d["correct"] += AB.grade(task, resp, sm["correct"])
                else:   d["false"] += int(AB.is_false_answer(task, resp, ans))
        res = {}
        for ans, label in ((True, "answerable"), (False, "unanswerable")):
            res[label] = {}
            for m in ("base", "zmlp", "gate"):
                d = agg[ans][m]; n = max(d["n"], 1)
                res[label][m] = {"acc": round(d["correct"] / n, 3), "false_answer": round(d["false"] / n, 3), "n": d["n"]}
        out["results"][task] = res
        a, u = res["answerable"], res["unanswerable"]
        print(f"  {task}: ANS acc base={a['base']['acc']} zmlp={a['zmlp']['acc']} gate={a['gate']['acc']} | "
              f"UNANS false base={u['base']['false_answer']} zmlp={u['zmlp']['false_answer']} gate={u['gate']['false_answer']}", flush=True)
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
