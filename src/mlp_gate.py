#!/usr/bin/env python3
"""Generalized ADAPTIVE never-hurt realization gate + OPTIONS-FREE trigger search.
Extends mlp_adaptive.py to (a) support numeric/free-form tasks (gsm8k) so the
"never-hurt" claim can be shown where always-zMLP genuinely DESTROYS accuracy, and
(b) compare several per-example triggers -- crucially options-free ones -- to close
the deployable gap for the wrong-commit failure mode (3B/Qwen), where the MCQ-only
le-trigger works but needs the option set.

Per example we decode twice (base, zMLP-zeroed) and score both with eval_drllm.grade
(handles letter / cot-last-Answer / gsm8k-numeric). Each trigger yields an adaptive
prediction = (zMLP response if it fires else base response), graded end-to-end.

Triggers:
  defer          : base did NOT commit a well-formed answer (deferral)  [task-general, DEPLOYABLE]
  le             : loglik-argmax letter != base letter                  [MCQ only; upper-ref]
  disagree       : base answer != zMLP answer                           [options-free]
  disagree_short : disagree AND protocol==short (guard reasoning)       [options-free; wrong-commit frontier]
  conf           : lowest-margin examples, budget matched to #defer      [options-free; confidence baseline]
Also reports base / always_zmlp, per-trigger fire rate, and a confidence diagnostic
(mean first-token margin for committed-correct vs committed-wrong vs deferred).

Usage: python mlp_gate.py --model_path P --tokenizer_path P --layer L
  --eval_specs agieval_logiqa:short,gsm8k:short,arc_challenge:cot,mmlu:short --limit 200 [--mm_model] --out X
"""
import argparse, json, re, statistics, torch
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


def extract_ans(task, resp, proto):
    """Extracted answer token consistent with eval_drllm.grade; None => deferral."""
    if task == "gsm8k" or task.startswith("dart"):
        nums = eval_drllm.GSM_NUM_RE.findall(resp.replace(",", "").replace("$", ""))
        return nums[-1] if nums else None
    if proto == "cot":
        ms = eval_drllm.ARC_COT_RE.findall(resp)
        return ms[-1].upper() if ms else None
    m = eval_drllm.ARC_RE.match(resp.strip())
    return m.group(1).upper() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True); ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--eval_specs", required=True)
    ap.add_argument("--limit", type=int, default=200); ap.add_argument("--mm_model", action="store_true")
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
    def first_tok_conf(pids):
        with torch.no_grad(): lg = model(pids).logits[0, -1].float()
        lp = torch.log_softmax(lg, -1); top2 = torch.topk(lp, 2).values
        return (top2[0] - top2[1]).item()  # log-prob margin (larger = more confident)

    out = {"config": {"model": args.model_path, "layer": args.layer}, "acc": {}}
    for spec in args.eval_specs.split(","):
        task, proto = spec.split(":"); mx = 512 if proto == "cot" else 12
        samples = eval_drllm.load_task(task, proto, 0, -1)[: args.limit]
        recs = []
        for sm in samples:
            pids = chat(sm["input"]); gold = sm["correct"]
            has_ch = bool(sm.get("choices")); nopt = len(sm["choices"]) if has_ch else 0
            Z["on"] = False
            base_resp = gen(pids, mx); marg = first_tok_conf(pids)
            Z["on"] = True; zmlp_resp = gen(pids, mx); Z["on"] = False
            base_ok = eval_drllm.grade(task, base_resp, gold, proto)
            zmlp_ok = eval_drllm.grade(task, zmlp_resp, gold, proto)
            base_ans = extract_ans(task, base_resp, proto); zmlp_ans = extract_ans(task, zmlp_resp, proto)
            committed = base_ans is not None
            tl = None
            if has_ch:
                le = max(letters(nopt), key=lambda L: logprob(pids, f"Answer: {L}"))
                tl = (le != (base_ans or "?"))
            recs.append(dict(base_ok=base_ok, zmlp_ok=zmlp_ok, committed=committed, tl=tl,
                             marg=marg, disagree=(base_ans != zmlp_ans)))
        n = len(recs)
        r3 = lambda x: round(x, 3)
        def acc_of(fire): return r3(sum((rr["zmlp_ok"] if fire(rr) else rr["base_ok"]) for rr in recs) / n)
        def rate_of(fire): return r3(sum(int(bool(fire(rr))) for rr in recs) / n)
        # equal-budget confidence trigger: fire on the k lowest-margin examples, k=#defer-fires
        k = sum(int(not rr["committed"]) for rr in recs)
        conf_thr = sorted(rr["marg"] for rr in recs)[k - 1] if 0 < k <= n else -1e9
        triggers = {
            "defer":          lambda rr: not rr["committed"],
            "le":             lambda rr: bool(rr["tl"]),
            # DEPLOYABLE universal recommendation: options-aware le when the option
            # set exists (catches MCQ wrong-commit), else the task-general safe defer
            # (numeric/open tasks, where disagree over-trusts the corrupted zMLP decode).
            "le_or_defer":    lambda rr: (bool(rr["tl"]) if rr["tl"] is not None else not rr["committed"]),
            "disagree":       lambda rr: rr["disagree"],
            "disagree_short": lambda rr: rr["disagree"] and proto == "short",
            "conf":           lambda rr: rr["marg"] <= conf_thr,
        }
        entry = {"base": r3(sum(rr["base_ok"] for rr in recs) / n),
                 "always_zmlp": r3(sum(rr["zmlp_ok"] for rr in recs) / n)}
        for name, fn in triggers.items():
            entry["ad_" + name] = acc_of(fn); entry["trig_" + name] = rate_of(fn)
        # confidence diagnostic: does first-token margin separate the failure classes?
        def mm(sel):
            v = [rr["marg"] for rr in recs if sel(rr)]
            return r3(sum(v) / len(v)) if v else None
        entry["marg_commit_ok"] = mm(lambda rr: rr["committed"] and rr["base_ok"])
        entry["marg_commit_wrong"] = mm(lambda rr: rr["committed"] and not rr["base_ok"])
        entry["marg_defer"] = mm(lambda rr: not rr["committed"])
        out["acc"][spec] = entry
        print(f"  {spec:<22} base={entry['base']} zmlp={entry['always_zmlp']} | "
              f"defer={entry['ad_defer']}({entry['trig_defer']}) le={entry['ad_le']} "
              f"leORdef={entry['ad_le_or_defer']} dis={entry['ad_disagree']} "
              f"disS={entry['ad_disagree_short']} conf={entry['ad_conf']}", flush=True)
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("\n=== generalized gate: want ad_* >= always_zmlp on SCAR tasks, ad_* == base on REASON tasks ===")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
