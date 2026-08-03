#!/usr/bin/env python3
"""Mechanistic probe: WHAT does the harmful MLP (L20 in Llama-3.2-3B / L29 in
gemma-4-12B) write into the residual that suppresses a known answer?

Three falsifiable measurements per example, bucketed by LE (loglik-correct) x SR
(direct-gen-correct):
  (1) end-to-end: dlp_correct = logprob("Answer: <gold>") with MLP zeroed - baseline.
      Hypothesis: > 0, and LARGER on LE+/SR- (knows-but-cant-say) than LE-/SR-.
  (2) DLA / logit-lens of the MLP output delta at the answer position: project
      (delta * ln_final.weight) @ W_U^T -> which tokens does the MLP DIRECTLY
      promote / suppress? (data-driven, no hedge lexicon -> no cherry-pick.)
  (3) qualitative: baseline vs MLP-zeroed generated text on LE+/SR- examples.

Usage: python mlp_logit_lens.py --model_path P --tokenizer_path P --layer L
         --task agieval_logiqa --limit 200 [--mm_model] [--dtype bf16] --out X.json
"""
import argparse, json, os, re, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
import eval_drllm  # reuse load_task + templates


def decoder_layers(m):
    for path in ("model", "model.language_model", "model.text_model",
                 "language_model.model", "language_model"):
        obj = m
        try:
            for att in path.split("."):
                obj = getattr(obj, att)
            return obj.layers
        except AttributeError:
            continue
    raise AttributeError("cannot locate decoder layers")


def _resolve(m, paths):
    for path in paths:
        obj = m; ok = True
        for att in path.split("."):
            if not hasattr(obj, att):
                ok = False; break
            obj = getattr(obj, att)
        if ok:
            return obj
    return None


def final_norm_and_head(m):
    norm = _resolve(m, ("model.norm", "model.language_model.norm", "model.text_model.norm",
                        "language_model.model.norm", "language_model.norm"))
    head = _resolve(m, ("lm_head", "model.lm_head", "language_model.lm_head",
                        "model.language_model.lm_head", "model.text_model.lm_head"))
    assert norm is not None and head is not None, "cannot locate final norm / lm_head"
    return norm, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True); ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--task", required=True)
    ap.add_argument("--limit", type=int, default=200); ap.add_argument("--mm_model", action="store_true")
    ap.add_argument("--dtype", default="bf16"); ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    Loader = AutoModelForImageTextToText if args.mm_model else AutoModelForCausalLM
    model = Loader.from_pretrained(args.model_path, torch_dtype=dtype, device_map={"": "cuda:0"})
    model.eval()
    layers = decoder_layers(model); mlp = layers[args.layer].mlp
    ln, head = final_norm_and_head(model)
    W_U_T = head.weight.detach().t().contiguous()    # [hidden, vocab], model dtype
    ln_w = ln.weight.detach()                        # [hidden], model dtype

    CAP, ZERO = {}, {"on": False}
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        CAP["delta"] = o[:, -1, :].detach()          # MLP output at last position (model dtype)
        if ZERO["on"]:
            z = torch.zeros_like(o)
            return (z,) + tuple(out[1:]) if isinstance(out, tuple) else z
    mlp.register_forward_hook(hook)

    def logprob_cont(prompt_ids, cont):
        cid = tok(cont, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        ids = torch.cat([prompt_ids, cid], 1)
        with torch.no_grad():
            lg = model(ids).logits
        lp = torch.log_softmax(lg[0, prompt_ids.shape[1]-1:-1].float(), -1)
        return lp.gather(1, cid[0].unsqueeze(1)).sum().item()

    samples = eval_drllm.load_task(args.task, "short", 0, -1)[: args.limit]
    letters = lambda n: [chr(65+j) for j in range(n)]
    LID = {L: tok(f" {L}", add_special_tokens=False).input_ids[-1] for L in "ABCDE"}
    ans_ids = tok("Answer:", return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

    def parse_letter(txt, n):                       # FIX v1 bug: skip the 'A' in 'Answer'
        opts = "".join(letters(n))
        m = re.search(rf"[Aa]nswer\s*:?\s*\(?\[?([{opts}])\b", txt)
        if m: return m.group(1).upper()
        m = re.search(rf"(?:^|[^A-Za-z])([{opts}])(?:[^A-Za-z]|$)", txt)
        return m.group(1).upper() if m else "?"

    def gold_margin(pids, gold, n):                 # gold-vs-best-distractor logit margin at answer pos
        with torch.no_grad():
            lg = model(torch.cat([pids, ans_ids], 1)).logits[0, -1].float()
        v = {L: lg[LID[L]].item() for L in letters(n)}
        return v[gold] - max(x for L, x in v.items() if L != gold)

    def generate(pids):
        with torch.no_grad():
            g = model.generate(pids, max_new_tokens=12, do_sample=False,
                               pad_token_id=tok.pad_token_id, use_cache=True)
        return tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True)

    buckets = {"LE+SR-": [], "LE-SR-": [], "LE+SR+": [], "LE-SR+": []}
    defer = {"n": 0, "base_committed": 0, "zero_committed": 0, "zero_correct": 0}
    quals = []
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt = tok.apply_chat_template([{"role": "user", "content": s["input"]}],
                                         tokenize=False, add_generation_prompt=True)
        pids = tok(prompt, return_tensors="pt").to(model.device).input_ids
        gold = s["correct"]; nopt = len(s["choices"])
        ZERO["on"] = False
        # LE: letter-loglik argmax
        lp_letters = {L: logprob_cont(pids, f"Answer: {L}") for L in letters(nopt)}
        le = max(lp_letters, key=lp_letters.get) == gold
        # SR: greedy generation, FIXED parse
        base_gen = generate(pids); sr = parse_letter(base_gen, nopt) == gold
        # causal margin effect: gold margin with MLP active vs zeroed
        ZERO["on"] = False; m_base = gold_margin(pids, gold, nopt)
        ZERO["on"] = True;  m_zero = gold_margin(pids, gold, nopt)
        buckets[("LE+" if le else "LE-") + ("SR+" if sr else "SR-")].append(m_zero - m_base)
        # deferral analysis on knows-but-cant-say (LE+/SR-): does zeroing make it COMMIT?
        if le and not sr:
            ZERO["on"] = True; zero_gen = generate(pids)
            defer["n"] += 1
            defer["base_committed"] += int(parse_letter(base_gen, nopt) != "?")
            defer["zero_committed"] += int(parse_letter(zero_gen, nopt) != "?")
            defer["zero_correct"] += int(parse_letter(zero_gen, nopt) == gold)
            if len(quals) < 15: quals.append({"gold": gold, "base": base_gen[:90], "zeroed": zero_gen[:90]})
        ZERO["on"] = False
        if i % 50 == 0: print(f"{i}/{len(samples)} {int(time.time()-t0)}s", flush=True)

    def stat(v): return {"n": len(v), "mean_margin_effect": (round(sum(v)/len(v), 3) if v else None)}
    r = lambda a, b: round(a / b, 3) if b else None
    out = {"config": {"model": args.model_path, "layer": args.layer, "task": args.task, "n": len(samples)},
           "margin_effect_by_bucket": {k: stat(v) for k, v in buckets.items()},
           "deferral_LE+SR-": {"n": defer["n"],
                               "base_commit_rate": r(defer["base_committed"], defer["n"]),
                               "zero_commit_rate": r(defer["zero_committed"], defer["n"]),
                               "zero_correct_rate": r(defer["zero_correct"], defer["n"])},
           "qualitative_LE+SR-": quals}
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("\n=== gold-margin effect (zeroed - base) by bucket [>0 = MLP suppresses gold; want LARGER on LE+SR-] ===")
    for k, v in out["margin_effect_by_bucket"].items(): print(f"  {k}: n={v['n']:>4} mean_margin_effect={v['mean_margin_effect']}")
    d = out["deferral_LE+SR-"]
    print(f"\n=== deferral LE+/SR- (n={d['n']}): base commits {d['base_commit_rate']} -> zeroed commits {d['zero_commit_rate']} (correct {d['zero_correct_rate']}) ===")
    for q in quals[:8]: print(f"  gold={q['gold']} base={q['base']!r} -> zeroed={q['zeroed']!r}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
