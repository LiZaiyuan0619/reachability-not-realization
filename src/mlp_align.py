#!/usr/bin/env python3
"""ALIGNMENT ORIGIN: did post-training INSTALL the scar in this one MLP's weights? Loads
the INSTRUCT and BASE checkpoints of the same family and cross-transplants layer-L MLP:
  I           : instruct as-is (scar present on its LE+/SR- sick examples)
  I<-B mlp    : instruct with BASE's layer-L MLP swapped in -> does the scar DISAPPEAR?
  I zeroMLP   : reference (full-zero repair)
  B           : base on the same prompts (expected: no scar)
  B<-I mlp    : base with INSTRUCT's layer-L MLP swapped in -> does the scar APPEAR?
If I<-B repairs (like zero) AND B<-I induces failure, the scar lives in the instruct
MLP weights = installed by post-training (causal, localized -- what Commitment-Failure
shows only correlationally/behaviorally). Also reports the alignment weight-delta:
|W_down_I - W_down_B| on the top-k v-writing channels vs random channels.
Usage: python mlp_align.py --instruct P --base P --layer L --task T --limit 250 --out X
"""
import argparse, json, re, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import eval_drllm


def decoder_layers(m):
    for path in ("model", "model.language_model", "language_model.model"):
        obj = m
        try:
            for att in path.split("."): obj = getattr(obj, att)
            return obj.layers
        except AttributeError: continue
    raise AttributeError("no layers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--task", required=True)
    ap.add_argument("--limit", type=int, default=250); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.instruct)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    def load(p): return AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}).eval()
    I = load(args.instruct); B = load(args.base)
    Imlp = decoder_layers(I)[args.layer].mlp; Bmlp = decoder_layers(B)[args.layer].mlp
    # saved weight states for swap
    I_state = {k: v.clone() for k, v in Imlp.state_dict().items()}
    B_state = {k: v.clone() for k, v in Bmlp.state_dict().items()}

    Z = {"on": False}
    def zhook(mod, inp, out):
        if not Z["on"]: return
        o = out[0] if isinstance(out, tuple) else out
        z = torch.zeros_like(o)
        return (z,) + tuple(out[1:]) if isinstance(out, tuple) else z
    Imlp.register_forward_hook(zhook)

    letters = lambda n: [chr(65 + j) for j in range(n)]
    def parse_letter(txt, n):
        opts = "".join(letters(n))
        m = re.search(rf"[Aa]nswer\s*:?\s*\(?\[?([{opts}])\b", txt)
        if m: return m.group(1).upper()
        m = re.search(rf"(?:^|[^A-Za-z])([{opts}])(?:[^A-Za-z]|$)", txt)
        return m.group(1).upper() if m else "?"
    def chat_ids(inp):
        p = tok.apply_chat_template([{"role": "user", "content": inp}], tokenize=False, add_generation_prompt=True)
        return tok(p, return_tensors="pt").to("cuda:0").input_ids
    def logprob(model, pids, cont):
        cid = tok(cont, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
        with torch.no_grad(): lg = model(torch.cat([pids, cid], 1)).logits
        lp = torch.log_softmax(lg[0, pids.shape[1]-1:-1].float(), -1)
        return lp.gather(1, cid[0].unsqueeze(1)).sum().item()
    def gen(model, pids):
        with torch.no_grad():
            g = model.generate(pids, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        return tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True)

    # ---- Pass1 on INSTRUCT: find sick (LE+/SR-) examples ----
    samples = eval_drllm.load_task(args.task, "short", 0, -1)[: args.limit]
    sick = []; t0 = time.time()
    for i, s in enumerate(samples):
        pids = chat_ids(s["input"]); gold = s["correct"]; n = len(s["choices"])
        lp = {L: logprob(I, pids, f"Answer: {L}") for L in letters(n)}
        le = max(lp, key=lp.get) == gold
        sr = parse_letter(gen(I, pids), n) == gold
        if le and not sr:
            sick.append({"input": s["input"], "gold": gold, "n": n})
        if len(sick) >= 35: break
        if i % 40 == 0: print(f"p1 {i} sick={len(sick)} {int(time.time()-t0)}s", flush=True)
    if len(sick) < 5:
        json.dump({"error": "too few sick", "n_sick": len(sick)}, open(args.out, "w")); print("FEW SICK", len(sick)); return

    def measure(model, zero=False):
        c = cor = 0
        for it in sick:
            pids = chat_ids(it["input"]); Z["on"] = zero
            g = gen(model, pids); Z["on"] = False
            pl = parse_letter(g, it["n"]); c += int(pl != "?"); cor += int(pl == it["gold"])
        m = len(sick)
        return {"commit": round(c/m, 3), "correct": round(cor/m, 3), "n": m}

    out = {"config": {"instruct": args.instruct, "base": args.base, "layer": args.layer,
                      "task": args.task, "n_sick": len(sick)}}
    out["I"] = measure(I)
    out["I_zeroMLP"] = measure(I, zero=True)
    Imlp.load_state_dict(B_state); out["I_with_Bmlp"] = measure(I); Imlp.load_state_dict(I_state)  # transplant base->instruct
    out["B"] = measure(B)
    Bmlp.load_state_dict(I_state); out["B_with_Imlp"] = measure(B); Bmlp.load_state_dict(B_state)  # transplant instruct->base

    # ---- alignment weight-delta on down_proj ----
    with torch.no_grad():
        dW = (I_state["down_proj.weight"].float() - B_state["down_proj.weight"].float())  # [d_model, d_inter]
        col = dW.norm(dim=0)                                # per-channel alignment change
        order = torch.argsort(col, descending=True)
        out["weight_delta"] = {"mean_col_norm_all": round(col.mean().item(), 4),
                               "mean_col_norm_top128": round(col[order[:128]].mean().item(), 4),
                               "ratio_top128_vs_all": round((col[order[:128]].mean() / col.mean()).item(), 3)}

    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n=== ALIGNMENT ORIGIN (n_sick={len(sick)}) commit/correct ===")
    for k in ("I", "I_zeroMLP", "I_with_Bmlp", "B", "B_with_Imlp"):
        print(f"  {k:14} commit={out[k]['commit']} correct={out[k]['correct']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
