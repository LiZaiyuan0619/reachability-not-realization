"""W4-c: leave-one-out (true intervention) per-layer causal contribution.

First-order signals (magnitude AND direction) failed to predict oracle
consensus (z-AUC ~0.5). This is the interventional upgrade: actually skip
layer l (layer_indices mechanism, data_models drop-in) and measure the effect
on the model's output at the last prompt token:
    delta_gold_l = logp(gold | full) - logp(gold | skip l)
        < 0  => layer l HURTS this sample (skipping helps) — per-sample harmful-layer detector
    flip_l = does the argmax token change?
Compared against oracle-v2 consensus skip marginals. If even true intervention
cannot predict the consensus, the consensus itself is noise (re-roll
hypothesis); if it can, LOO becomes search-free supervision source #2.
ARC only for gold tokens (single-letter answers tokenize cleanly).
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from eval_drllm import load_task

L = 28


def rank_auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    for v, c in Counter(allv.tolist()).items():
        if c > 1:
            m = allv == v
            ranks[m] = ranks[m].mean()
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def dec_of(pl):
    c = Counter(pl)
    return [min(c.get(l, 0), 2) for l in range(L)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", required=True, choices=["arc_challenge", "arc_easy"])
    ap.add_argument("--oracle_file", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    import transformers
    print("transformers from:", transformers.__file__)
    from transformers import LlamaForCausalLM

    samples = load_task(args.task)[: args.limit]
    orc = json.load(open(args.oracle_file))["records"]
    assert len(orc) == len(samples)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map={"": "cuda:0"})
    model.eval()
    assert hasattr(model.model, "layer_indices"), "needs data_models drop-in"

    N = len(samples)
    d_gold = np.zeros((N, L), np.float32)
    d_top = np.zeros((N, L), np.float32)
    flip = np.zeros((N, L), np.float32)
    t0 = time.time()
    full = list(range(L))
    for i, s in enumerate(samples):
        prompt = tok.apply_chat_template([{"role": "user", "content": s["input"]}],
                                         tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        # gold = the correct letter; data-gen protocol answers start "Answer: X",
        # so measure at the position of X given forced prefix "Answer: "
        pre = tok("Answer: ", add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        inp = torch.cat([ids.input_ids, pre], 1)
        gold_id = tok(" " + s["correct"], add_special_tokens=False).input_ids[-1]
        gold_id2 = tok(s["correct"], add_special_tokens=False).input_ids[-1]
        with torch.no_grad():
            model.model.layer_indices = full
            lp_full = F.log_softmax(model(inp).logits[0, -1].float(), -1)
            top_full = int(lp_full.argmax())
            g = gold_id if lp_full[gold_id] > lp_full[gold_id2] else gold_id2
            for l in range(L):
                model.model.layer_indices = [x for x in full if x != l]
                lp = F.log_softmax(model(inp).logits[0, -1].float(), -1)
                d_gold[i, l] = (lp_full[g] - lp[g]).item()
                d_top[i, l] = (lp_full[top_full] - lp[top_full]).item()
                flip[i, l] = float(int(lp.argmax()) != top_full)
        model.model.layer_indices = full
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{N} ({(time.time()-t0)/(i+1):.1f}s/sample)", flush=True)

    # labels: consensus skip marginal + rescue-path skip indicator
    marg = np.full((N, L), np.nan, np.float32)
    resc = np.zeros((N, L), np.float32)
    for i, r in enumerate(orc):
        good = [p for p in r["paths"] if p["r"] > 0.5]
        if len(good) >= 3:
            marg[i] = np.mean([[dec_of(p["layers"])[l] == 0 for l in range(L)] for p in good], 0)
        if r["orig"] == 0 and good:
            best = min(good, key=lambda p: len(p["layers"]))
            d = dec_of(best["layers"])
            resc[i] = [d[l] == 0 for l in range(L)]

    rep = {"task": args.task, "n": N,
           "harmful_cells_frac": float((d_gold < 0).mean()),
           "flip_rate": float(flip.mean())}
    have = ~np.isnan(marg[:, 0])
    if have.sum() >= 20:
        m, v = marg[have], -d_gold[have]  # higher v = skipping helps gold more
        hi = m >= np.quantile(m, 0.9)
        lo = m <= np.quantile(m, 0.5)
        z = (v - v.mean(0)) / (v.std(0) + 1e-9)
        rep["auc_raw[-d_gold->consensus]"] = rank_auc(v[hi], v[lo])
        rep["auc_z[-d_gold->consensus]"] = rank_auc(z[hi], z[lo])
    rm = resc.sum(1) > 0
    if rm.sum() >= 10:
        v = -d_gold[rm]
        z = (v - v.mean(0)) / (v.std(0) + 1e-9)
        lab = resc[rm] > 0
        rep["auc_raw[-d_gold->rescue_skips]"] = rank_auc(v[lab], v[~lab])
        rep["auc_z[-d_gold->rescue_skips]"] = rank_auc(z[lab], z[~lab])
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, f"loo_{args.task}.npz"),
                        d_gold=d_gold, d_top=d_top, flip=flip, marginal=marg, rescue=resc)
    with open(os.path.join(args.out_dir, f"loo_{args.task}_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("REPORT", json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
