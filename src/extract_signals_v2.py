"""W4-b: DIRECTIONAL layer-contribution signals vs oracle CONSENSUS labels.

E1 tested magnitude-style signals (cos, |dh|) against single-best-path MCTS
labels and found ~noise. Two upgrades here:
  * better labels: oracle-v2 per-sample consensus skip-marginals (all correct
    paths of 200 full-sweep sims), not one noisy first-hit path;
  * better signals (direction, not just size — "the Delta is what matters"):
      align_final  cos(h_{l+1}-h_l, h_final-h_l)   is the layer moving TOWARD
                                                   the final state?
      margin_delta logit-lens top1-margin change   does the layer make the
                                                   answer MORE decided?
      rot          angle on the normed sphere      RMSNorm-semantics rotation
      cos/relnorm  (E1 baselines for comparison)

Extracts signals for the SAME samples oracle_v2 used (same load_task order),
joins with results/w3_oracle2/oraclev2_<task>.json, reports AUCs.
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


def dec_of(path_layers):
    c = Counter(path_layers)
    return [min(c.get(l, 0), 2) for l in range(L)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", required=True, choices=["arc_challenge", "arc_easy", "dart", "dart_pool"])
    ap.add_argument("--oracle_file", required=True)
    ap.add_argument("--per_level", type=int, default=60)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    import transformers
    print("transformers from:", transformers.__file__)
    from transformers import LlamaForCausalLM

    # same sample selection as oracle_v2
    if args.task.startswith("dart"):
        samples = load_task(args.task, "short", args.per_level if args.task == "dart" else 0)
        if args.task == "dart_pool":
            by = {}
            for s in samples:
                by.setdefault(s["level"], []).append(s)
            samples = [s for lv in sorted(by) for s in by[lv][: args.per_level]]
    else:
        samples = load_task(args.task)[: args.limit]

    orc = json.load(open(args.oracle_file))["records"]
    assert len(orc) == len(samples), f"oracle/sample mismatch {len(orc)} vs {len(samples)}"

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map={"": "cuda:0"})
    model.eval()
    assert not hasattr(model.model, "routers")

    N = len(samples)
    keys = ["cos", "relnorm", "align_final", "margin_delta", "rot"]
    sig = {k: np.zeros((N, L), np.float32) for k in keys}
    lm_head, norm = model.lm_head, model.model.norm

    t0 = time.time()
    for i, s in enumerate(samples):
        prompt = tok.apply_chat_template([{"role": "user", "content": s["input"]}],
                                         tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.model(**ids, output_hidden_states=True, use_cache=False)
            hs = [h[0, -1].float() for h in out.hidden_states]      # 29 x [D] last token
            h_final = hs[-1]
            logits = lm_head(norm(torch.stack(hs)).to(model.dtype)).float()  # [29, V]
            probs = F.softmax(logits, dim=-1)
            top2 = probs.topk(2, dim=-1).values                     # [29, 2]
            margins = (top2[:, 0] - top2[:, 1])
            for l in range(L):
                hi, ho = hs[l], hs[l + 1]
                d = ho - hi
                sig["cos"][i, l] = F.cosine_similarity(hi, ho, dim=0)
                sig["relnorm"][i, l] = d.norm() / (hi.norm() + 1e-6)
                tgt = h_final - hi
                sig["align_final"][i, l] = F.cosine_similarity(d, tgt, dim=0) if tgt.norm() > 1e-4 else 0.0
                sig["margin_delta"][i, l] = margins[l + 1] - margins[l]
                nh, no = hi / hi.norm(), ho / ho.norm()
                sig["rot"][i, l] = torch.arccos(torch.clamp((nh * no).sum(), -1, 1))
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{N} ({(time.time()-t0)/(i+1):.2f}s/sample)", flush=True)

    # ---- labels from oracle consensus ----
    # skip-marginal per (sample, layer) across that sample's correct paths
    marg = np.full((N, L), np.nan, np.float32)
    for i, r in enumerate(orc):
        good = [p for p in r["paths"] if p["r"] > 0.5]
        if len(good) >= 3:
            marg[i] = np.mean([[dec_of(p["layers"])[l] == 0 for l in range(L)] for p in good], 0)
    have = ~np.isnan(marg[:, 0])
    print(f"samples with consensus labels (>=3 correct paths): {int(have.sum())}/{N}")

    rep = {"task": args.task, "n": N, "n_consensus": int(have.sum())}
    if have.sum() >= 20:
        m = marg[have]
        hi_lab = m >= np.quantile(m, 0.9)     # top-decile "consensus skippable" cells
        lo_lab = m <= np.quantile(m, 0.5)
        for k in keys:
            v = sig[k][have]
            z = (v - v.mean(0)) / (v.std(0) + 1e-9)   # within-layer z: input-dependence only
            rep[f"auc_raw[{k}]"] = rank_auc(v[hi_lab], v[lo_lab])
            rep[f"auc_z[{k}]"] = rank_auc(z[hi_lab], z[lo_lab])
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, f"signals_v2_{args.task}.npz"),
                        marginal=marg, **sig)
    with open(os.path.join(args.out_dir, f"signals_v2_{args.task}_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("REPORT", json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
