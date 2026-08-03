"""E1: intrinsic per-layer signals from the STOCK model vs MCTS labels (4001 samples).

Question: is the routing supervision Dr.LLM buys with MCTS already present, for
free, in the model's own residual stream? For every training sample we run ONE
stock forward on the prompt and extract per-layer signals, then measure how well
they predict the MCTS skip/exec/repeat labels and the per-sample "difficulty"
(all-exec vs any-skip). Also probes the per-token routing headroom (direction B):
if within-sample per-token variance of layer redundancy >> between-sample
variance, token-level routing has much more room than sequence-level.

Signals per layer l (h_in = hs[l], h_out = hs[l+1], prompt tokens only):
  cos_mean   mean_t cos(h_in[t], h_out[t])      ~ layer redundancy (high = skippable)
  relnorm    mean_t |h_out-h_in|/|h_in|         ~ update magnitude
  cos_last   cos at last prompt token
  ll_kl      logit-lens KL(p_l || p_{l+1}) at last token   ~ local convergence
  ll_agree   argmax p_l == argmax p_final                  ~ already-decided depth
  tok_std    std_t of per-token cos              ~ per-token routing headroom

Outputs: signals.npz + report.json (AUCs etc.) in --out_dir.
"""
import argparse
import glob
import json
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

L = 28


def rank_auc(pos, neg):
    """AUC via rank statistic, no sklearn."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    for v, c in Counter(allv.tolist()).items():
        if c > 1:
            m = allv == v
            ranks[m] = ranks[m].mean()
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_glob", default="data/train/*.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import transformers
    print("transformers from:", transformers.__file__)
    from transformers import LlamaForCausalLM

    # ---- load samples: prompt text + decisions from best_path ----
    samples = []
    for f in sorted(glob.glob(args.data_glob)):
        obj = json.load(open(f))
        for q, bp in zip(obj["questions"], obj["best_path"]):
            cnt = Counter(bp)
            dec = [min(cnt.get(l, 0), 2) for l in range(L)]
            samples.append({"q": q, "dec": dec, "dataset": obj["dataset"]})
    if args.limit:
        samples = samples[: args.limit]
    print("samples:", len(samples))

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map={"": "cuda:0"})
    model.eval()
    assert not hasattr(model.model, "routers"), "signals must come from STOCK model"
    lm_head, final_norm = model.lm_head, model.model.norm

    N = len(samples)
    sig = {k: np.zeros((N, L), np.float32)
           for k in ["cos_mean", "relnorm", "cos_last", "ll_kl", "ll_agree", "tok_std"]}
    labels = np.zeros((N, L), np.int8)
    dsets = []

    t0 = time.time()
    for i, s in enumerate(samples):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": s["q"]}], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.model(**ids, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states  # 29 x [1, T, D]
            # logit-lens at last token for all layer outputs
            last = torch.stack([h[0, -1] for h in hs])          # [29, D]
            logits = lm_head(final_norm(last)).float()          # [29, V]
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            final_top1 = int(logits[-1].argmax())
            for l in range(L):
                hi, ho = hs[l][0].float(), hs[l + 1][0].float()  # [T, D]
                cos_t = F.cosine_similarity(hi, ho, dim=-1)      # [T]
                sig["cos_mean"][i, l] = cos_t.mean()
                sig["tok_std"][i, l] = cos_t.std()
                sig["cos_last"][i, l] = cos_t[-1]
                sig["relnorm"][i, l] = ((ho - hi).norm(dim=-1) / (hi.norm(dim=-1) + 1e-6)).mean()
                # KL(p_l || p_{l+1}) : how much does this layer still change the belief
                sig["ll_kl"][i, l] = (p[l] * (logp[l] - logp[l + 1])).sum()
                sig["ll_agree"][i, l] = float(int(logits[l].argmax()) == final_top1)
        labels[i] = s["dec"]
        dsets.append(s["dataset"])
        if (i + 1) % 250 == 0:
            print(f"{i+1}/{N} ({(time.time()-t0)/(i+1):.3f}s/sample)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, "signals.npz"),
                        labels=labels, dsets=np.array(dsets), **sig)

    # ---- report ----
    rep = {}
    skip, rept, execm = labels == 0, labels == 2, labels == 1
    # pooled AUC: signal -> skip (vs exec) and -> repeat (vs exec)
    for k in ["cos_mean", "cos_last", "relnorm", "ll_kl", "ll_agree"]:
        v = sig[k]
        rep[f"auc_skip[{k}]"] = rank_auc(v[skip], v[execm])
        rep[f"auc_repeat[{k}]"] = rank_auc(v[rept], v[execm])
    # per-layer-normalized (z-score within layer) pooled AUC — controls for the
    # trivial "which layer index" effect, isolating input-dependence
    for k in ["cos_mean", "ll_kl"]:
        v = sig[k]
        z = (v - v.mean(0)) / (v.std(0) + 1e-9)
        rep[f"aucZ_skip[{k}]"] = rank_auc(z[skip], z[execm])
        rep[f"aucZ_repeat[{k}]"] = rank_auc(z[rept], z[execm])
    # difficulty probe: does total redundancy predict the all-exec label?
    allexec = (labels != 1).sum(1) == 0
    red = sig["cos_mean"].mean(1)
    rep["auc_allexec[mean_cos]"] = rank_auc(red[~allexec], red[allexec])  # redundant -> has skips
    # per-token headroom (direction B): within-sample tok std vs between-sample std
    rep["tok_std_within_mean"] = float(sig["tok_std"].mean())
    rep["between_sample_std"] = float(sig["cos_mean"].std(0).mean())
    rep["headroom_ratio_tok_over_sample"] = rep["tok_std_within_mean"] / (rep["between_sample_std"] + 1e-9)
    rep["n"] = N
    rep["label_frac"] = {"skip": float(skip.mean()), "exec": float(execm.mean()),
                         "repeat": float(rept.mean())}
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("REPORT", json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
