"""W2-b: token-level layer-redundancy profile along real generation trajectories.

E1 showed 5.4x more redundancy variance across tokens (within a sample) than
across samples. This script builds the design map for token-level routing:
  1) lag-1 autocorrelation of per-layer redundancy across decode steps
     -> is "route step t with step t-1's signal" (zero-cost lagged routing) viable?
  2) hard vs easy tokens (output margin quartiles) x per-layer redundancy
     -> where do hard tokens need MORE compute (repeat candidates)?
  3) per-token skippable-layer count at various cos thresholds
     -> FLOPs-saving potential of token-level skipping.

Stock model, greedy CoT generation (cache on); hidden states come per-step from
generate(output_hidden_states=True), i.e. exactly the current token's layer
stack — the native per-token view.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from eval_drllm import load_task, cot_math  # noqa: F401  (cot template used via load_task)

L = 28


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", default="dart_pool", choices=["dart", "dart_pool"])
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    import transformers
    print("transformers from:", transformers.__file__)
    from transformers import LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map={"": "cuda:0"})
    model.eval()
    assert not hasattr(model.model, "routers")

    # CoT protocol: realistic trajectories with actual reasoning tokens
    samples = load_task(args.task, protocol="cot",
                        subset_per_level=max(1, args.n_samples // 5) if args.task == "dart" else 0)
    if args.task == "dart_pool":
        by = {}
        for s in samples:
            by.setdefault(s["level"], []).append(s)
        samples = [s for lv in sorted(by) for s in by[lv][: max(1, args.n_samples // 5)]]
        # dart_pool prompts come from load_task in SHORT template; rebuild as CoT
        for s in samples:
            q = s["input"].split("Question: ", 1)[1].rsplit("\nThe final answer", 1)[0]
            s["input"] = cot_math.format(question=q)
    samples = samples[: args.n_samples]
    print(f"{args.task} CoT trajectories: n={len(samples)} max_new={args.max_new}")

    all_cos, all_margin, all_entropy, meta = [], [], [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt = tok.apply_chat_template([{"role": "user", "content": s["input"]}],
                                         tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new, do_sample=False,
                temperature=None, top_p=None, pad_token_id=tok.eos_token_id,
                output_hidden_states=True, output_scores=True,
                return_dict_in_generate=True, use_cache=True)
        # out.hidden_states: tuple(steps) of tuple(29) of [1, k, D] (k=prompt len at step0, else 1)
        cos_traj, mar_traj, ent_traj = [], [], []
        for st, (hs, sc) in enumerate(zip(out.hidden_states, out.scores)):
            stack = torch.stack([h[0, -1] for h in hs]).float()      # [29, D] current token
            cos = F.cosine_similarity(stack[:-1], stack[1:], dim=-1)  # [28]
            probs = F.softmax(sc[0].float(), dim=-1)
            top2 = probs.topk(2).values
            cos_traj.append(cos.cpu().numpy())
            mar_traj.append(float(top2[0] - top2[1]))
            ent_traj.append(float(-(probs * (probs + 1e-12).log()).sum()))
        all_cos.append(np.stack(cos_traj))          # [T, 28]
        all_margin.append(np.array(mar_traj))
        all_entropy.append(np.array(ent_traj))
        meta.append({"level": s["level"], "steps": len(cos_traj)})
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(samples)} ({(time.time()-t0)/(i+1):.1f}s/sample)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out_dir, f"profile_{args.task}.npz"),
        cos=np.concatenate(all_cos), margin=np.concatenate(all_margin),
        entropy=np.concatenate(all_entropy),
        sample_id=np.concatenate([[j] * m["steps"] for j, m in enumerate(meta)]),
        level=np.concatenate([[m["level"]] * m["steps"] for m in meta]))

    # ---- analysis ----
    rep = {"task": args.task, "n_samples": len(samples),
           "total_tokens": int(sum(m["steps"] for m in meta))}
    # 1) lag-1 autocorrelation per layer (within trajectories, then averaged)
    acs = []
    for c in all_cos:
        if len(c) < 8:
            continue
        x0, x1 = c[:-1], c[1:]
        num = ((x0 - x0.mean(0)) * (x1 - x1.mean(0))).mean(0)
        den = x0.std(0) * x1.std(0) + 1e-9
        acs.append(num / den)
    ac = np.mean(acs, axis=0)
    rep["lag1_autocorr_mean"] = float(ac.mean())
    rep["lag1_autocorr_per_layer"] = [round(float(v), 3) for v in ac]
    # 2) hard vs easy tokens: margin quartiles x mean redundancy
    M = np.concatenate(all_margin)
    C = np.concatenate(all_cos)
    q1, q3 = np.quantile(M, 0.25), np.quantile(M, 0.75)
    hard, easy = C[M <= q1], C[M >= q3]
    rep["redundancy_hard_minus_easy_per_layer"] = [
        round(float(v), 4) for v in (hard.mean(0) - easy.mean(0))]
    rep["mean_skippable@0.99_hard"] = float((hard > 0.99).sum(1).mean())
    rep["mean_skippable@0.99_easy"] = float((easy > 0.99).sum(1).mean())
    # 3) skippable layers per token at thresholds
    for th in [0.98, 0.99, 0.995]:
        k = (C > th).sum(1)
        rep[f"skippable@{th}_mean"] = float(k.mean())
        rep[f"skippable@{th}_p10_p90"] = [float(np.quantile(k, 0.1)), float(np.quantile(k, 0.9))]
    with open(os.path.join(args.out_dir, f"profile_{args.task}_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("REPORT", json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
