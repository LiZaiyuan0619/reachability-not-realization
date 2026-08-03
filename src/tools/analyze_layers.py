"""Per-layer diagnostics for the harmful-layer mechanism (W11): logit lens +
residual-stream I/O stats, base vs instruct comparable (same raw-text prompts).

For N ARC prompts, one forward with output_hidden_states=True; at the last prompt
token and each layer boundary i (hidden_states[i] = input of layer i):
  - logit lens: final_norm(h) @ lm_head -> top1 agreement with the model's own
    final prediction, entropy
  - residual delta: ||h_{i+1}-h_i|| / ||h_i||  (how much layer i writes)
  - cos(h_{i+1}, h_i)                          (how much it rotates the stream)
Anomalies at L20 that exist in instruct but not base localize the lesion.
Writes <out>/lens_<tag>.json with per-layer means.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_drllm import load_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map={"": "cuda:0"})
    model.eval()
    L = model.config.num_hidden_layers

    samples = load_task("arc_challenge")[: args.n]
    agree = [0.0] * (L + 1)
    entropy = [0.0] * (L + 1)
    delta = [0.0] * L
    cos = [0.0] * L

    for s in samples:
        prompt = s["input"] + "\nAnswer:"  # same raw-text prompt for base & instruct
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
            hs = [h[0, -1].float() for h in out.hidden_states]  # L+1 vectors
            final_top1 = out.logits[0, -1].argmax().item()
            for i, h in enumerate(hs):
                logits = model.lm_head(model.model.norm(h.to(model.dtype).unsqueeze(0)))[0]
                p = torch.softmax(logits.float(), -1)
                agree[i] += float(logits.argmax().item() == final_top1)
                entropy[i] += float(-(p * (p + 1e-12).log()).sum())
            for i in range(L):
                d = hs[i + 1] - hs[i]
                delta[i] += float(d.norm() / (hs[i].norm() + 1e-6))
                cos[i] += float(torch.nn.functional.cosine_similarity(
                    hs[i + 1], hs[i], dim=0))

    n = len(samples)
    res = {
        "tag": args.tag, "n": n, "num_layers": L,
        "lens_top1_agree": [a / n for a in agree],
        "lens_entropy": [e / n for e in entropy],
        "residual_delta_ratio": [d / n for d in delta],
        "residual_cos": [c / n for c in cos],
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"lens_{args.tag}.json")
    json.dump(res, open(path, "w"), indent=1)
    print("saved", path)
    for i in range(max(0, L - 14), L):
        print(f"L{i:2d} agree={res['lens_top1_agree'][i+1]:.3f} "
              f"H={res['lens_entropy'][i+1]:.2f} dR={res['residual_delta_ratio'][i]:.3f} "
              f"cos={res['residual_cos'][i]:.4f}")


if __name__ == "__main__":
    main()
