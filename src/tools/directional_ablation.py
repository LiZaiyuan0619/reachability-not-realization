"""Directional surgery on the L20-MLP write (the last algorithmic increment).

zero/scale silences the WHOLE MLP write (off300 ref: zero=0.783, baseline=0.680,
skip-layer=0.787, CoT ceiling=0.813). If the toxicity lives in a low-rank
subspace, projecting out top-k directions should match/beat zeroing while
keeping the healthy part of the write. Directions are estimated ONLY on the dev
segment (first 300 items); evaluation is on offset 300-599.

Direction candidates:
  meandiff : normalize( mean(MLPout | KB) - mean(MLPout | KA) )   (supervised by
             the dev LE x SR grid computed in-process: KB = LE+/SR-, KA = LE+/SR+)
  pca<k>   : top-k PCA of centered dev MLP outputs (unsupervised)

Outputs <out_dir>/dsurg_<config>.json per config with the same summary schema
as eval_drllm (acc_micro/acc_lenient) so harvesting stays uniform.
"""
import argparse
import json
import os

import torch

from eval_drllm import load_task, grade, grade_lenient

LAYER = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--configs", default="meandiff,pca1,pca4,pca8")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map={"": "cuda:0"})
    model.eval()
    mlp = model.model.layers[LAYER].mlp

    samples = load_task("arc_challenge")
    dev, ev = samples[:300], samples[300:600]

    def prompt_of(s):
        return tok.apply_chat_template([{"role": "user", "content": s["input"]}],
                                       tokenize=False, add_generation_prompt=True)

    # ---- pass 1: dev grid (LE via letter logprob, SR via generation) + capture ----
    cap = {}
    h = mlp.register_forward_hook(lambda m, i, o: cap.__setitem__("v", o[0, -1].float()))
    vecs, LE, SR = [], [], []
    for s in dev:
        ids = tok(prompt_of(s), return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**ids).logits
        vecs.append(cap["v"].cpu())
        n_opt = len(s["choices"])
        lps = []
        for j in range(n_opt):
            cont = tok(f"Answer: {chr(65+j)}", return_tensors="pt",
                       add_special_tokens=False).input_ids.to(model.device)
            full = torch.cat([ids.input_ids, cont], dim=1)
            with torch.no_grad():
                lg = model(full).logits
            lp = torch.log_softmax(lg[0, ids.input_ids.shape[1]-1:-1].float(), -1)
            lps.append(lp.gather(1, cont[0].unsqueeze(1)).sum().item())
        LE.append(chr(65 + max(range(n_opt), key=lambda j: lps[j])) == s["correct"])
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=10, do_sample=False,
                                 temperature=None, top_p=None,
                                 pad_token_id=tok.eos_token_id, use_cache=True)
        resp = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
        SR.append(grade_lenient("arc_challenge", resp, s["correct"]) == 1.0)
    h.remove()

    V = torch.stack(vecs)
    KB = [i for i in range(len(dev)) if LE[i] and not SR[i]]
    KA = [i for i in range(len(dev)) if LE[i] and SR[i]]
    print(f"dev grid: KB={len(KB)} KA={len(KA)} (LE acc={sum(LE)/len(LE):.3f} SR acc={sum(SR)/len(SR):.3f})")

    dirs_bank = {}
    d = (V[KB].mean(0) - V[KA].mean(0))
    dirs_bank["meandiff"] = torch.nn.functional.normalize(d, dim=0).unsqueeze(0)
    Vc = V - V.mean(0, keepdim=True)
    _, _, Vt = torch.pca_lowrank(Vc, q=16)
    for k in (1, 4, 8):
        dirs_bank[f"pca{k}"] = Vt[:, :k].T.contiguous()

    # ---- pass 2: evaluate each config on off300 ----
    os.makedirs(args.out_dir, exist_ok=True)
    for cfg in args.configs.split(","):
        D = dirs_bank[cfg].to(model.device)

        def proj_hook(m, i, o, D=D):
            of = o.float()
            of = of - (of @ D.T) @ D
            return of.to(o.dtype)
        hh = mlp.register_forward_hook(proj_hook)
        records = []
        for s in ev:
            ids = tok(prompt_of(s), return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=10, do_sample=False,
                                     temperature=None, top_p=None,
                                     pad_token_id=tok.eos_token_id, use_cache=True)
            resp = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
            records.append({"acc": grade("arc_challenge", resp, s["correct"]),
                            "acc_len": grade_lenient("arc_challenge", resp, s["correct"]),
                            "response": resp, "correct": s["correct"]})
        hh.remove()
        n = len(records)
        summary = {"task": "arc_challenge", "mode": f"dsurg_{cfg}", "n": n,
                   "acc_micro": sum(r["acc"] for r in records) / n,
                   "acc_lenient": sum(r["acc_len"] for r in records) / n,
                   "n_dirs": int(D.shape[0]), "offset": 300}
        path = os.path.join(args.out_dir, f"dsurg_{cfg}.json")
        json.dump({"summary": summary, "records": records}, open(path, "w"), indent=1)
        print("SUMMARY", cfg, json.dumps(summary))


if __name__ == "__main__":
    main()
