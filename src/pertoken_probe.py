"""W8: per-token routing probes (zero-train measurements).

Sequence-level per-sample routing is dead (labels=noise, router correctly
learns all-execute, reroll explains the oracle). The remaining live signal is
per-token: E1 measured a 5.4x headroom for token-level decisions. These probes
measure whether that headroom is behaviorally real, before we invest in a
per-token router.

All probes use a hand-rolled decode loop: at every decode step the layer path
may change, so KV caches are invalid across steps -> full-sequence re-forward
per step (use_cache=False). Short-answer protocol only (<=15 new tokens).

Probes:
  perturb_diversity : pass@N + vote + conf for
                        per_seq  arm (one random path edit reused all steps;
                                      = W6 pathsc granularity)
                        per_step arm (fresh random edit every decode step)
                      If per_step ceiling >> per_seq ceiling, token granularity
                      unlocks diversity that sequence granularity cannot.
  entropy_gated     : easy steps (prev-step top1 prob >= tau) skip the given
                      band; hard steps run full. Zero-train attempt at a gain:
                      acc + saved-layer stats vs baseline, tau sweep.
  greedy_action     : every step, forward each action in {full, skip_band,
                      repeat_mid}; emit the token of the max-top1-prob action.
                      No-lookahead upper-ish bound for token-level
                      confidence routing. Reports acc + action usage.

Requires data_models drop-in (model.model.layer_indices).
"""
import argparse
import json
import os
import random
import time
from collections import Counter

import torch

from eval_drllm import load_task, ARC_RE
from null_oracle import random_edit_path
from path_sc import norm_answer, final_grade

L = 28


def build_samples(args):
    if args.task.startswith("dart"):
        samples = load_task(args.task, "short",
                            args.per_level if args.task == "dart" else 0)
        if args.task == "dart_pool":
            by = {}
            for s in samples:
                by.setdefault(s["level"], []).append(s)
            samples = [s for lv in sorted(by) for s in by[lv][: args.per_level]]
    else:
        samples = load_task(args.task)[: args.limit]
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", required=True,
                    choices=["arc_challenge", "arc_easy", "dart", "dart_pool"])
    ap.add_argument("--probe", required=True,
                    choices=["perturb_diversity", "entropy_gated", "greedy_action"])
    ap.add_argument("--per_level", type=int, default=30)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--budget", type=int, default=8, help="perturb_diversity: N per arm")
    ap.add_argument("--band", default="20,21,22,23")
    ap.add_argument("--taus", default="0.5,0.7,0.9", help="entropy_gated thresholds")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    import transformers
    print("transformers from:", transformers.__file__)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    samples = build_samples(args)
    print(f"pertoken_probe {args.probe}: {args.task} n={len(samples)}")

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map={"": "cuda:0"})
    model.eval()
    assert hasattr(model.model, "layer_indices"), "needs data_models drop-in"

    max_new = 15 if args.task.startswith("dart") else 10
    gc = getattr(model, "generation_config", None)
    eos_ids = gc.eos_token_id if gc and gc.eos_token_id is not None else tok.eos_token_id
    eos_set = set(eos_ids) if isinstance(eos_ids, (list, tuple)) else {eos_ids}
    nl = model.config.num_hidden_layers  # NOT module L=28: 8B=32, qwen3B=36
    full = list(range(nl))
    band = [int(x) for x in args.band.split(",")]
    skip_band = [l for l in full if l not in set(band)]
    mid = nl // 2
    repeat_mid = full[: mid + 1] + [mid] + full[mid + 1:]

    @torch.no_grad()
    def step_forward(cur_ids, path):
        """One full-sequence forward under `path`; -> last-position logits."""
        model.model.layer_indices = path
        out = model(input_ids=cur_ids, use_cache=False)
        return out.logits[0, -1]

    def decode(ids, path_fn):
        """path_fn(t, prev_probs) -> path for step t. Greedy decode.
        -> (text, per-step metadata list)"""
        cur = ids["input_ids"]
        prev_probs = None
        meta = []
        for t in range(max_new):
            path = path_fn(t, prev_probs)
            logits = step_forward(cur, path)
            probs = torch.softmax(logits.float(), -1)
            nxt = int(probs.argmax())
            meta.append({"len": len(path), "p1": float(probs.max())})
            prev_probs = probs
            cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], 1)
            if nxt in eos_set:
                break
        text = tok.decode(cur[0, ids["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        return text, meta

    def decode_greedy_action(ids, actions):
        """Every step: forward all actions, emit token of max-top1-prob one."""
        cur = ids["input_ids"]
        usage = Counter()
        for t in range(max_new):
            best = None
            for name, path in actions.items():
                logits = step_forward(cur, path)
                probs = torch.softmax(logits.float(), -1)
                p1, nxt = float(probs.max()), int(probs.argmax())
                if best is None or p1 > best[0]:
                    best = (p1, nxt, name)
            usage[best[2]] += 1
            cur = torch.cat([cur, torch.tensor([[best[1]]], device=cur.device)], 1)
            if best[1] in eos_set:
                break
        text = tok.decode(cur[0, ids["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        return text, usage

    rng = random.Random(0)
    recs = []
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt = tok.apply_chat_template([{"role": "user", "content": s["input"]}],
                                         tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        rec = {"level": s.get("level")}

        if args.probe == "perturb_diversity":
            for arm in ("per_seq", "per_step"):
                answers = []
                for k in range(args.budget):
                    if k == 0:
                        pf = lambda t, p: full
                    elif arm == "per_seq":
                        fixed = random_edit_path(rng, rng.choice([1, 2, 3]), nl).layers
                        pf = lambda t, p, _f=fixed: _f
                    else:
                        pf = lambda t, p: (full if t == 0 else
                                           random_edit_path(rng, rng.choice([1, 2, 3]), nl).layers)
                    text, _ = decode(ids, pf)
                    answers.append(norm_answer(args.task, text))
                graded = [final_grade(args.task, a, s["correct"]) for a in answers]
                rec[f"{arm}_pass"] = float(max(graded))
                vs = [a for a in answers if a]
                top = Counter(vs).most_common(1)[0][0] if vs else ""
                rec[f"{arm}_vote"] = final_grade(args.task, top, s["correct"])
                rec[f"{arm}_first"] = graded[0]

        elif args.probe == "entropy_gated":
            for tau in [float(x) for x in args.taus.split(",")]:
                def pf(t, p, _tau=tau):
                    if t == 0 or p is None:
                        return full
                    return skip_band if float(p.max()) >= _tau else full
                text, meta = decode(ids, pf)
                ans = norm_answer(args.task, text)
                rec[f"acc_tau{tau}"] = final_grade(args.task, ans, s["correct"])
                rec[f"layers_tau{tau}"] = sum(m["len"] for m in meta) / len(meta)
            text, meta = decode(ids, lambda t, p: full)
            rec["acc_full"] = final_grade(
                args.task, norm_answer(args.task, text), s["correct"])

        else:  # greedy_action
            actions = {"full": full, "skip_band": skip_band, "repeat_mid": repeat_mid}
            text, usage = decode_greedy_action(ids, actions)
            rec["acc_greedy"] = final_grade(
                args.task, norm_answer(args.task, text), s["correct"])
            for k, v in usage.items():
                rec[f"use_{k}"] = v
            text, _ = decode(ids, lambda t, p: full)
            rec["acc_full"] = final_grade(
                args.task, norm_answer(args.task, text), s["correct"])

        recs.append(rec)
        if (i + 1) % 10 == 0:
            keys = [k for k in recs[0] if k.startswith(("per_", "acc"))]
            msg = " ".join(f"{k}={sum(r.get(k, 0) for r in recs)/len(recs):.3f}"
                           for k in keys[:6])
            print(f"{i+1}/{len(samples)} {msg} ({(time.time()-t0)/(i+1):.1f}s/sample)",
                  flush=True)

    n = len(recs)
    summary = {"task": args.task, "probe": args.probe, "n": n,
               "budget": args.budget, "band": args.band}
    num_keys = sorted({k for r in recs for k in r
                       if isinstance(r[k], (int, float)) and k != "level"})
    for k in num_keys:
        vals = [r[k] for r in recs if k in r]
        summary[k] = sum(vals) / len(vals)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"pertoken_{args.probe}_{args.task}.json")
    json.dump({"summary": summary, "records": recs}, open(out, "w"), indent=1)
    print("SUMMARY", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
