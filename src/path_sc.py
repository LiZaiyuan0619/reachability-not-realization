"""W6: path-perturbation self-consistency vs sampling self-consistency.

Null controls showed random layer-path edits produce MORE answer diversity
that hits the truth than temperature sampling at equal budget (arc: 99.7 vs
90.7 oracle). That was with a gold verifier. The deployable question: does the
advantage survive MAJORITY VOTING (no verifier)?

Arms, same N budget per sample:
  path-SC : N generations, greedy, each under a random 1-3-edit layer path
            (full path included as the first vote).
  samp-SC : N generations, full path, temperature sampling (first vote greedy).
Vote: normalized answer string majority (ties -> first occurrence). Same
normalization for both arms; math-equivalence vote splitting hurts both arms
equally.

Reports acc vs N in {1, 4, 8, 16} per task. If path-SC > samp-SC at equal N,
layer-path perturbation is a new test-time-compute axis. Requires data_models
drop-in (layer_indices).

W7 v2 additions (after W6 verdict: random-edit voting collapses on free-form):
  --edit_mode harmful_band : perturb only within a known-beneficial skip band
      (3B arc {20-23}: every non-empty subset skip, 15 subsets + full = 16
      votes). Diversity from perturbations that individually HELP, not hurt.
  --edit_mode repeat_only  : repeat-edit perturbations (repeats never remove
      information; null showed repeatonly oracle 87.7 > full-edit 77.3).
  confidence arm (always on): per generation, mean transition logprob; report
      {arm}_conf@N = answer of the max-confidence generation among first N.
      Verifier-free BoN-by-confidence — can it cash the reroll ceiling where
      majority voting could not? NOTE samp-arm scores are post-temperature.
"""
import argparse
import json
import os
import random
import re
import time
from collections import Counter
from itertools import combinations

import torch
from mathruler.grader import extract_boxed_content, grade_answer

from eval_drllm import load_task, ARC_RE
from null_oracle import random_edit_path
from data_generation import LayerPath, get_is_instruct

L = 28
NS = [1, 4, 8, 16]


def harmful_band_paths(band, nl=L):
    """All non-empty skip-subsets of the band, small subsets first."""
    subs = [c for r in range(1, len(band) + 1) for c in combinations(band, r)]
    return [[l for l in range(nl) if l not in set(c)] for c in subs]


def repeat_edit_path(rng, n_edits, nl=L):
    path = list(range(nl))
    for _ in range(n_edits):
        s = rng.randrange(0, len(path))
        if len(path) + 1 <= 2 * nl:
            path = path[: s + 1] + [path[s]] + path[s + 1:]
    return LayerPath(path)


def norm_answer(task, resp):
    if task.startswith("dart"):
        return (extract_boxed_content(resp.strip()) or "").strip()
    m = ARC_RE.match(resp.strip())
    return m.group(1).upper() if m else ""


def final_grade(task, ans, correct):
    if not ans:
        return 0.0
    if task.startswith("dart"):
        return float(grade_answer(ans, correct))
    return float(ans.lower() == correct.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", required=True,
                    choices=["arc_challenge", "arc_easy", "dart", "dart_pool"])
    ap.add_argument("--per_level", type=int, default=100)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--offset", type=int, default=0,
                    help="arc tasks: skip first K samples (0-299=dev, 300+=clean test)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--edit_mode", choices=["random", "harmful_band", "repeat_only"],
                    default="random")
    ap.add_argument("--band", default="20,21,22,23",
                    help="harmful_band mode: comma-separated layer ids")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    import transformers
    print("transformers from:", transformers.__file__)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.task.startswith("dart"):
        samples = load_task(args.task, "short",
                            args.per_level if args.task == "dart" else 0)
        if args.task == "dart_pool":
            by = {}
            for s in samples:
                by.setdefault(s["level"], []).append(s)
            samples = [s for lv in sorted(by) for s in by[lv][: args.per_level]]
    else:
        samples = load_task(args.task)[args.offset: args.offset + args.limit]
    print(f"path-SC vs samp-SC: {args.task} n={len(samples)} offset={args.offset} N={NS}")

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map={"": "cuda:0"})
    model.eval()
    assert hasattr(model.model, "layer_indices"), "needs data_models drop-in"
    max_new = 15 if args.task.startswith("dart") else 10
    nmax = max(NS)
    rng = random.Random(0)
    nl = model.config.num_hidden_layers  # NOT module L=28: 8B=32, qwen3B=36
    full = list(range(nl))

    def gen(ids, path, sample_T=None):
        """-> (text, mean transition logprob of the generated tokens)"""
        model.model.layer_indices = path
        dup = len(path) != len(set(path))
        kw = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id,
                  use_cache=not dup, return_dict_in_generate=True, output_scores=True)
        if sample_T is None:
            kw.update(do_sample=False, temperature=None, top_p=None)
        else:
            kw.update(do_sample=True, temperature=sample_T, top_p=0.95)
        with torch.no_grad():
            out = model.generate(**ids, **kw)
        n_in = ids["input_ids"].shape[1]
        text = tok.decode(out.sequences[0][n_in:], skip_special_tokens=True).strip()
        ts = model.compute_transition_scores(out.sequences, out.scores,
                                             normalize_logits=True)[0]
        conf = ts[torch.isfinite(ts)].mean().item() if torch.isfinite(ts).any() else -1e9
        return text, conf

    band = [int(x) for x in args.band.split(",")]
    band_paths = harmful_band_paths(band, nl)

    def perturbed_path(k):
        if args.edit_mode == "harmful_band":
            return band_paths[(k - 1) % len(band_paths)]
        if args.edit_mode == "repeat_only":
            return repeat_edit_path(rng, rng.choice([1, 2, 3]), nl).layers
        return random_edit_path(rng, rng.choice([1, 2, 3]), nl).layers

    recs = []
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt = tok.apply_chat_template([{"role": "user", "content": s["input"]}],
                                         tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        path_votes, samp_votes = [], []
        for k in range(nmax):
            p = full if k == 0 else perturbed_path(k)
            t, c = gen(ids, p)
            path_votes.append((norm_answer(args.task, t), c))
            t, c = gen(ids, full, None if k == 0 else args.temperature)
            samp_votes.append((norm_answer(args.task, t), c))
        rec = {"level": s.get("level")}
        for N in NS:
            for arm, votes in [("path", path_votes), ("samp", samp_votes)]:
                vs = [v for v, _ in votes[:N] if v]
                top = Counter(vs).most_common(1)[0][0] if vs else ""
                rec[f"{arm}@{N}"] = final_grade(args.task, top, s["correct"])
                cvs = [(v, c) for v, c in votes[:N] if v]
                best = max(cvs, key=lambda x: x[1])[0] if cvs else ""
                rec[f"{arm}_conf@{N}"] = final_grade(args.task, best, s["correct"])
        recs.append(rec)
        if (i + 1) % 25 == 0:
            msg = " ".join(f"{a}@{N}={sum(r[f'{a}@{N}'] for r in recs)/len(recs):.3f}"
                           for N in [1, 16] for a in ["path", "samp"])
            print(f"{i+1}/{len(samples)} {msg} ({(time.time()-t0)/(i+1):.1f}s/sample)", flush=True)

    n = len(recs)
    summary = {"task": args.task, "n": n, "temperature": args.temperature,
               "edit_mode": args.edit_mode,
               "band": args.band if args.edit_mode == "harmful_band" else None}
    for N in NS:
        for arm in ["path", "samp"]:
            summary[f"{arm}@{N}"] = sum(r[f"{arm}@{N}"] for r in recs) / n
            summary[f"{arm}_conf@{N}"] = sum(r[f"{arm}_conf@{N}"] for r in recs) / n
    os.makedirs(args.out_dir, exist_ok=True)
    tag = "" if args.edit_mode == "random" else f"_{args.edit_mode}"
    if args.offset:
        tag += f"_off{args.offset}"
        summary["offset"] = args.offset
    out = os.path.join(args.out_dir, f"pathsc_{args.task}{tag}.json")
    json.dump({"summary": summary, "records": recs}, open(out, "w"), indent=1)
    print("SUMMARY", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
