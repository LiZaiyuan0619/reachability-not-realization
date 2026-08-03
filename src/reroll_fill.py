"""Fill the reproduction-audit table (Table 1) cells that the original w4_null run
never covered: arc_easy (short) and dart (CoT protocol).

One consistent tool for both protocols and all four audit arms, so every cell in a
column is produced the same way:
  (d)  random-path verifier@budget   -- early-exit oracle over random layer edits
  (e)  best-of-N verifier@budget     -- early-exit oracle over full-path T-sampling
  (d') deployable path-vote@vote_k   -- majority over vote_k random layer edits
  (e') deployable samp-vote@vote_k   -- majority over vote_k full-path samples

short mode reuses null_oracle.py's random_edit_path + early-exit logic and greedy
decode; cot mode mirrors oracle_cot.py (cot_math prompt, 512-token boxed decode).

VALIDATE before trusting new cells: run --task arc_challenge --protocol short and
check it reproduces the logged w4_null cell (orig .66 / randpath .997 / bestofn .907).
Only then are arc_easy (short) and dart (cot) trustworthy.

Model is the same drop-in MCTSModel used for every existing result (default
Llama-3.2-3B-Instruct); needs the original code dir + env so layer_indices works.
"""
import argparse
import json
import os
import random
import re
import time
from collections import Counter

import torch
from mathruler.grader import extract_boxed_content, grade_answer

from data_generation import LayerPath, MCTSModel
from eval_drllm import cot_math, load_task

ARC_RE = re.compile(r"^(?:Answer:\s*)?([A-Da-d])\.?$")           # short: strict, == evaluate_path
ARC_COT_RE = re.compile(r"Answer:\s*\(?([A-Da-d])\)?")            # cot: last "Answer: X"


def random_edit_path(rng, n_edits, nl, skip_only=False):
    """null_oracle.py parity when skip_only=False: 1-2 contiguous skips (60%) or one
    repeat (40%). skip_only=True forces skips, so every path is a strict subsequence
    (no repeated layer) and generate_with_path keeps the KV cache on -- the only way
    reroll is affordable under the CoT protocol's long decodes."""
    path = list(range(nl))
    for _ in range(n_edits):
        op = rng.random()
        if skip_only or (op < 0.6 and len(path) > 3):
            if len(path) <= 3:
                continue
            k = rng.choice([1, 2])
            s = rng.randrange(0, len(path) - k + 1)
            path = path[:s] + path[s + k:]
        else:
            s = rng.randrange(0, len(path))
            if len(path) + 1 <= 2 * nl:
                path = path[: s + 1] + [path[s]] + path[s + 1:]
    return LayerPath(path)


def extract_pred(task, protocol, raw):
    """Normalized answer string ('' if unparseable). Matches evaluate_path / grade."""
    raw = (raw or "").strip()
    if task.startswith("dart"):
        return extract_boxed_content(raw)
    if protocol == "cot":
        ms = ARC_COT_RE.findall(raw)
        return ms[-1].upper() if ms else ""
    m = ARC_RE.match(raw)
    return m.group(1).upper() if m else ""


def is_correct(task, pred, correct):
    if not pred:
        return False
    if task.startswith("dart"):
        return bool(grade_answer(pred, correct))
    return pred.lower() == correct.lower().strip()[0]


def majority(preds):
    """Most common non-empty prediction; '' if none parseable (an abstention vote)."""
    c = Counter(p for p in preds if p)
    return c.most_common(1)[0][0] if c else ""


def sample_full(model, tok, query, max_new, temp):
    """Full-path temperature sample (best-of-N / samp-vote arm). Default layer_indices."""
    prompt = model.prepare_prompt(query, tok)
    ids = tok(prompt, return_tensors="pt").to(model.model.device)
    input_len = ids.input_ids.shape[1]
    with torch.no_grad():
        out = model.model.generate(
            **ids, max_new_tokens=max_new, do_sample=True, temperature=temp,
            top_p=0.95, pad_token_id=tok.eos_token_id, min_length=input_len + 2)
    return tok.decode(out[0][input_len:], skip_special_tokens=True).strip()


def build_samples(task, protocol, limit, per_level):
    """short: like null_oracle.py. cot(dart): like oracle_cot.py (per_level -> cot_math)."""
    if protocol == "cot":
        assert task.startswith("dart"), "cot fill is for the DART-CoT column"
        samples = load_task(task, "short", 0)
        by = {}
        for s in samples:
            by.setdefault(s["level"], []).append(s)
        samples = [s for lv in sorted(by) for s in by[lv][:per_level]]
        for s in samples:                       # rebuild prompts as CoT (oracle_cot.py:62-65)
            q = s["input"].split("Question: ", 1)[1].rsplit("\nThe final answer", 1)[0]
            s["input"] = cot_math.format(question=q)
        return samples
    if task.startswith("dart"):
        samples = load_task(task, "short", per_level if task == "dart" else 0)
        if task == "dart_pool":
            by = {}
            for s in samples:
                by.setdefault(s["level"], []).append(s)
            samples = [s for lv in sorted(by) for s in by[lv][:per_level]]
        return samples
    return load_task(task)[:limit]


def main():
    ap = argparse.ArgumentParser()
    # HF hub-cache snapshot (has config.json -> model_type=llama; the top-level
    # cache dir has none, so transformers falls back to name matching and "stoRAGe"
    # -> rag config bug). Same 3B-Instruct path the mlp_* rjob scripts use.
    ap.add_argument("--model_path", default="models/Llama-3.2-3B-Instruct/snapshots/392a143b624368100f77a3eafaa4a2468ba50a72")
    ap.add_argument("--task", required=True,
                    choices=["arc_easy", "arc_challenge", "dart", "dart_pool"])
    ap.add_argument("--protocol", choices=["short", "cot"], default="short")
    ap.add_argument("--budget", type=int, default=200)      # oracle arms (d)(e)
    ap.add_argument("--vote_k", type=int, default=16)        # vote arms (d')(e')
    ap.add_argument("--limit", type=int, default=300)        # short arc
    ap.add_argument("--per_level", type=int, default=60)     # dart short=60(->300); cot=20(->100)
    ap.add_argument("--cot_max_new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--skip_only", action="store_true",
                    help="reroll skips layers only (no repeats) -> KV cache stays on; "
                         "required for the CoT protocol to be affordable")
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    samples = build_samples(args.task, args.protocol, args.limit, args.per_level)
    is_dart = args.task.startswith("dart")
    max_new = args.cot_max_new if args.protocol == "cot" else (15 if is_dart else 10)
    print(f"reroll_fill: {args.task}/{args.protocol} n={len(samples)} budget={args.budget} "
          f"vote_k={args.vote_k} max_new={max_new}", flush=True)

    model = MCTSModel(args.model_path, rank=0)
    tok = model.tokenizer
    nl = model.num_layers
    full = LayerPath(list(range(nl)))
    print(f"num_layers={nl} model={args.model_path}", flush=True)

    def greedy(query, path):
        return model.generate_with_path(query, path, max_new_tokens=max_new)

    torch.backends.cudnn.benchmark = False   # avoid per-shape autotune spikes
    greedy(samples[0]["input"], full)        # warm decode kernels once (cold shapes cost 10-30s)
    print("[warmup] done", flush=True)

    rng = random.Random(0)
    recs = []
    t0 = time.time()
    for i, s in enumerate(samples):
        q, gold = s["input"], s["correct"]
        orig_hit = is_correct(args.task, extract_pred(args.task, args.protocol, greedy(q, full)), gold)

        # (d) random-path oracle, early-exit + dedupe (null_oracle.py parity)
        rp_hit, seen, evals = orig_hit, set(), 0
        while evals < args.budget and not rp_hit:
            p = random_edit_path(rng, rng.choice([1, 2, 3]), nl, args.skip_only)
            key = tuple(p.layers)
            if key in seen:
                continue
            seen.add(key)
            evals += 1
            if is_correct(args.task, extract_pred(args.task, args.protocol, greedy(q, p)), gold):
                rp_hit = True

        # (e) best-of-N oracle, early-exit sampling
        bo_hit, ns = orig_hit, 0
        while ns < args.budget and not bo_hit:
            ns += 1
            raw = sample_full(model, tok, q, max_new, args.temperature)
            if is_correct(args.task, extract_pred(args.task, args.protocol, raw), gold):
                bo_hit = True

        # (d') deployable path-vote@vote_k  (e') deployable samp-vote@vote_k  -- no early-exit
        path_preds, samp_preds = [], []
        for _ in range(args.vote_k):
            p = random_edit_path(rng, rng.choice([1, 2, 3]), nl, args.skip_only)
            path_preds.append(extract_pred(args.task, args.protocol, greedy(q, p)))
            samp_preds.append(extract_pred(
                args.task, args.protocol, sample_full(model, tok, q, max_new, args.temperature)))
        pv_hit = is_correct(args.task, majority(path_preds), gold)
        sv_hit = is_correct(args.task, majority(samp_preds), gold)

        recs.append({"level": s.get("level"), "orig": float(orig_hit),
                     "randpath": float(rp_hit), "bestofn": float(bo_hit),
                     "path_vote": float(pv_hit), "samp_vote": float(sv_hit)})
        if (i + 1) % args.print_every == 0:
            def m(k):
                return sum(r[k] for r in recs) / len(recs)
            print(f"{i+1}/{len(samples)} orig={m('orig'):.3f} randpath={m('randpath'):.3f} "
                  f"bestofn={m('bestofn'):.3f} pathvote={m('path_vote'):.3f} "
                  f"sampvote={m('samp_vote'):.3f} ({(time.time()-t0)/(i+1):.1f}s/sample)", flush=True)

    n = len(recs)
    summary = {"task": args.task, "protocol": args.protocol, "n": n,
               "budget": args.budget, "vote_k": args.vote_k, "temperature": args.temperature,
               "orig_acc": sum(r["orig"] for r in recs) / n,
               "randpath_oracle": sum(r["randpath"] for r in recs) / n,
               "bestofn_oracle": sum(r["bestofn"] for r in recs) / n,
               "path_vote_acc": sum(r["path_vote"] for r in recs) / n,
               "samp_vote_acc": sum(r["samp_vote"] for r in recs) / n}
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.task}_{args.protocol}"
    out = os.path.join(args.out_dir, f"fill_{tag}.json")
    json.dump({"summary": summary, "records": recs}, open(out, "w"), indent=1)
    print("SUMMARY", json.dumps(summary, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
