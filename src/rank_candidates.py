#!/usr/bin/env python3
"""N5 "RL AS JUDGE" (07-13). Did RLVR improve GENERATION or DISCRIMINATION?
Take the BASE model's sampled candidate answers (per-question records already on disk
from the 07-12 K200 rerolls) and let a SCORER model rank them by answer log-likelihood
(R-channel-as-selector). Report:
    rank_acc     : accuracy of the scorer's top-1 candidate
    majority_acc : self-consistency vote over the same candidates (verifier-free floor)
    oracle_acc   : any-candidate-correct (cheating ceiling)
    recovery     : (rank_acc - majority_acc) / (oracle_acc - majority_acc)
Pre-registered readout (ledger 07-13 §8): scorer=RL-Zero recovering >=50% of the
selection gap => "RLVR mainly built an internal judge"; recovery ~ 0 and == base-scorer
control => "pure distribution collapse, no discrimination gain". Connects the seven-route
seal (self-verify was dead for base models -- is the RL'd model a better judge?).

    python rank_candidates.py --records results/lens/reroll_rlvrK200_olmo3base_gsm8k_t08.json \
        --task gsm8k --scorer <path> --out results/lens/rank_<scorer>_basecands_gsm8k.json
    python rank_candidates.py --selftest
"""
import argparse, collections, hashlib, json, os

import reroll_sc  # GSM_COT_TMPL / grade_ans / task conventions (same dir)


def build_inputs(task, limit):
    """Rebuild the exact per-question prompts the reroll used (deterministic), so the
    records' qid (md5 of input) can be re-attached to question text."""
    import eval_drllm
    sub = 30 if task.startswith("dart") else 0
    samples = eval_drllm.load_task(task, "cot", sub, -1)[:limit]
    if task == "gsm8k":
        for s in samples:
            q = s["input"].split("Question: ", 1)[1].rsplit("\nGive ONLY", 1)[0]
            s["input"] = reroll_sc.GSM_COT_TMPL.format(q=q)
    return samples


def continuation(task, ans):
    if task.startswith("dart"):
        return f"The final answer is $\\boxed{{{ans}}}$."
    return f"The answer is {ans}."


def dedup_candidates(sample_answers):
    """distinct non-null candidates + vote counts (majority baseline uses counts)."""
    counts = collections.Counter(a for a in sample_answers if a is not None)
    return list(counts.keys()), counts


def recovery(rank_acc, maj_acc, orc_acc):
    gap = orc_acc - maj_acc
    return None if gap <= 1e-9 else (rank_acc - maj_acc) / gap


def judge_records(records, score_fn, task, inputs_by_qid):
    """Core loop, model-free for testability: score_fn(qtext, cand) -> float."""
    per, acc = [], {"rank": 0.0, "majority": 0.0, "oracle": 0.0, "single": 0.0}
    n_used = n_skipped = 0
    for rec in records:
        qtext = inputs_by_qid.get(rec["qid"])
        cands, counts = dedup_candidates(rec["sample_answers"])
        if qtext is None or not cands:
            n_skipped += 1
            continue
        n_used += 1
        gold = rec["gold"]
        scores = {a: score_fn(qtext, a) for a in cands}
        rank_ans = max(scores, key=scores.get)
        maj_ans = counts.most_common(1)[0][0]
        r_hit = reroll_sc.grade_ans(task, rank_ans, gold, "cot")
        m_hit = reroll_sc.grade_ans(task, maj_ans, gold, "cot")
        o_hit = float(any(reroll_sc.grade_ans(task, a, gold, "cot") > 0 for a in cands))
        acc["rank"] += float(r_hit > 0); acc["majority"] += float(m_hit > 0)
        acc["oracle"] += o_hit; acc["single"] += rec.get("single_hit", 0)
        per.append({"qid": rec["qid"], "gold": gold, "n_cands": len(cands),
                    "rank_ans": rank_ans, "rank_hit": int(r_hit > 0),
                    "majority_hit": int(m_hit > 0), "oracle_hit": int(o_hit > 0)})
    if n_used:
        for k in acc: acc[k] = round(acc[k] / n_used, 3)
    return per, acc, n_used, n_skipped


def run_selftest():
    # synthetic 3-question set; scorer prefers longer answers ("42" beats "7")
    inputs = {"q1": "one plus one?", "q2": "six times seven?", "q3": "empty"}
    qid = {k: hashlib.md5(v.encode()).hexdigest()[:10] for k, v in inputs.items()}
    recs = [
        {"qid": qid["q1"], "gold": "2", "single_hit": 0,
         "sample_answers": ["3", "3", "2", None]},          # maj wrong, oracle right, rank->"3"(len1 tie kept)
        {"qid": qid["q2"], "gold": "42", "single_hit": 1,
         "sample_answers": ["42", "7", "7"]},               # maj wrong(7), rank->"42" (longer)
        {"qid": qid["q3"], "gold": "1", "single_hit": 0,
         "sample_answers": [None, None]},                   # no candidates -> skipped
    ]
    inputs_by_qid = {qid[k]: v for k, v in inputs.items()}
    per, acc, used, skipped = judge_records(
        recs, lambda q, a: float(len(a)), "gsm8k", inputs_by_qid)
    assert used == 2 and skipped == 1, (used, skipped)
    assert acc["oracle"] == 1.0 and acc["majority"] == 0.0, acc     # both questions: maj wrong, oracle right
    assert acc["rank"] == 0.5, acc                                   # q2 ranked right, q1 ranked wrong
    assert recovery(acc["rank"], acc["majority"], acc["oracle"]) == 0.5
    assert recovery(0.5, 0.4, 0.4) is None                           # zero-gap guard
    c, cnt = dedup_candidates(["a", "a", None, "b"])
    assert set(c) == {"a", "b"} and cnt["a"] == 2
    # qid mismatch -> question skipped, not misjoined
    _, _, u2, s2 = judge_records([{"qid": "deadbeef00", "gold": "1", "single_hit": 0,
                                   "sample_answers": ["1"]}], lambda q, a: 0.0, "gsm8k", inputs_by_qid)
    assert u2 == 0 and s2 == 1
    print("[selftest] dedup + majority + rank + recovery + alignment-skip OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records"); ap.add_argument("--task")
    ap.add_argument("--scorer"); ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out"); ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--mm_model", action="store_true")
    ap.add_argument("--think", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        run_selftest(); return
    assert args.records and args.task and args.scorer and args.out

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
    rec_blob = json.load(open(args.records))
    records = rec_blob["records"]
    samples = build_inputs(args.task, args.limit)
    inputs_by_qid = {hashlib.md5(s["input"].encode()).hexdigest()[:10]: s["input"] for s in samples}
    n_match = sum(1 for r in records if r["qid"] in inputs_by_qid)
    assert n_match >= 0.95 * len(records), \
        f"qid alignment {n_match}/{len(records)} < 95% -- task/limit mismatch with records"

    tokp = args.tokenizer or args.scorer
    tok = AutoTokenizer.from_pretrained(tokp)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        for _cand in ("chat_template.json", "chat_template.jinja"):
            _p = os.path.join(tokp, _cand)
            if os.path.exists(_p):
                _raw = open(_p).read()
                tok.chat_template = json.loads(_raw)["chat_template"] if _cand.endswith(".json") else _raw
                break
    use_base = tok.chat_template is None
    Loader = AutoModelForImageTextToText if args.mm_model else AutoModelForCausalLM
    model = Loader.from_pretrained(args.scorer, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"})
    model.eval()

    def wrap(q):
        if use_base:
            return q + "\n"
        kw = {}
        if args.think != "auto":
            kw["enable_thinking"] = (args.think == "on")
        return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, **kw)

    _pcache = {}
    def score_fn(qtext, cand):
        key = qtext[:64]
        if key not in _pcache:
            _pcache.clear()  # one question at a time; keep memory flat
            _pcache[key] = tok(wrap(qtext), return_tensors="pt").input_ids.to(model.device)
        pids = _pcache[key]
        cont = tok(continuation(args.task, cand), return_tensors="pt",
                   add_special_tokens=False).input_ids.to(model.device)
        full = torch.cat([pids, cont], dim=1)
        with torch.no_grad():
            logits = model(full).logits.float()
        lp = torch.log_softmax(logits[0, :-1], dim=-1)
        tgt = full[0, 1:]
        cs = lp[torch.arange(pids.shape[1] - 1, full.shape[1] - 1), tgt[pids.shape[1] - 1:]]
        return cs.mean().item()

    per, acc, used, skipped = judge_records(records, score_fn, args.task, inputs_by_qid)
    rec_acc = rec_blob.get("acc", {})
    out = {"config": {"records": args.records, "task": args.task, "scorer": args.scorer,
                      "think": args.think, "prompt_mode": "base" if use_base else "chat",
                      "n_used": used, "n_skipped": skipped},
           "acc": acc,
           "recovery": recovery(acc["rank"], acc["majority"], acc["oracle"]),
           "records_source_acc": rec_acc, "per_question": per}
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"=== rank-as-judge {args.task} n={used} (skipped {skipped}) ===")
    print(f"  rank={acc['rank']} majority={acc['majority']} oracle={acc['oracle']} "
          f"single(src)={acc['single']} recovery={out['recovery']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
