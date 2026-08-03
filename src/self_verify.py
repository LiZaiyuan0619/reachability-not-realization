"""W8 SELF-VERIFICATION selector -- the FOURTH deployable selector for the selection-
unrecoverability claim. majority (flat/negative), confidence (collapses) and the router
(~baseline) all failed to recover the oracle@k headroom stored in the v2 fair-budget
audit; here the model itself is the judge: each stored candidate answer is fed back with
the original question ("Is the proposed answer correct?") and scored as a FORCED-CHOICE
Yes/No continuation logprob -- no generation, one short full-path forward per UNIQUE
candidate answer (Yes/No are single tokens for Llama-3 tokenizers, so both logprobs are
read off one forward's final logits).

Pipeline per (fairbudget v2 json, arm in {mcts,rand,samp}):
  1. re-derive the task samples in v2's exact order (eval_drllm.load_task with the
     w8_fairbudget_v2.sh invocation: MC -> load_task(task)[:limit], dart -> 60/level),
     then AUDIT the alignment: record count == rebuild count; for MC EVERY record with a
     parseable orig_answer must satisfy (orig_answer == gold) == (orig > 0.5) (over 300
     samples full agreement is near-proof of ordering); for dart* the per-record level
     sequence must match and 5 sampled records must pass grade_answer(orig_answer, gold)
     == orig (mathruler is GPU-env-only, so the dart check runs on the worker).
  2. take the first --k candidates of the arm, de-duplicate the answer strings and drop
     unverifiable ones: '' always, plus the literal string 'None' for dart* -- mathruler's
     extract_boxed_content returns the STRING 'None' (not empty) when the candidate
     emitted no \\boxed{...}; such candidates are never graded correct (28302/28302 in
     the l3b dart file), so "verifying" a non-answer would be meaningless. NOTE the
     majority BASELINE keeps the audit's established convention (fair_budget_analyze.
     majority_hit: only ''/None abstain -> dart 'None' strings DO vote there); the
     baseline numbers must stay bit-identical to the fairv2 summaries.
  3. score each unique answer once (cache shared across arms of the same file: the
     greedy candidate and most letters overlap between rand and samp -> ~40% fewer
     forwards) and select:
        margin = logp("Yes") - logp("No");   verdict Yes  <=>  margin > 0
        pick argmax margin; ties -> higher raw logp(Yes); still tied -> earliest first
        appearance. If NO unique answer is judged Yes (or nothing is verifiable), FALL
        BACK to candidate #1 (index 0 = the full-path greedy answer, i.e. the deployed
        default) -- so selfverify@1 == pass@1 == baseline by construction.
  4. write results/w8_selfverify/sv_<tag>_<task>_<arm>.json: per-sample pick + per-
     candidate Yes/No scores + summary selfverify@k vs majority@k / confidence@k /
     oracle@k (baselines recomputed with fair_budget_analyze's selectors and CROSS-
     CHECKED bit-for-bit against the stored fairv2 summary where k overlaps), plus
     verifier calibration (TPR/FPR of the Yes verdict over scored candidates).

Verify prompt (user message, chat-templated exactly like eval_drllm / MCTSModel.
prepare_prompt: single user turn, add_generation_prompt=True):
    <original task input verbatim>
    \n\nProposed answer: <candidate>\n\n
    Is the proposed answer correct? Answer with only "Yes" or "No".
The original input is reused VERBATIM (zero reconstruction risk; it is the exact text
the candidates were produced from). Its embedded answer-format instruction cannot leak
into the judgment because nothing is generated -- only the Yes/No token logprobs are
compared. For MC tasks the candidate letter is expanded to "C. <option text>" (the
rebuilt samples carry `choices`), giving the judge the semantic content of the choice;
dart candidates are the raw boxed strings (same strings grade() scored, digits included).

Model loading: plain transformers AutoModelForCausalLM (stock modeling code, NO
data_models drop-in / PYTHONPATH shadow) -- the verify pass is an ordinary full-path
forward, layer routing never enters. bf16 on cuda:0, one load amortized over every
(file, arm) combo passed in.

Self-test (no GPU, no model, no datasets): python self_verify.py --selftest
Run via rjob/self_verify_run.sh (submit: rjob/submit_selfverify_0710.sh).
"""
import argparse
import json
import os
import time

# pure-python baseline selectors: single source of truth shared with the fairv2 pipeline
# so the majority/confidence/oracle columns are bit-identical to the stored summaries.
from fair_budget_analyze import confidence_hit, majority_hit, oracle_hit

VERIFY_PROMPT = ("{question}\n\n"
                 "Proposed answer: {answer}\n\n"
                 'Is the proposed answer correct? Answer with only "Yes" or "No".')


# ----------------------------------------------------------------- pure-python core
def nonvote_markers(task):
    """Answer strings that mean 'the candidate produced NO parseable answer' and are
    therefore skipped by the verifier (nothing to judge): '' for every task; dart*
    additionally uses the literal 'None' (mathruler extract_boxed_content's not-found
    marker -- see module docstring)."""
    return {"", "None"} if task.startswith("dart") else {""}


def unique_verifiable(answers, k, task):
    """Ordered (answer, first_index) of the unique verifiable answer strings among the
    first k candidates. first_index is the answer's earliest position in the k-window;
    identical strings share a stored grade (guaranteed by fair_budget_oracle_v2's
    extract_answer == grade mirroring), so one verification covers all duplicates."""
    skip = nonvote_markers(task)
    out, seen = [], set()
    for i, a in enumerate(answers[:k]):
        if a is None or a in skip or a in seen:
            continue
        seen.add(a)
        out.append((a, i))
    return out


def finalize(scored, answers, correct, k):
    """Pure selection over scored = [(answer, first_idx, yes_lp, no_lp), ...]:
    argmax margin (= yes - no), verdict Yes only if margin > 0; ties -> higher raw
    yes_lp; still tied -> earliest first_idx. All-No / nothing-verifiable -> FALL BACK
    to candidate #1 (index 0, the deployed greedy default). Returns
    {index, answer, correct, fallback, margin} (margin = winner's margin, or the best
    rejected margin under fallback, or None if nothing was verifiable)."""
    win_a, win_c = answers[:k], correct[:k]
    best = None
    for a, ix, y, n in scored:
        key = (y - n, y, -ix)
        if best is None or key > best[0]:
            best = (key, a, ix)
    if best is not None and best[0][0] > 0:
        key, a, ix = best
        return {"index": ix, "answer": a, "correct": int(float(win_c[ix]) > 0.5),
                "fallback": False, "margin": key[0]}
    return {"index": 0, "answer": win_a[0] if win_a else "",
            "correct": int(float(win_c[0]) > 0.5) if win_c else 0,
            "fallback": True, "margin": best[0][0] if best else None}


def select_by_verify(answers, correct, k, task, score_fn):
    """De-dup the k-window, score each unique answer once via score_fn(ans) -> (yes, no),
    select per finalize(). Returns (selection dict, scored list) -- the scored list feeds
    the per-record dump and the k-sweep (smaller windows reuse the same scores)."""
    scored = [(a, ix) + tuple(score_fn(a)) for a, ix in unique_verifiable(answers, k, task)]
    return finalize(scored, answers, correct, k), scored


def sweep_hits(scored, answers, correct, ks):
    """selfverify hit at every k' in ks from ONE full-k scored list: the k'-window's
    unique set is exactly the scored entries with first_idx < k' (first occurrences only
    move earlier as the window shrinks), so no re-scoring is needed."""
    return {k: finalize([t for t in scored if t[1] < k], answers, correct, k)["correct"]
            for k in ks}


def render_candidate(task, ans, sample):
    """Presentation of the candidate inside the verify prompt. MC letters are expanded
    with the option text ('C' -> 'C. photosynthesis') so the judge sees the semantic
    content, not just a label; dart/gsm8k answers are the raw extracted strings."""
    if not task.startswith("dart") and task != "gsm8k":
        ch = sample.get("choices")
        if ch and len(ans) == 1:
            j = ord(ans.upper()) - 65
            if 0 <= j < len(ch):
                return f"{ans}. {ch[j]}"
    return ans


def build_verify_text(task, sample, ans):
    """The verify user message: original task input VERBATIM + rendered candidate +
    the Yes/No question. Chat template is applied by the caller (needs the tokenizer)."""
    return VERIFY_PROMPT.format(question=sample["input"],
                                answer=render_candidate(task, ans, sample))


def rebuild_samples(task, limit, per_level):
    """Samples in fair_budget_oracle_v2.run_task's exact order (== w8_fairbudget_v2.sh:
    --per_level 60 --limit 300). Imports eval_drllm lazily (torch/datasets/mathruler)."""
    from eval_drllm import load_task
    if task.startswith("dart"):
        samples = load_task(task, "short", per_level if task == "dart" else 0)
        if task == "dart_pool":
            by = {}
            for s in samples:
                by.setdefault(s["level"], []).append(s)
            samples = [s for lv in sorted(by) for s in by[lv][:per_level]]
        return samples
    return load_task(task)[:limit]


def audit_alignment(task, samples, records, n_show=5):
    """Prove the rebuilt sample order matches the v2 records before burning GPU time.
    MC: (orig_answer == gold) <=> (orig > 0.5) must hold for EVERY record with a
    parseable orig_answer (extract_answer and grade share the same anchored regex, so
    this is an exact invariant; full agreement over ~300 samples is near-proof of
    ordering -- a shuffle would break it at ~chance rate). dart*: level sequence must
    match record-for-record, plus grade_answer(orig_answer, gold) == orig on n_show
    sampled records (mathruler import is GPU-env-only, hence only here)."""
    assert len(samples) == len(records), \
        f"{task}: rebuilt {len(samples)} samples but v2 file has {len(records)} records"
    if task.startswith("dart"):
        lv_rec = [r["level"] for r in records]
        lv_new = [s["level"] for s in samples]
        assert lv_rec == lv_new, f"{task}: level sequence mismatch (order drift)"
        from mathruler.grader import grade_answer
        idxs = [i for i, r in enumerate(records)
                if r["orig_answer"] not in ("", "None")][:n_show]
        assert idxs, f"{task}: no parseable orig_answer to audit"
        for i in idxs:
            g = float(grade_answer(records[i]["orig_answer"], samples[i]["correct"]))
            assert (g > 0.5) == (records[i]["orig"] > 0.5), \
                (f"{task} sample {i}: orig_answer={records[i]['orig_answer']!r} vs "
                 f"gold={samples[i]['correct']!r} regrades {g} but orig={records[i]['orig']}")
        print(f"[audit {task}] levels aligned; {len(idxs)} regraded records consistent")
    else:
        checked = 0
        for i, (r, s) in enumerate(zip(records, samples)):
            if r["orig_answer"]:
                assert (r["orig_answer"] == s["correct"]) == (r["orig"] > 0.5), \
                    (f"{task} sample {i}: orig_answer={r['orig_answer']!r} "
                     f"gold={s['correct']!r} orig={r['orig']} -- order drift?")
                checked += 1
        assert checked >= n_show, f"{task}: only {checked} parseable orig_answers"
        for i in [j for j, r in enumerate(records) if r["orig_answer"]][:n_show]:
            print(f"[audit {task}] i={i} orig_answer={records[i]['orig_answer']} "
                  f"gold={samples[i]['correct']} orig={records[i]['orig']} OK")
        print(f"[audit {task}] {checked}/{len(records)} parseable records all consistent")


# ------------------------------------------------------------------------- selftest
def _selftest():
    """CPU-only: selection logic on simulated records -- no model, no datasets."""
    def fixed(scores, calls=None):
        def f(a):
            if calls is not None:
                calls.append(a)
            return scores[a]
        return f

    # Case 1 RESCUE: gold=B. majority picks A (2 votes) -> wrong; the verifier says Yes
    # only to B -> self-verify recovers the oracle-reachable answer. Empties skipped,
    # duplicates scored once.
    ans1, cor1 = ["A", "B", "A", ""], [0, 1, 0, 0]
    sc1 = {"A": (-3.0, -0.1), "B": (-0.2, -2.5)}
    calls = []
    sel, scored = select_by_verify(ans1, cor1, 4, "arc_challenge", fixed(sc1, calls))
    assert calls == ["A", "B"], "dedup + empty-skip: exactly one call per unique answer"
    assert scored == [("A", 0, -3.0, -0.1), ("B", 1, -0.2, -2.5)]
    assert (sel["index"], sel["answer"], sel["correct"], sel["fallback"]) == (1, "B", 1, False)
    assert oracle_hit(cor1, 4) == 1 and majority_hit(ans1, cor1, 4) == 0, \
        "sv recovers a sample majority loses"

    # Case 2 ALL-NO -> fall back to candidate #1 (both polarities of candidate #1);
    # margin == 0 also counts as No.
    sc_no = {"A": (-2.0, -0.1), "B": (-1.5, -0.2)}
    sel, _ = select_by_verify(["A", "B"], [0, 1], 2, "arc_challenge", fixed(sc_no))
    assert sel["fallback"] and sel["index"] == 0 and sel["correct"] == 0
    sel, _ = select_by_verify(["B", "A"], [1, 0], 2, "arc_challenge", fixed(sc_no))
    assert sel["fallback"] and sel["index"] == 0 and sel["correct"] == 1
    sel, _ = select_by_verify(["A"], [1], 1, "arc_challenge", fixed({"A": (-1.0, -1.0)}))
    assert sel["fallback"], "yes == no is NOT a Yes verdict"

    # Case 3 margin TIE -> higher raw Yes logit wins; full tie -> earliest appearance.
    sel, _ = select_by_verify(["A", "B"], [0, 1], 2, "arc_challenge",
                              fixed({"A": (-1.0, -2.0), "B": (-0.5, -1.5)}))
    assert sel["answer"] == "B", "equal margins -> pick the higher yes-logit"
    sel, _ = select_by_verify(["A", "B"], [1, 0], 2, "arc_challenge",
                              fixed({"A": (-0.5, -1.5), "B": (-0.5, -1.5)}))
    assert sel["answer"] == "A", "full tie -> earliest first appearance"

    # Case 4 nothing verifiable (all empty) -> fallback, zero scoring calls.
    calls = []
    sel, scored = select_by_verify(["", "", ""], [0, 0, 0], 3, "arc_challenge",
                                   fixed({}, calls))
    assert calls == [] and scored == []
    assert sel["fallback"] and sel["index"] == 0 and sel["correct"] == 0 and sel["margin"] is None

    # Case 5 dart's 'None' marker abstains (a non-answer), while an MC task would treat
    # the same string as a real candidate -- the skip rule is task-scoped.
    assert [a for a, _ in unique_verifiable(["None", "7", ""], 3, "dart")] == ["7"]
    assert [a for a, _ in unique_verifiable(["None", "7", ""], 3, "arc_challenge")] == ["None", "7"]
    calls = []
    sel, _ = select_by_verify(["None", "7"], [0, 1], 2, "dart",
                              fixed({"7": (-0.1, -2.0)}, calls))
    assert calls == ["7"] and sel["answer"] == "7" and sel["correct"] == 1

    # Case 6 the k-window: only the first k candidates exist; at k=1 self-verify equals
    # the baseline by construction (Yes -> keep candidate #1, No -> fall back to it).
    calls = []
    sel, _ = select_by_verify(["A", "B"], [0, 1], 1, "arc_challenge",
                              fixed({"A": (-0.1, -2.0)}, calls))
    assert calls == ["A"] and sel["index"] == 0 and not sel["fallback"]
    sel, _ = select_by_verify(["A", "B"], [0, 1], 1, "arc_challenge",
                              fixed({"A": (-2.0, -0.1)}))
    assert sel["index"] == 0 and sel["fallback"], "k=1 all-No falls back to the same candidate"

    # Case 7 sweep from one scored list == fresh per-window selection; the rescue only
    # appears once the Yes-judged candidate enters the window.
    ans7, cor7 = ["A", "A", "A", "B"], [0, 0, 0, 1]
    sc7 = {"A": (-2.0, -0.1), "B": (-0.1, -2.0)}
    _, scored7 = select_by_verify(ans7, cor7, 4, "arc_challenge", fixed(sc7))
    hits = sweep_hits(scored7, ans7, cor7, [1, 2, 4])
    assert hits == {1: 0, 2: 0, 4: 1}
    for k in (1, 2, 4):
        fresh, _ = select_by_verify(ans7, cor7, k, "arc_challenge", fixed(sc7))
        assert hits[k] == fresh["correct"], f"sweep@{k} != fresh selection@{k}"

    # prompt rendering (pure string): MC letter expanded with option text; dart raw.
    s = {"input": "Question: q?\nA. x\nB. y", "choices": ["x", "y"], "correct": "B"}
    assert render_candidate("arc_challenge", "B", s) == "B. y"
    assert render_candidate("arc_challenge", "E", s) == "E", "letter beyond options stays raw"
    assert render_candidate("dart", "\\frac{1}{2}", {"input": "..."}) == "\\frac{1}{2}"
    txt = build_verify_text("arc_challenge", s, "B")
    assert txt.startswith(s["input"]) and "Proposed answer: B. y" in txt \
        and 'Answer with only "Yes" or "No"' in txt
    print("SELFTEST OK (rescue + dedup/empty-skip + all-No fallback [both polarities, "
          "margin==0] + margin-tie->yes-logit->earliest + dart 'None' abstain + k-window/"
          "k=1==baseline + sweep==fresh + prompt rendering)")


# ----------------------------------------------------------------------- GPU driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="CPU-only logic check, no model")
    ap.add_argument("--fairbudget", nargs="+", default=[],
                    help="fairv2_<task>_seed<seed>.json v2 audit file(s); several files "
                         "run serially under ONE model load")
    ap.add_argument("--arms", "--arm", dest="arms", default="rand,samp",
                    help="comma list subset of {mcts,rand,samp} (single value also fine)")
    ap.add_argument("--k", type=int, default=64, help="candidate budget (window into the "
                    "stored kmax candidates; l3b stored 200, l318b stored 64)")
    ap.add_argument("--model_path")
    ap.add_argument("--tag", default="", help="output name piece: sv_<tag>_<task>_<arm>.json "
                    "(l3b / l318b, matching the fairbudget dir)")
    ap.add_argument("--limit", type=int, default=300, help="MC sample cap (== v2 run)")
    ap.add_argument("--per_level", type=int, default=60, help="dart samples/level (== v2 run)")
    ap.add_argument("--out_dir", default="results/w8_selfverify")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    assert args.model_path and args.fairbudget and args.tag, \
        "need --model_path, --fairbudget and --tag"
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    assert arms and all(a in ("mcts", "rand", "samp") for a in arms), f"bad --arms {args.arms!r}"
    KS = sorted({k for k in [1, 2, 4, 5, 8, 16, 32, args.k] if k <= args.k})

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map={"": "cuda:0"})
    model.eval()
    # the verify pass is a plain full-path forward: STOCK modeling code, no drop-in
    assert not hasattr(model.model, "routers"), "router drop-in active -- run without shadow"

    tmpl_kw = {}
    if "qwen3" in args.model_path.lower():
        tmpl_kw["enable_thinking"] = False  # == eval_drllm's template dialect handling
    yes_ids = tok("Yes", add_special_tokens=False).input_ids
    no_ids = tok("No", add_special_tokens=False).input_ids
    fast = len(yes_ids) == 1 and len(no_ids) == 1
    print(f"self-verify: model loaded ({args.model_path}) k={args.k} arms={arms} "
          f"Yes->{yes_ids} No->{no_ids} single-token-fast-path={fast}", flush=True)

    def yes_no_scores(user_text):
        """(logp Yes, logp No) as assistant-reply continuations of user_text. Single-token
        fast path (Llama-3): ONE forward, read both token logprobs off the final position.
        Multi-token tokenizers fall back to mean logprob per continuation (2 forwards),
        the eval_drllm.cont_logprob convention."""
        prompt = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                         tokenize=False, add_generation_prompt=True, **tmpl_kw)
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            if fast:
                lp = torch.log_softmax(model(ids).logits[0, -1].float(), -1)
                return lp[yes_ids[0]].item(), lp[no_ids[0]].item()

            def mean_lp(cont_ids):
                cid = torch.tensor([cont_ids], device=model.device)
                logits = model(torch.cat([ids, cid], 1)).logits
                lps = torch.log_softmax(logits[0, ids.shape[1] - 1: -1].float(), -1)
                return lps.gather(1, cid[0].unsqueeze(1)).squeeze(1).mean().item()
            return mean_lp(yes_ids), mean_lp(no_ids)

    def run_arm(fb_path, d, arm, task, samples, score_cache, fwd_count):
        summ, recs = d["summary"], d["records"]
        assert f"{arm}_answers" in recs[0], f"{fb_path} lacks the {arm} arm's v2 fields"
        rows = []
        t0 = time.time()
        for i, (rec, s) in enumerate(zip(recs, samples)):
            answers, correct = rec[f"{arm}_answers"], rec[f"{arm}_correct"]

            def score_fn(ans, _i=i, _s=s):
                ck = (_i, ans)  # cache shared across arms: same (sample, answer) -> same prompt
                if ck not in score_cache:
                    score_cache[ck] = yes_no_scores(build_verify_text(task, _s, ans))
                    fwd_count[0] += 1 if fast else 2
                return score_cache[ck]

            sel, scored = select_by_verify(answers, correct, args.k, task, score_fn)
            hits = sweep_hits(scored, answers, correct, KS)
            rows.append({
                "i": i, "level": rec.get("level"), "orig": rec["orig"],
                "gold": s["correct"], "n_uniq": len(scored),
                "pick_index": sel["index"], "pick_answer": sel["answer"],
                "pick_correct": sel["correct"], "fallback": sel["fallback"],
                "margin": None if sel["margin"] is None else round(sel["margin"], 4),
                "verify": [{"ans": a, "idx": ix, "yes": round(y, 4), "no": round(n, 4),
                            "cor": int(float(correct[ix]) > 0.5)} for a, ix, y, n in scored],
                "hits": {str(k): v for k, v in hits.items()},
            })
            if (i + 1) % 50 == 0:
                done = i + 1
                sv = sum(r["hits"][str(args.k)] for r in rows) / done
                print(f"[{task}/{arm}] {done}/{len(recs)} sv@{args.k}={sv:.3f} "
                      f"fwd={fwd_count[0]} ({(time.time() - t0) / done:.2f}s/sample)", flush=True)

        n = len(rows)
        sv = {k: sum(r["hits"][str(k)] for r in rows) / n for k in KS}
        maj = {k: sum(majority_hit(r[f"{arm}_answers"], r[f"{arm}_correct"], k)
                      for r in recs) / n for k in KS}
        cnf = {k: sum(confidence_hit(r[f"{arm}_correct"], r[f"{arm}_conf"], k)
                      for r in recs) / n for k in KS}
        orc = {k: sum(oracle_hit(r[f"{arm}_correct"], k) for r in recs) / n for k in KS}
        orig_acc = sum(1 for r in recs if r["orig"] > 0.5) / n

        # cross-check the recomputed baselines against the fairv2 stored summary wherever
        # the k grids overlap -- guards against any silent record/selector drift.
        matched = []
        for k in KS:
            for mine, key in [(maj[k], f"{arm}_majority"), (orc[k], f"{arm}_passk"),
                              (cnf[k], f"{arm}_confidence")]:
                stored = summ.get(key, {}).get(str(k))
                if stored is not None:
                    assert abs(mine - stored) < 1e-9, \
                        f"{task}/{arm} {key}@{k}: recomputed {mine} != stored {stored}"
            if str(k) in summ.get(f"{arm}_majority", {}):
                matched.append(k)
        assert abs(orig_acc - summ["orig_acc"]) < 1e-9

        # verifier calibration over all scored (unique) candidates at the full window
        pos = neg = tp = fp = 0
        for r in rows:
            for v in r["verify"]:
                say_yes = v["yes"] > v["no"]
                if v["cor"]:
                    pos += 1
                    tp += say_yes
                else:
                    neg += 1
                    fp += say_yes
        summary = {
            "task": task, "arm": arm, "tag": args.tag, "k": args.k, "n": n,
            "seed": summ.get("seed"), "stored_kmax": summ.get("kmax"),
            "model_path": args.model_path, "fairbudget": fb_path,
            "orig_acc": orig_acc,
            "selfverify": sv, "majority": maj, "confidence": cnf, "oracle": orc,
            "delta_sv_minus_majority": {k: sv[k] - maj[k] for k in KS},
            "gap_oracle_minus_sv": {k: orc[k] - sv[k] for k in KS},
            "fallback_rate": sum(r["fallback"] for r in rows) / n,
            "mean_n_uniq": sum(r["n_uniq"] for r in rows) / n,
            "verifier": {"n_scored": pos + neg, "yes_rate": (tp + fp) / max(pos + neg, 1),
                         "tpr": tp / max(pos, 1), "fpr": fp / max(neg, 1)},
            "n_forwards_cum": fwd_count[0], "elapsed_s": round(time.time() - t0, 1),
            "crosscheck_stored_ks": matched,
        }
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, f"sv_{args.tag}_{task}_{arm}.json")
        json.dump({"summary": summary, "records": rows}, open(out, "w"), indent=1)
        print("SUMMARY", json.dumps(summary, indent=1))
        print("saved", out, flush=True)

    # serial loop: per-(file,arm) failures are isolated (traceback + continue) so one bad
    # combo never discards the others; exit non-zero if ANY failed (== v2's convention).
    failed = []
    for fb_path in args.fairbudget:
        score_cache, fwd_count = {}, [0]  # per-file: shared across its arms
        try:
            d = json.load(open(fb_path))
            task = d["summary"]["task"]
            samples = rebuild_samples(task, args.limit, args.per_level)
            audit_alignment(task, samples, d["records"])
        except Exception as e:  # noqa: BLE001 -- salvage the other files
            import traceback
            traceback.print_exc()
            failed += [(fb_path, a) for a in arms]
            print(f"[{fb_path}] load/rebuild/audit FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        for arm in arms:
            try:
                run_arm(fb_path, d, arm, task, samples, score_cache, fwd_count)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                failed.append((fb_path, arm))
                print(f"[{fb_path}/{arm}] FAILED: {type(e).__name__}: {e}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} (file, arm) combo(s) FAILED: {failed}")


if __name__ == "__main__":
    main()
