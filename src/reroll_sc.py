#!/usr/bin/env python3
"""PILLAR 5: does the reroll/verifier-inflation critique GENERALIZE beyond layer-routing
to sampling-based test-time compute (self-consistency / best-of-N)? For each reasoning
question we sample K CoT completions and report:
  single      : greedy (temp 0) -- the deployable baseline
  random@K    : expected accuracy of one random sample (= mean per-sample correctness)
  majority@K  : self-consistency majority vote (deployable, NO external verifier)
  oracle@K    : best-of-N with an ORACLE verifier (= pass@K) -- the CHEATING ceiling
The reroll point (mirrors the layer-routing audit): oracle@K >> majority@K ~ single, i.e.
the headline "ceiling" is a verifier/reroll artifact of the K-sample budget, and the
deployable (verifier-free) gain is modest -- the SAME pattern as random-layer-path+verifier
beating structured MCTS routing. Generalizes "reported gains are reroll" to a 2nd method.
Usage: python reroll_sc.py --model_path P --tokenizer_path P --task gsm8k --K 8
  --temp 0.8 --limit 150 [--mm_model] --out X
"""
import argparse, collections, hashlib, json, os, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
from mathruler.grader import extract_boxed_content, grade_answer
import eval_drllm


# CoT gsm8k prompt (load_task's default gsm8k template forces DIRECT answer
# "give ONLY the numeric answer, no explanation" and ignores proto -> that makes
# self-consistency degenerate/strawman and crushes single to ~0.13). We rewrite to
# a CoT prompt that asks the model to reason then emit an explicit final-answer line.
GSM_COT_TMPL = ("Solve the following grade-school math problem step by step. "
                "End your response with a line 'The answer is N.' where N is the final number.\n"
                "Question: {q}")


def extract(task, resp, proto):
    if task.startswith("dart"):
        # MATH answers live in \boxed{}; grade via mathruler on the boxed content.
        # (GSM_NUM_RE would strip fractions/expressions and boxed grading returns empty.)
        b = extract_boxed_content(resp.strip())
        return b if b else None
    if task == "gsm8k":
        # On a CoT response the GLOBAL last number is often a spurious trailing figure
        # (units, day counts, restatements) -> the pre-fix failure. Prefer an explicit
        # final-answer marker; only fall back to last-number if no marker is present.
        text = resp.replace(",", "").replace("$", "")
        m = re.findall(r"answer\s*(?:is|:|=)*\s*\*{0,2}\s*(-?\d*\.?\d+)", text, re.I)
        if m:
            return m[-1]
        m = re.findall(r"####\s*(-?\d*\.?\d+)", text)
        if m:
            return m[-1]
        nums = eval_drllm.GSM_NUM_RE.findall(text)
        return nums[-1] if nums else None
    ms = eval_drllm.ARC_COT_RE.findall(resp)
    return ms[-1].upper() if ms else None


def grade_ans(task, ans, gold, proto):
    if ans is None: return 0.0
    if task.startswith("dart"):
        return float(grade_answer(ans, gold))
    if task == "gsm8k":
        return eval_drllm.grade(task, ans, gold, "short")
    return eval_drllm.grade(task, f"Answer: {ans}", gold, "cot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True); ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--task", required=True); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8); ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--max_new", type=int, default=512,
                    help="gen budget; MATH/dart CoT needs >=1024 or answers get truncated before \\boxed{}")
    ap.add_argument("--mm_model", action="store_true"); ap.add_argument("--out", required=True)
    ap.add_argument("--think", choices=["auto", "on", "off"], default="auto",
                    help="hybrid-thinking toggle passed to apply_chat_template as enable_thinking; "
                         "'auto' omits the kwarg entirely (byte-identical to all pre-07-13 runs); "
                         "'off' = Qwen3-family nothink protocol, 'on' forces the think block")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    Loader = AutoModelForImageTextToText if args.mm_model else AutoModelForCausalLM
    model = Loader.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"})
    model.eval()
    proto = "cot"; mx = args.max_new
    # BASE-ARM FIX (07-11 RLVR ledger): base (non-instruct) tokenizers carry no
    # chat_template, so the original unconditional apply_chat_template() raised
    # ValueError and killed the base reroll arm (base oracle@K is the CENTRAL ledger
    # measure vs RL-Zero-Math). Mirror eval_drllm.py's already-GPU-verified fix: load
    # a standalone chat_template file if present, else fall back to the raw base prompt.
    # Chat ckpts (chat_template not None: native / embedded / standalone) are
    # BYTE-IDENTICAL to every prior reroll run -- only the previously-crashing base
    # code path changes.
    if tok.chat_template is None:
        for _cand in ("chat_template.json", "chat_template.jinja"):
            _p = os.path.join(args.tokenizer_path, _cand)
            if os.path.exists(_p):
                _raw = open(_p).read()
                tok.chat_template = json.loads(_raw)["chat_template"] if _cand.endswith(".json") else _raw
                print(f"[tok] chat_template loaded from {_cand}", flush=True); break
    _use_base = tok.chat_template is None
    if _use_base:
        print(f"[tok] no chat_template -> base_prompt fallback for reroll ({args.model_path})", flush=True)
    if args.think != "auto" and _use_base:
        print(f"[tok] WARNING --think={args.think} requested but no chat_template -> ignored", flush=True)
    def chat(inp):
        if _use_base:
            p = inp + "\nAnswer:"
        else:
            kw = {}
            if args.think != "auto":
                # jinja templates simply ignore unknown context vars, so this is safe on
                # non-hybrid templates; only Qwen3-style templates branch on it.
                kw["enable_thinking"] = (args.think == "on")
            p = tok.apply_chat_template([{"role": "user", "content": inp}], tokenize=False,
                                        add_generation_prompt=True, **kw)
        return tok(p, return_tensors="pt").to(model.device).input_ids
    def gen(pids, sample, k=1):
        # K>64 chunking (07-12): K=200 x ~1k tok in ONE generate() call would push KV
        # past H200 on full-MHA ckpts (Olmo-3 has no GQA). Chunks are statistically
        # identical (independent samples) and K<=64 calls are byte-identical to before.
        if sample and k > 64:
            outs = []
            for i in range(0, k, 64):
                outs += gen(pids, True, min(64, k - i))
            return outs
        kw = dict(max_new_tokens=mx, pad_token_id=tok.pad_token_id, use_cache=True)
        if sample: kw.update(do_sample=True, temperature=args.temp, num_return_sequences=k)
        else: kw.update(do_sample=False)
        with torch.no_grad(): g = model.generate(pids, **kw)
        return [tok.decode(g[j, pids.shape[1]:], skip_special_tokens=True) for j in range(g.shape[0])]

    sub = 0
    if args.task.startswith("dart"):
        # even coverage across 5 MATH levels; 30/level for limit<=150 (byte-identical to
        # all pre-07-13 runs), scales up so limit=300 (F2 fate-n300) actually gets 300.
        sub = max(30, -(-args.limit // 5))
    samples = eval_drllm.load_task(args.task, proto, sub, -1)[: args.limit]
    if args.task == "gsm8k":
        # Override load_task's direct-answer gsm8k prompt with a CoT prompt so that
        # sampling explores reasoning paths (the faithful self-consistency setup).
        for s in samples:
            q = s["input"].split("Question: ", 1)[1].rsplit("\nGive ONLY", 1)[0]
            s["input"] = GSM_COT_TMPL.format(q=q)
    acc = {"single": 0.0, "random@K": 0.0, "majority@K": 0.0, "oracle@K": 0.0}; n = 0
    # PER-SAMPLE RECORDS (07-12, fate-bucket prerequisite): per question we keep the
    # extracted answers + hit booleans (NOT the completions -- keeps files ~MB not GB)
    # so cross-checkpoint joins (which bucket did each question live in before/after RL)
    # can be computed offline. qid = md5 of the prompt text, the join key + alignment
    # guard across checkpoints (same task+limit => same sample order, qid verifies it).
    recs = []
    for s in samples:
        pids = chat(s["input"]); gold = s["correct"]; n += 1
        s_ans = extract(args.task, gen(pids, False)[0], proto)
        s_hit = grade_ans(args.task, s_ans, gold, proto)
        acc["single"] += s_hit
        comps = gen(pids, True, args.K)
        anss = [extract(args.task, c, proto) for c in comps]
        cors = [grade_ans(args.task, a, gold, proto) for a in anss]
        acc["oracle@K"] += float(any(c > 0 for c in cors))
        acc["random@K"] += sum(cors) / len(cors)
        valid = [a for a in anss if a is not None]
        maj, m_hit = None, 0.0
        if valid:
            maj = collections.Counter(valid).most_common(1)[0][0]
            m_hit = grade_ans(args.task, maj, gold, proto)
            acc["majority@K"] += m_hit
        recs.append({"qid": hashlib.md5(s["input"].encode()).hexdigest()[:10],
                     "gold": str(gold), "single_ans": s_ans, "single_hit": int(s_hit > 0),
                     "majority_ans": maj, "majority_hit": int(m_hit > 0),
                     "oracle_hit": int(any(c > 0 for c in cors)),
                     "n_sample_hits": int(sum(1 for c in cors if c > 0)),
                     "sample_answers": anss})
        if n % 25 == 0: print(f"{n}/{len(samples)}", flush=True)
    out = {"config": {"model": args.model_path, "task": args.task, "K": args.K, "temp": args.temp, "n": n,
                      "think": args.think, "prompt_mode": "base" if _use_base else "chat"},
           "acc": {k: round(v / n, 3) for k, v in acc.items()}, "records": recs}
    a = out["acc"]
    out["reroll_gap"] = {"oracle_minus_majority": round(a["oracle@K"] - a["majority@K"], 3),
                         "majority_minus_single": round(a["majority@K"] - a["single"], 3)}
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n=== {args.task} K={args.K} n={n} ===")
    print(f"  single={a['single']} random@K={a['random@K']} majority@K={a['majority@K']} oracle@K={a['oracle@K']}")
    print(f"  ORACLE ceiling over deployable majority = +{out['reroll_gap']['oracle_minus_majority']} (reroll/verifier artifact)")
    print(f"  deployable majority gain over single = +{out['reroll_gap']['majority_minus_single']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
