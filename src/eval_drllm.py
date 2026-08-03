"""Generation-based in-domain eval (ARC / DART) matching data_generation.py's protocol.

The paper's Table 3 in-domain numbers are in THIS protocol, not lm-eval loglikelihood:
Num Layers 103.9 (ARC) / 422.0 (DART) for the 28-layer 3B == ~3.7 / ~15 forwards per
sample == short greedy generation with the data-gen prompts (ARC "Answer: X" max 10
new tokens, DART boxed max 15 new tokens), graded exactly like MCTS rewards.

Modes:
  baseline: stock modeling code (caller guarantees stock file / shadow), use_cache=True
  routed:   models/ drop-in installed (shadow), router ckpt; batch is ALWAYS 1 because
            the drop-in applies decisions[0] batch-wide; use_cache=False so REPEAT
            never hits the KV-cache misalignment and routing pools the full prefix.

Outputs <out_dir>/<task>_<mode>[_p<penalty>].json with summary + per-sample records.
"""
import argparse
import glob
import json
import os
import random
import re
import time

import torch
from datasets import load_dataset
from mathruler.grader import extract_boxed_content, grade_answer
from transformers import AutoTokenizer

from prompts import answer_letter_long, answer_math, format_choices

ARC_RE = re.compile(r"^(?:Answer:\s*)?([A-Ea-e])\.?$")

# W11: AGIEval subsets beyond sat_math have up to 5 options -> letter-agnostic
# instruction (the original answer_letter_long enumerates "A, B, C, or D" and
# must stay untouched for comparability with all existing results)
answer_letter_any = """"Question: {question}
{choices_text}
Answer with a single letter and no explanation. You answer should start with "Answer: " and be followed by the letter of the answer you choose. Do not include any other text in your response."""

# C-probe: realistic CoT protocol (the paper's in-domain protocol forbids CoT;
# does the routed gain survive when the model is allowed to reason?)
cot_math = ("Solve the following math problem step by step. "
            "Put your final answer in \\boxed{{}}.\nQuestion: {question}")

# W11 robustness probe: does the harmful-layer gain survive under CoT on MC tasks,
# or is it an artifact of the forced short-answer format?
cot_mc = """Question: {question}
{choices_text}
Think step by step, then give your final answer on a new line in the form "Answer: <letter>"."""

COT_TASKS = {"dart", "arc_easy", "arc_challenge", "mmlu", "piqa",
             "agieval_sat_math", "truthfulqa"}


def load_task(task, protocol="short", subset_per_level=0, shuffle_seed=-1):
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    samples = []
    shuf_rng = random.Random(shuffle_seed) if shuffle_seed >= 0 else None

    def maybe_shuffle(choices, correct_idx):
        if shuf_rng is None:
            return choices, correct_idx
        order = list(range(len(choices)))
        shuf_rng.shuffle(order)
        return [choices[i] for i in order], order.index(correct_idx)
    if task == "dart":
        files = sorted(glob.glob(f"{hf}/datasets/EleutherAI__hendrycks_math/*/test-*.parquet"))
        assert len(files) == 7, f"expected 7 subject parquets, got {files}"
        ds = load_dataset("parquet", data_files=files, split="train")
        tmpl = cot_math if protocol == "cot" else answer_math
        for item in ds:
            level = int(item["level"].split()[-1])
            gt = extract_boxed_content(item["solution"])
            samples.append({
                "input": tmpl.format(question=item["problem"]),
                "correct": gt,
                "level": level,
            })
        if subset_per_level:
            picked = []
            for lv in range(1, 6):
                pool = [s for s in samples if s["level"] == lv]
                stride = max(1, len(pool) // subset_per_level)
                picked += pool[::stride][:subset_per_level]  # even coverage across subject blocks
            samples = picked
    elif task == "dart_pool":
        # Held-out queries from the MCTS supervision pool (dart-math-pool =
        # MATH-train synthetic queries). Same distribution the router was
        # supervised on; the paper's DART numbers are plausibly on this, not on
        # MATH-test (pool original acc ~70% vs MATH-test 11.8% under this
        # protocol). 200/level from the TAIL of each level (data-gen consumed
        # from the head), minus any query used in training (belt & braces).
        used = set()
        for f in glob.glob("data/train/dart-*.json"):
            used.update(json.load(open(f))["questions"])
        dart_dir = os.path.join(hf, "datasets/hkust-nlp__dart-math-pool-math/data")
        ds = load_dataset("parquet", data_files=os.path.join(dart_dir, "train-*.parquet"),
                          split="train")
        for lv in range(1, 6):
            sub = ds.filter(lambda x: x["query_metadata"]["level"] == lv, num_proc=16)
            picked, seen = [], set()
            for item in reversed(list(sub)):
                inp = answer_math.format(question=item["query"])
                if inp in used or item["query"] in seen:
                    continue
                seen.add(item["query"])
                picked.append({"input": inp, "correct": item["gt_ans"], "level": lv})
                if len(picked) >= 200:
                    break
            samples += picked
    elif task in ("mmlu", "piqa", "agieval_sat_math", "truthfulqa", "gsm8k",
                  "hellaswag", "openbookqa", "commonsense_qa", "boolq", "winogrande") \
            or task.startswith("agieval_"):
        # OOD tasks (W9 harmful-layer breadth scan). All datasets live in
        # $HF_HOME/datasets as load_from_disk dirs (NOT load_dataset).
        # MC tasks reuse the exact ARC protocol (answer_letter_long + letter
        # grading); gsm8k is generative with numeric grading.
        from datasets import load_from_disk
        import random as _random
        rng = _random.Random(0)

        mc_tmpl = cot_mc if protocol == "cot" else answer_letter_long

        def mc(question, choices, correct_idx):
            choices, correct_idx = maybe_shuffle(choices, correct_idx)
            samples.append({
                "input": mc_tmpl.format(
                    question=question, choices_text=format_choices(choices)),
                "correct": chr(65 + correct_idx),
                "level": None,
                "question": question,
                "choices": choices,
            })

        if task == "mmlu":
            ds = load_from_disk(f"{hf}/datasets/cais__mmlu__all")["test"]
            stride = max(1, len(ds) // 600)  # test is subject-sorted -> stride
            for item in ds.select(range(0, len(ds), stride)):
                mc(item["question"], item["choices"], item["answer"])
        elif task == "piqa":
            # piqa test split has label=-1; validation is the labeled one
            ds = load_from_disk(f"{hf}/datasets/ybisk__piqa__plain_text")["validation"]
            for item in ds:
                mc(item["goal"], [item["sol1"], item["sol2"]], item["label"])
        elif task.startswith("agieval_"):
            sub = task[len("agieval_"):]
            ds = load_from_disk(f"{hf}/datasets/baber__agieval__{sub}")["test"]
            if sub != "sat_math":
                # 5-option subsets need the letter-agnostic instruction; sat_math
                # keeps the original template for comparability with W9 results
                mc_tmpl = cot_mc if protocol == "cot" else answer_letter_any
            for item in ds:
                opts = [re.sub(r"^\([A-E]\)\s*", "", o) for o in item["options"]]
                mc(item["question"], opts, ord(item["label"]) - 65)
        elif task == "truthfulqa":
            # mc1_targets lists the correct answer FIRST -> take 1 correct +
            # up to 3 random wrong, shuffle (deterministic) to kill position bias
            ds = load_from_disk(
                f"{hf}/datasets/truthfulqa__truthful_qa__multiple_choice")["validation"]
            for item in ds:
                t = item["mc1_targets"]
                correct = [c for c, l in zip(t["choices"], t["labels"]) if l == 1][0]
                wrong = [c for c, l in zip(t["choices"], t["labels"]) if l == 0]
                opts = [correct] + rng.sample(wrong, min(3, len(wrong)))
                rng.shuffle(opts)
                mc(item["question"], opts, opts.index(correct))
        elif task == "hellaswag":
            ds = load_from_disk(f"{hf}/datasets/rowan__hellaswag")["validation"]
            for item in ds:
                mc("Choose the most plausible continuation.\nContext: " + item["ctx"],
                   item["endings"], int(item["label"]))
        elif task == "openbookqa":
            ds = load_from_disk(f"{hf}/datasets/allenai__openbookqa__main")["test"]
            for item in ds:
                ch = item["choices"]
                mc(item["question_stem"], ch["text"], ch["label"].index(item["answerKey"]))
        elif task == "commonsense_qa":
            ds = load_from_disk(f"{hf}/datasets/tau__commonsense_qa")["validation"]
            for item in ds:
                ch = item["choices"]
                mc(item["question"], ch["text"], ch["label"].index(item["answerKey"]))
        elif task == "boolq":
            ds = load_from_disk(f"{hf}/datasets/google__boolq")["validation"]
            for item in ds:
                mc(f"Passage: {item['passage']}\nQuestion: {item['question']}",
                   ["No", "Yes"], int(item["answer"]))
        elif task == "winogrande":
            ds = load_from_disk(f"{hf}/datasets/allenai__winogrande__xl")["validation"]
            for item in ds:
                mc("Fill in the blank (_) with the correct option.\nSentence: "
                   + item["sentence"], [item["option1"], item["option2"]],
                   int(item["answer"]) - 1)
        else:  # gsm8k
            ds = load_from_disk(f"{hf}/datasets/openai__gsm8k__main")["test"]
            for item in ds:
                gt = item["answer"].split("####")[-1].strip().replace(",", "")
                samples.append({
                    "input": ("Question: {q}\nGive ONLY the final numeric answer, "
                              "no explanation.").format(q=item["question"]),
                    "correct": gt,
                    "level": None,
                })
    else:
        cfg = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge"}[task]
        ds = load_dataset("allenai/ai2_arc", cfg, split="test")
        arc_tmpl = cot_mc if protocol == "cot" else answer_letter_long
        for item in ds:
            choices = item["choices"]
            labels = choices["label"]
            correct_idx = labels.index(item["answerKey"])
            ch_texts, correct_idx = maybe_shuffle(list(choices["text"]), correct_idx)
            samples.append({
                "input": arc_tmpl.format(
                    question=item["question"],
                    choices_text=format_choices(ch_texts)),
                "correct": chr(65 + correct_idx),
                "level": None,
                "question": item["question"],
                "choices": ch_texts,
            })
    return samples


GSM_NUM_RE = re.compile(r"-?\d*\.?\d+")

# MC-CoT: the final answer is the LAST "Answer: X" in the response (reasoning text
# may mention option letters, so first-match / anywhere-match would false-positive)
ARC_COT_RE = re.compile(r"Answer:\s*\(?([A-Da-d])\)?")


def grade(task, response, correct, protocol="short"):
    if task.startswith("dart"):
        pred = extract_boxed_content(response.strip())
        return float(grade_answer(pred, correct))
    if task == "gsm8k":
        nums = GSM_NUM_RE.findall(response.replace(",", "").replace("$", ""))
        if not nums:
            return 0.0
        try:
            return float(abs(float(nums[-1]) - float(correct)) < 1e-4)
        except ValueError:
            return 0.0
    if protocol == "cot":
        ms = ARC_COT_RE.findall(response)
        return float(bool(ms) and correct.lower() == ms[-1].lower())
    m = ARC_RE.match(response.strip())
    if not m:
        return 0.0
    return float(correct.lower() == m.group(1).strip().lower())


ARC_ANY_RE = re.compile(r"(?:Answer:\s*|\b)([A-Ea-e])\b")


def grade_lenient(task, response, correct, protocol="short"):
    """Cross-model-fair grading: verbose styles like 'C Answer: C' still count.
    The strict grade matches the MCTS-reward protocol; report both."""
    if task.startswith("dart") or task == "gsm8k":
        return grade(task, response, correct, protocol)
    if protocol == "cot":
        ms = ARC_COT_RE.findall(response)
        if ms:
            return float(correct.lower() == ms[-1].lower())
        # fallback: last standalone letter in the last non-empty line
        lines = [l for l in response.strip().splitlines() if l.strip()]
        tail = lines[-1] if lines else ""
        m2 = re.findall(r"\b([A-Da-d])\b", tail)
        return float(bool(m2) and correct.lower() == m2[-1].lower())
    m = ARC_ANY_RE.search(response.strip())
    if not m:
        return 0.0
    return float(correct.lower() == m.group(1).lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="stock snapshot (baseline) or router ckpt (routed)")
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--task", required=True,
                    help="arc_easy|arc_challenge|dart|dart_pool|mmlu|piqa|truthfulqa|"
                         "gsm8k|agieval_<subset> (subsets: sat_math, logiqa, lsat_lr, "
                         "lsat_ar, lsat_rc, sat_en, sat_en_wop, aqua_rat)")
    ap.add_argument("--mode", required=True, choices=["baseline", "routed"])
    ap.add_argument("--router_penalty", type=float, default=-0.5)
    ap.add_argument("--num_windows", type=int, default=8)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip first K samples before --limit (dev/test split: "
                         "0-299 = dev used for selection, 300+ = clean test)")
    ap.add_argument("--protocol", choices=["short", "cot"], default="short")
    ap.add_argument("--cot_max_new", type=int, default=512,
                    help="generation budget under cot protocol; thinking-native "
                         "models need 2048+ to finish reasoning before the answer")
    ap.add_argument("--base_prompt", action="store_true",
                    help="raw-text prompt + 'Answer:' cue for base (non-instruct) "
                         "models; grade via acc_lenient (strict letter format "
                         "not expected from base models)")
    ap.add_argument("--subset_per_level", type=int, default=0)
    ap.add_argument("--static_skip", default="",
                    help="baseline mode + data_models drop-in: comma-separated layer ids "
                         "to statically skip (ShortGPT-style control), e.g. '20,23,25'")
    ap.add_argument("--zero_sublayer", default="",
                    help="lesion localization within a block: 'attn:20' / 'mlp:20' / "
                         "'attn:20,mlp:20' zeroes that sublayer's output (residual "
                         "passthrough survives). Works on stock modeling code.")
    ap.add_argument("--scale_sublayer", default="",
                    help="toxicity dose-response: 'mlp:20:0.5' scales that sublayer's "
                         "output by alpha (0.0 == zero_sublayer; 1.0 == stock). "
                         "GPT-advice v3 P0: is the MLP write toxic continuously?")
    ap.add_argument("--mc_loglik", action="store_true",
                    help="format-artifact control (GPT-advice P0): no generation; "
                         "score each choice letter's logprob at the answer position "
                         "(acc_micro) and each option text's mean logprob (acc_lenient "
                         "column reused). MC tasks only.")
    ap.add_argument("--shuffle_choices", type=int, default=-1,
                    help="label-bias control: reshuffle option order with this seed "
                         "(>=0 enables; correct letter recomputed). MC tasks only.")
    ap.add_argument("--device_map_auto", action="store_true",
                    help="multi-GPU sharding for 70B-class models (device_map=auto)")
    ap.add_argument("--mm_model", action="store_true",
                    help="multimodal checkpoint (e.g. gemma-3 -it): load via "
                         "AutoModelForImageTextToText; text-only eval works as usual")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16",
                    help="bf16 for overflow-prone families (GLM-4 '!!!!' / gemma)")
    ap.add_argument("--no_think", action="store_true",
                    help="disable thinking across template dialects (SmolLM3 "
                         "enable_thinking, Seed-OSS thinking_budget; qwen3 is auto)")
    ap.add_argument("--n_shot", type=int, default=0,
                    help="prepend K solved examples (taken from samples[0:K], i.e. the "
                         "dev head — no overlap with offset>=K eval segments). Does "
                         "few-shot, like CoT, provide an alternative realization path "
                         "that removes the harmful-layer gain?")
    args = ap.parse_args()
    assert args.protocol == "short" or args.task in COT_TASKS, \
        f"cot protocol supports {sorted(COT_TASKS)}"

    import transformers
    print(f"transformers from: {transformers.__file__}")
    from transformers import AutoModelForCausalLM

    routed = args.mode == "routed"
    if args.protocol == "cot":
        max_new = args.cot_max_new
    else:
        max_new = 15 if args.task.startswith("dart") else 10

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tok.chat_template is None:
        # mistral-native / processor-style checkpoints keep the template in a
        # standalone file the tokenizer loader ignores (zskj Mistral-Small-3.1)
        for cand in ("chat_template.json", "chat_template.jinja"):
            p = os.path.join(args.tokenizer_path, cand)
            if os.path.exists(p):
                raw = open(p).read()
                tok.chat_template = json.loads(raw)["chat_template"] if cand.endswith(".json") else raw
                print(f"[tok] chat_template loaded from {cand}")
                break
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if args.mm_model:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, torch_dtype=dtype,
            device_map="auto" if args.device_map_auto else {"": "cuda:0"})
        num_layers = model.config.text_config.num_hidden_layers
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype,
            device_map="auto" if args.device_map_auto else {"": "cuda:0"})
        num_layers = model.config.num_hidden_layers
    model.eval()

    layer_counter = {"n": 0}
    if routed:
        model.model.router_penalty = args.router_penalty
        model.model.num_windows = args.num_windows
        assert hasattr(model.model, "routers"), "routed mode but drop-in not active"
        orig_fwd = model.model.forward

        def counting_fwd(*a, **k):
            out = orig_fwd(*a, **k)
            layer_counter["n"] += sum(out.decisions)  # 0=skip,1=exec,2=repeat
            return out
        model.model.forward = counting_fwd
    else:
        assert not hasattr(model.model, "routers"), "baseline mode but drop-in active"
        if args.static_skip:
            drop = {int(x) for x in args.static_skip.split(",")}
            assert hasattr(model.model, "layer_indices"), \
                "static_skip needs the data_models drop-in (layer_indices)"
            model.model.layer_indices = [l for l in range(num_layers) if l not in drop]
            print(f"static skip: dropping layers {sorted(drop)}")
        def _decoder_layers(m):
            # text models: model.model.layers; mm wrappers (gemma-3/4-it, Mistral3):
            # the text stack hides one level down (language_model / text_model)
            for path in ("model", "model.language_model", "model.text_model",
                         "language_model.model", "language_model"):
                obj = m
                try:
                    for att in path.split("."):
                        obj = getattr(obj, att)
                    return obj.layers
                except AttributeError:
                    continue
            raise AttributeError(f"cannot locate decoder layers on {type(m).__name__}")
        if args.zero_sublayer:
            def _zero_hook(mod, inp, out):
                if isinstance(out, tuple):
                    return (torch.zeros_like(out[0]),) + tuple(out[1:])
                return torch.zeros_like(out)
            for spec in args.zero_sublayer.split(","):
                kind, lid = spec.split(":")
                layer = _decoder_layers(model)[int(lid)]
                target = layer.self_attn if kind == "attn" else layer.mlp
                target.register_forward_hook(_zero_hook)
                print(f"zeroing {kind} output of layer {lid}")
        if args.scale_sublayer:
            def _make_scale_hook(alpha):
                def _hook(mod, inp, out):
                    if isinstance(out, tuple):
                        return (out[0] * alpha,) + tuple(out[1:])
                    return out * alpha
                return _hook
            for spec in args.scale_sublayer.split(","):
                kind, lid, alpha = spec.split(":")
                layer = _decoder_layers(model)[int(lid)]
                target = layer.self_attn if kind == "attn" else layer.mlp
                target.register_forward_hook(_make_scale_hook(float(alpha)))
                print(f"scaling {kind} output of layer {lid} by {alpha}")

    MC_TASKS = {"arc_easy", "arc_challenge", "mmlu", "piqa", "truthfulqa",
                "hellaswag", "openbookqa", "commonsense_qa", "boolq", "winogrande"}
    if args.task.startswith("agieval_"):
        MC_TASKS.add(args.task)
    if args.mc_loglik or args.shuffle_choices >= 0:
        assert args.task in MC_TASKS, "mc_loglik/shuffle_choices are MC-task-only"
        assert args.protocol == "short" and not routed
    samples = load_task(args.task, args.protocol, args.subset_per_level, args.shuffle_choices)
    shot_prefix = ""
    if args.n_shot:
        assert args.task in MC_TASKS and args.offset >= args.n_shot, "shots must not overlap eval"
        shot_prefix = "\n\n".join(
            x["input"] + f"\nAnswer: {x['correct']}" for x in samples[: args.n_shot]) + "\n\n"
    if args.offset:
        samples = samples[args.offset:]
    if args.limit:
        samples = samples[: args.limit]
    print(f"{args.task} {args.mode} p={args.router_penalty if routed else '-'} n={len(samples)}")

    records = []
    t0 = time.time()
    tmpl_kw = {}
    if "qwen3" in args.model_path.lower():
        tmpl_kw["enable_thinking"] = False  # short-answer protocol; thinking blows the token budget
    if args.no_think:
        tmpl_kw["enable_thinking"] = False
        tmpl_kw["thinking_budget"] = 0  # Seed-OSS dialect; unknown kwargs are ignored by jinja
    # Base (non-instruct) tokenizers carry no chat_template; apply_chat_template()
    # then raises ValueError and — under a "FAILED ... (continuing)" runner — silently
    # yields an empty result dir with a Succeeded job (the 07-10 Llama-base fleet
    # accident). Fall back to the raw base-prompt path whenever no template exists.
    # For chat models tok.chat_template is not None (native, or loaded above from a
    # standalone chat_template.{json,jinja}), so use_base_prompt == args.base_prompt
    # EXACTLY -> byte-identical behavior to the hundreds of already-run chat jobs.
    # This only ever changes the code path for models that previously crashed.
    use_base_prompt = args.base_prompt or tok.chat_template is None
    if use_base_prompt and not args.base_prompt:
        print(f"[tok] no chat_template found -> auto base_prompt fallback for "
              f"{args.model_path}", flush=True)
    def cont_logprob(prompt_ids, cont_text):
        """sum & mean logprob of cont_text tokens given prompt_ids (no generation)."""
        cont_ids = tok(cont_text, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(model.device)
        ids = torch.cat([prompt_ids, cont_ids], dim=1)
        with torch.no_grad():
            logits = model(ids).logits
        lp = torch.log_softmax(logits[0, prompt_ids.shape[1] - 1: -1].float(), -1)
        tok_lp = lp.gather(1, cont_ids[0].unsqueeze(1)).squeeze(1)
        return tok_lp.sum().item(), tok_lp.mean().item()

    for i, s in enumerate(samples):
        if use_base_prompt:
            prompt = shot_prefix + s["input"] + "\nAnswer:"
        else:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": shot_prefix + s["input"]}],
                tokenize=False, add_generation_prompt=True, **tmpl_kw)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]
        layer_counter["n"] = 0

        if args.mc_loglik:
            n_opt = len(s["choices"])
            # view 1 (letter): same prompt as generation, score "Answer: <L>" (equal
            # token length across letters -> sum logprob comparable). base_prompt
            # already ends with "\nAnswer:" so continue with just the letter.
            letter_cont = " {L}" if use_base_prompt else "Answer: {L}"
            letter_lp = [cont_logprob(inputs.input_ids, letter_cont.format(L=chr(65+j)))[0]
                         for j in range(n_opt)]
            pick_letter = chr(65 + max(range(n_opt), key=lambda j: letter_lp[j]))
            # view 2 (text): raw-LM view, no chat template, no letters in prompt;
            # mean logprob of the option text (length-normalized, lm-eval style)
            p2 = tok(f"Question: {s['question']}\nAnswer:",
                     return_tensors="pt").input_ids.to(model.device)
            text_lp = [cont_logprob(p2, " " + s["choices"][j])[1] for j in range(n_opt)]
            pick_text = chr(65 + max(range(n_opt), key=lambda j: text_lp[j]))
            per_fwd = len(model.model.layer_indices) if args.static_skip else num_layers
            records.append({"acc": float(pick_letter == s["correct"]),
                            "acc_len": float(pick_text == s["correct"]),
                            "level": s["level"], "layers": per_fwd,
                            "response": f"{pick_letter}|{pick_text}",
                            "correct": s["correct"]})
            if i < 3:
                print(f"[sanity {i}] letter={pick_letter} text={pick_text} gold={s['correct']}")
            if (i + 1) % 250 == 0:
                done = i + 1
                print(f"{done}/{len(samples)} letter_acc="
                      f"{sum(r['acc'] for r in records)/done:.4f} "
                      f"({(time.time()-t0)/done:.2f}s/sample)", flush=True)
            continue
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.eos_token_id,
                min_length=input_len + 2,
                use_cache=not routed,
            )
        new_tokens = out[0][input_len:]
        response = tok.decode(new_tokens, skip_special_tokens=True).strip()
        n_new = max(len(new_tokens), 1)  # == number of forwards with cache (prefill + decodes)
        # layers actually executed: routed -> counted from decisions; baseline with
        # cache -> one (possibly statically pruned) pass per generated token
        per_fwd = len(model.model.layer_indices) if (not routed and args.static_skip) else num_layers
        layers_used = layer_counter["n"] if routed else per_fwd * n_new
        acc = grade(args.task, response, s["correct"], args.protocol)
        records.append({"acc": acc, "acc_len": grade_lenient(args.task, response, s["correct"], args.protocol),
                        "level": s["level"], "layers": layers_used,
                        "response": response, "correct": s["correct"]})
        if i < 3:
            print(f"[sanity {i}] acc={acc} layers={layers_used} resp={response[:60]!r}")
        if (i + 1) % 250 == 0:
            done = i + 1
            cur = sum(r["acc"] for r in records) / done
            print(f"{done}/{len(samples)} acc={cur:.4f} "
                  f"({(time.time()-t0)/done:.2f}s/sample)", flush=True)

    n = len(records)
    summary = {
        "task": args.task, "mode": args.mode, "protocol": args.protocol,
        "router_penalty": args.router_penalty if routed else None,
        "n": n,
        "acc_micro": sum(r["acc"] for r in records) / n,
        "acc_lenient": sum(r["acc_len"] for r in records) / n,
        "avg_layers": sum(r["layers"] for r in records) / n,
    }
    if args.task.startswith("dart"):
        per_level = {}
        for lv in range(1, 6):
            rs = [r for r in records if r["level"] == lv]
            per_level[lv] = {"n": len(rs), "acc": sum(r["acc"] for r in rs) / len(rs)}
        summary["per_level"] = per_level
        summary["acc_macro_levels"] = sum(v["acc"] for v in per_level.values()) / 5
    tag = f"_p{args.router_penalty}" if routed else ""
    if args.static_skip:
        tag += "_static" + args.static_skip.replace(",", "-")
        summary["static_skip"] = args.static_skip
    if args.zero_sublayer:
        tag += "_z" + args.zero_sublayer.replace(":", "").replace(",", "-")
        summary["zero_sublayer"] = args.zero_sublayer
    if args.scale_sublayer:
        tag += "_s" + args.scale_sublayer.replace(":", "").replace(",", "-").replace(".", "p")
        summary["scale_sublayer"] = args.scale_sublayer
    if args.mc_loglik:
        tag += "_mcll"
        summary["mc_loglik"] = True
        summary["note"] = "acc_micro=letter logprob (gen prompt); acc_lenient=option-text mean logprob (raw LM prompt)"
    if args.shuffle_choices >= 0:
        tag += f"_shuf{args.shuffle_choices}"
        summary["shuffle_choices"] = args.shuffle_choices
    if args.n_shot:
        tag += f"_fs{args.n_shot}"
        summary["n_shot"] = args.n_shot
    if args.protocol != "short":
        tag += f"_{args.protocol}"
    if args.offset:
        tag += f"_off{args.offset}"
        summary["offset"] = args.offset
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.task}_{args.mode}{tag}.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=1)
    print("SUMMARY", json.dumps(summary, indent=1))
    print("saved", out_path)


if __name__ == "__main__":
    main()
