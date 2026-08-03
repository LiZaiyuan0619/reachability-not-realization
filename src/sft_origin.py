"""CONTROLLED-ORIGIN pilot (W12): which post-training FORMAT manufactures the
recognition->production (R>G) scar?

Two arms trained from the SAME healthy base (Qwen2.5-3B-Instruct; the w11 census
shows G~R there, i.e. a clean negative control) on the SAME questions (ARC-Easy
train + CommonsenseQA train), differing ONLY in the answer format:

  --arm mc      MC-format SFT: question + lettered options -> "Answer: <letter>"
                (the exact answer_letter_long protocol eval_drllm.py grades; CSQA
                keeps that template's "(A, B, C, or D)" wording even with 5 options
                because eval_drllm.py does the same -- protocol match > prose).
  --arm direct  direct-answer SFT: same question WITHOUT options -> the gold answer
                TEXT. The "Answer: " envelope is kept in BOTH arms so the only
                manipulated variable is letter-pointing vs content production.
  --arm mixed   dose-response (W12b): requires --mix_mc <p in 0..1>. A fraction p
                of the SAME shuffled list is mc-formatted, the rest direct, via a
                DETERMINISTIC NESTED rule (see arms_for_records): raising the dose
                only converts additional direct records to mc, never re-samples
                which records are mc. mix_mc=0 reproduces --arm direct exactly and
                mix_mc=1 reproduces --arm mc exactly (selftest proves both).

Control = the untrained base (already in the w11_gap census). Both arms shuffle the
pooled records with the SAME seed, so they see the IDENTICAL question sequence.

LoRA r=16 alpha=32 (peft, fp32 adapters on a bf16 base), ~800 steps batch 8 lr 1e-4,
single GPU, ~15-20 min on an H200. Saves BOTH the adapter and a merged STOCK-format
HF checkpoint that eval_drllm.py loads as-is:

  <out_dir>/adapter/            peft adapter
  <out_dir>/merged/             standard HF model + tokenizer (--model_path/--tokenizer_path)
  <out_dir>/train_manifest.json args + data stats + final loss (worker idempotency key)

  python sft_origin.py --arm mc     --steps 800
  python sft_origin.py --arm direct --steps 800
  python sft_origin.py --arm mixed  --mix_mc 0.25 --steps 800 --lr 2e-5
  python sft_origin.py --selftest        # no model / no GPU: data + masking + argparse
"""
import argparse
import json
import math
import os
import random
import time

# offline discipline BEFORE any hf import (never download; dev and worker both
# read the shared cache)
os.environ.setdefault("HF_HOME", "models/cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from prompts import answer_letter_long, format_choices

DEFAULT_BASE = ("models/Qwen2.5-3B-Instruct/"
                "snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
DEFAULT_OUT_ROOT = "models/sft_origin"

# direct arm: parallel construction to answer_letter_long (same envelope, no options)
answer_direct = """"Question: {question}
Answer with a short answer and no explanation. You answer should start with "Answer: " and be followed by the answer. Do not include any other text in your response."""


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["mc", "direct", "mixed"],
                    help="training format (required unless --selftest); "
                         "'mixed' requires --mix_mc")
    ap.add_argument("--mix_mc", type=float, default=None,
                    help="dose in [0,1]: fraction of records mc-formatted (rest "
                         "direct), nested-deterministic across doses (see "
                         "arms_for_records). Only valid with --arm mixed; default "
                         "None keeps the pure --arm behavior (backward compatible)")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="EARLY-STOP cap: run at most this many optimizer steps while "
                         "keeping the LR-schedule horizon at --steps (0 = off = train the "
                         "full --steps; every existing call unchanged). Because data order "
                         "and lr_at() both key on --steps/seed, `--steps 800 --max_steps "
                         "700` saves the BYTE-IDENTICAL step-700 model of the 800-step run "
                         "-- the healthy converged ckpt just before the deterministic "
                         "step-701 poison-batch NaN (harvest-B). Reducing --steps instead "
                         "would re-anneal the cosine and change that model.")
    ap.add_argument("--out_dir", default="",
                    help=f"default: {DEFAULT_OUT_ROOT}/<arm>")
    ap.add_argument("--model_path", default=DEFAULT_BASE)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max_len", type=int, default=640,
                    help="examples LONGER than this are DROPPED (never truncated) so "
                         "both arms keep the identical question sequence; selftest "
                         "proves 0 drops at the default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--limit_data", type=int, default=0, help="per-source cap (smoke)")
    ap.add_argument("--selftest", action="store_true",
                    help="validate data construction + loss masking + argparse; "
                         "loads the TOKENIZER only (no model, no GPU)")
    return ap


def validate_args(args):
    """--arm mixed <=> --mix_mc given; the dose must lie in [0,1]."""
    if args.mix_mc is not None:
        if args.arm != "mixed":
            raise ValueError("--mix_mc is only valid with --arm mixed "
                             "(mc/direct are pure-format arms)")
        if not 0.0 <= args.mix_mc <= 1.0:
            raise ValueError(f"--mix_mc must be in [0,1], got {args.mix_mc}")
    elif args.arm == "mixed":
        raise ValueError("--arm mixed requires --mix_mc <float in [0,1]>")


# --------------------------------------------------------------------- data
def build_records(limit=0):
    """Canonical (question, choices, correct_idx) records; ARC-Easy + CSQA train.
    Arms are formatted FROM this shared list, so the question set/order is identical."""
    from datasets import load_dataset, load_from_disk
    hf = os.environ["HF_HOME"]
    recs = []
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    for item in ds:
        labels = list(item["choices"]["label"])   # letters OR "1"-"4"; index() remaps
        if item["answerKey"] not in labels:
            continue
        recs.append({"source": "arc_easy", "question": item["question"],
                     "choices": list(item["choices"]["text"]),
                     "correct_idx": labels.index(item["answerKey"])})
    n_arc = len(recs)
    csqa = load_from_disk(f"{hf}/datasets/tau__commonsense_qa")["train"]
    for item in csqa:
        labels = list(item["choices"]["label"])
        if item["answerKey"] not in labels:
            continue
        recs.append({"source": "commonsense_qa", "question": item["question"],
                     "choices": list(item["choices"]["text"]),
                     "correct_idx": labels.index(item["answerKey"])})
    if limit:
        arc = [r for r in recs if r["source"] == "arc_easy"][:limit]
        cs = [r for r in recs if r["source"] == "commonsense_qa"][:limit]
        recs, n_arc = arc + cs, len(arc)
    return recs, n_arc


def format_example(rec, arm):
    """-> (user_message, assistant_target). Envelope ("Answer: ") identical across
    arms; mc points at an option letter, direct produces the gold answer text."""
    if arm == "mc":
        user = answer_letter_long.format(
            question=rec["question"], choices_text=format_choices(rec["choices"]))
        target = f"Answer: {chr(65 + rec['correct_idx'])}"
    else:
        user = answer_direct.format(question=rec["question"])
        target = f"Answer: {rec['choices'][rec['correct_idx']]}"
    return user, target


def vdc_frac(i):
    """Base-2 van der Corput radical inverse of i, in [0,1): the bit-reversed
    fraction. vdc(0,1,2,3,4,5,6,7) = 0, .5, .25, .75, .125, .625, .375, .875.
    Every term is a negative power of 2, so the sum is float-exact for any
    realistic i (< 2**53)."""
    f, x = 0.0, 0.5
    while i:
        if i & 1:
            f += x
        i >>= 1
        x *= 0.5
    return f


def arms_for_records(n, arm, mix_mc):
    """Per-record format list over the SHARED seed-shuffled record list.

    NESTING RULE (the controlled-dose key): record at position i is mc-formatted
    iff vdc_frac(i) < mix_mc. Because vdc_frac(i) is a FIXED per-record score and
    the dose is a threshold on it, {i: vdc(i) < p1} is a subset of
    {i: vdc(i) < p2} whenever p1 <= p2 -- raising the dose only CONVERTS
    additional direct records to mc, it never re-samples which records are mc
    (mix=0.25's mc set is strictly inside mix=0.75's). So a dose-response curve
    across mix values compares nested training sets, and format-assignment
    randomness cannot masquerade as a dose effect. This generalizes the
    fixed-modulus idea (`i % 4 < 3` is nested only across thresholds sharing one
    modulus) to ANY real threshold in [0,1], and vdc is low-discrepancy: every
    contiguous chunk of the list carries ~mix_mc fraction mc (count error
    O(log2 n)), so the realized dose is uniform through the data, not a prefix
    block. Edge cases are exact: mix_mc=0 -> all direct (vdc >= 0 always),
    mix_mc=1 -> all mc (vdc < 1 always); with the shuffle/seed/order machinery
    unchanged, those reproduce the pure arms byte-for-byte (selftest [4/5])."""
    if arm != "mixed":
        return [arm] * n
    return ["mc" if vdc_frac(i) < mix_mc else "direct" for i in range(n)]


def encode_example(tok, user, target, max_len):
    """Chat-template prompt ids + target ids (+eos), loss masked to the target.
    Prompt ids come from apply_chat_template(add_generation_prompt=True) -- the EXACT
    conditioning eval_drllm.py generates from -- and target ids are appended to them,
    so train-time next-token supervision matches inference-time tokenization."""
    prompt = tok.apply_chat_template([{"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    t_ids = tok(target, add_special_tokens=False).input_ids + [tok.eos_token_id]
    if len(p_ids) + len(t_ids) > max_len:
        return None  # drop (never truncate) -> arms stay sequence-aligned
    return p_ids + t_ids, [-100] * len(p_ids) + t_ids


def encode_all(tok, recs, arms, max_len):
    """arms: per-record format list (arms_for_records), aligned with recs."""
    enc, dropped = [], 0
    for r, a in zip(recs, arms):
        e = encode_example(tok, *format_example(r, a), max_len)
        if e is None:
            dropped += 1
        else:
            enc.append(e)
    return enc, dropped


def collate(items, pad_id):
    import torch
    n = max(len(ids) for ids, _ in items)
    input_ids = torch.full((len(items), n), pad_id, dtype=torch.long)
    labels = torch.full((len(items), n), -100, dtype=torch.long)
    attn = torch.zeros((len(items), n), dtype=torch.long)
    for i, (ids, lab) in enumerate(items):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
        attn[i, : len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


# ----------------------------------------------------------------- selftest
def run_selftest(args):
    from transformers import AutoTokenizer
    print("== sft_origin selftest (tokenizer only; no model, no GPU) ==")

    # 1) argparse round-trip + mix contract
    t = build_parser().parse_args(
        ["--arm", "mc", "--steps", "500", "--out_dir", "/tmp/x", "--lr", "2e-5"])
    assert (t.arm, t.steps, t.out_dir, t.lr) == ("mc", 500, "/tmp/x", 2e-5)
    t2 = build_parser().parse_args(["--arm", "direct"])
    assert t2.steps == 800 and t2.batch_size == 8 and t2.lr == 1e-4
    assert t2.mix_mc is None  # backward compatible: pure arms carry no dose
    assert t2.max_steps == 0  # early-stop cap defaults off (full --steps trajectory)
    assert build_parser().parse_args(
        ["--arm", "mc", "--steps", "800", "--max_steps", "700"]).max_steps == 700
    validate_args(t)
    validate_args(t2)
    t3 = build_parser().parse_args(["--arm", "mixed", "--mix_mc", "0.75"])
    assert t3.arm == "mixed" and t3.mix_mc == 0.75
    validate_args(t3)
    for bad in (["--arm", "mixed"],                       # mixed without dose
                ["--arm", "mc", "--mix_mc", "0.5"],       # dose on a pure arm
                ["--arm", "mixed", "--mix_mc", "1.5"]):   # dose out of range
        try:
            validate_args(build_parser().parse_args(bad))
            raise AssertionError(f"validate_args accepted {bad}")
        except ValueError:
            pass
    print("[1/5] argparse OK (defaults: steps=800 batch=8 lr=1e-4; mix_mc=None; "
          "mixed<=>mix_mc contract enforced)")

    # 2) data construction
    recs, n_arc = build_records(args.limit_data)
    n_csqa = len(recs) - n_arc
    print(f"[2/5] records: total={len(recs)} arc_easy={n_arc} commonsense_qa={n_csqa}")
    if not args.limit_data:
        assert n_arc >= 2000 and n_csqa >= 9000, "unexpected dataset sizes"
    rng = random.Random(args.seed)
    rng.shuffle(recs)

    # 3) per-arm formatting invariants on the SAME shuffled list
    r0 = recs[0]
    for r in recs:
        u_mc, t_mc = format_example(r, "mc")
        u_di, t_di = format_example(r, "direct")
        ch_block = format_choices(r["choices"])
        assert ch_block in u_mc and r["question"] in u_mc
        assert ch_block not in u_di and r["question"] in u_di
        assert t_mc == f"Answer: {chr(65 + r['correct_idx'])}"
        assert t_di == f"Answer: {r['choices'][r['correct_idx']]}"
    print("[3/5] formatting OK: mc has options + letter target; direct has neither; "
          "'Answer: ' envelope identical; question order shared (seeded shuffle)")

    # 4) mix dose: nested deterministic assignment (no re-sampling across doses)
    n = len(recs)
    assert [vdc_frac(i) for i in range(8)] == [0.0, 0.5, 0.25, 0.75,
                                               0.125, 0.625, 0.375, 0.875]
    assert arms_for_records(n, "mc", None) == ["mc"] * n
    assert arms_for_records(n, "direct", None) == ["direct"] * n
    assert arms_for_records(n, "mixed", 0.0) == ["direct"] * n, "mix=0 must == direct"
    assert arms_for_records(n, "mixed", 1.0) == ["mc"] * n, "mix=1 must == mc"
    half = arms_for_records(n, "mixed", 0.5)
    n_mc_half = half.count("mc")
    assert abs(n_mc_half - n / 2) <= 1, "mix=0.5 must split the set in half"
    grid = [0.10, 0.25, 0.50, 0.75, 1.00]
    mc_sets = {p: {i for i, a in enumerate(arms_for_records(n, "mixed", p))
                   if a == "mc"} for p in grid}
    for lo, hi in zip(grid, grid[1:]):
        assert mc_sets[lo] < mc_sets[hi], f"nesting broken: {lo} not inside {hi}"
    for p in grid:
        c = len(mc_sets[p])
        assert abs(c - p * n) <= math.log2(n) + 2, f"dose {p}: count {c} off target"
        print(f"    mix_mc={p:.2f}: n_mc={c:5d} ({c/n:.4f} of {n})  n_direct={n-c}")
    print(f"[4/5] mixing OK: mix0==direct, mix1==mc; mix0.5 -> {n_mc_half} mc / "
          f"{n - n_mc_half} direct; grid {grid} chain-nested "
          "(each mc set is a strict subset of the next dose's)")

    # 5) tokenization / loss-mask invariants over the FULL set, both arms
    tok = AutoTokenizer.from_pretrained(args.model_path)
    assert tok.eos_token_id is not None
    for arm in ("mc", "direct"):
        t0, max_total, sup_tot, dropped = time.time(), 0, 0, 0
        n_ok = 0
        for r in recs:
            e = encode_example(tok, *format_example(r, arm), args.max_len)
            if e is None:
                dropped += 1
                continue
            ids, lab = e
            assert len(ids) == len(lab)
            n_sup = sum(1 for x in lab if x != -100)
            k = len(lab) - n_sup
            assert n_sup >= 3, "target must supervise >=2 tokens + eos"
            assert all(x == -100 for x in lab[:k])
            assert lab[k:] == ids[k:], "supervised labels must equal input ids"
            assert ids[-1] == tok.eos_token_id and lab[-1] == tok.eos_token_id
            max_total = max(max_total, len(ids))
            sup_tot += n_sup
            n_ok += 1
        assert dropped == 0, (f"{arm}: {dropped} examples exceed max_len={args.max_len}"
                              " -> arms would desync; raise --max_len")
        print(f"[5/5] {arm}: encoded {n_ok} ex, 0 dropped, max_len_seen={max_total}, "
              f"mean_supervised_tok={sup_tot/n_ok:.1f} ({time.time()-t0:.0f}s)")

    # mixed integration through the production encode_all path (small slice; every
    # (record, arm) pair was already fully validated in the two loops above)
    sl = recs[:200]
    arms_sl = arms_for_records(len(sl), "mixed", 0.5)
    enc_sl, dr_sl = encode_all(tok, sl, arms_sl, args.max_len)
    assert dr_sl == 0 and len(enc_sl) == len(sl)
    print(f"[5/5] mixed(0.5) encode_all on {len(sl)}-rec slice: "
          f"{arms_sl.count('mc')} mc + {arms_sl.count('direct')} direct, 0 dropped")

    for arm in ("mc", "direct"):
        u, t = format_example(r0, arm)
        print(f"\n---- sample [{arm}] user ----\n{u}\n---- target ----\n{t!r}")
    print("\nSELFTEST PASS")


# -------------------------------------------------------------------- train
def run_train(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    arm_tag = (args.arm if args.mix_mc is None
               else f"mixed_mc{round(args.mix_mc * 100)}")  # doses must not collide
    out_dir = args.out_dir or os.path.join(DEFAULT_OUT_ROOT, arm_tag)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    recs, n_arc = build_records(args.limit_data)
    rng = random.Random(args.seed)
    rng.shuffle(recs)  # same seed in both arms -> identical question sequence
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    arms = arms_for_records(len(recs), args.arm, args.mix_mc)
    n_mc_fmt = arms.count("mc")
    encoded, dropped = encode_all(tok, recs, arms, args.max_len)
    print(f"arm={args.arm} mix_mc={args.mix_mc} records={len(recs)} (arc={n_arc}) "
          f"formats: mc={n_mc_fmt} direct={len(recs) - n_mc_fmt} "
          f"encoded={len(encoded)} dropped={dropped}")
    assert dropped == 0, "drops would desync the two arms; raise --max_len"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"})
    lcfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    for p in model.parameters():        # fp32 adapters on the bf16 base (stable AdamW;
        if p.requires_grad:             # peft casts activations across dtypes itself)
            p.data = p.data.to(torch.float32)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    def lr_at(step):  # linear warmup -> cosine to 10%
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    model.train()
    order, pos = list(range(len(encoded))), len(encoded)  # force shuffle at step 0
    t0, win, tok_seen, final_loss = time.time(), [], 0, None
    steps_run = 0
    for step in range(args.steps):
        if args.max_steps and step >= args.max_steps:
            if win:  # flush any un-logged window so the manifest loss is real
                final_loss = sum(win) / len(win)
            print(f"[early-stop] --max_steps={args.max_steps} reached "
                  f"(LR horizon stays --steps={args.steps}); final_loss={final_loss}",
                  flush=True)
            break
        if pos + args.batch_size > len(order):
            rng.shuffle(order)
            pos = 0
        batch = collate([encoded[i] for i in order[pos: pos + args.batch_size]],
                        tok.pad_token_id)
        pos += args.batch_size
        batch = {k: v.to(model.device) for k, v in batch.items()}
        if step == 0:  # masking sanity in the production path
            lab0 = batch["labels"][0]
            sup = lab0[lab0 != -100]
            print(f"[step0 sanity] supervised span: {tok.decode(sup)!r}")
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = model(**batch, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        win.append(loss.item())
        tok_seen += int(batch["attention_mask"].sum())
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            el = time.time() - t0
            print(f"step {step+1}/{args.steps} loss={sum(win)/len(win):.4f} "
                  f"lr={lr_at(step):.2e} {tok_seen/el:.0f} tok/s {el:.0f}s", flush=True)
            final_loss, win = sum(win) / len(win), []
        steps_run = step + 1

    # ---- save: adapter, then merged STOCK checkpoint eval_drllm.py can load ----
    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    print("adapter saved ->", adapter_dir)
    model = model.to(torch.bfloat16)
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged_dir = os.path.join(out_dir, "merged")
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    print("merged stock checkpoint saved ->", merged_dir)

    manifest = {
        "arm": args.arm, "mix_mc": args.mix_mc,
        "mix_rule": ("vdc_base2_nested_threshold" if args.arm == "mixed" else None),
        "steps": args.steps, "max_steps": args.max_steps or None,
        "steps_run": steps_run, "batch_size": args.batch_size,
        "lr": args.lr, "warmup": args.warmup, "seed": args.seed,
        "max_len": args.max_len, "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
        "base_model": args.model_path,
        "data": {"n_records": len(recs), "n_arc_easy": n_arc,
                 "n_commonsense_qa": len(recs) - n_arc,
                 "n_mc_format": n_mc_fmt, "n_direct_format": len(recs) - n_mc_fmt,
                 "n_dropped": dropped,
                 "examples_seen": steps_run * args.batch_size},
        "final_loss_window": final_loss, "wall_sec": round(time.time() - t0, 1),
        "versions": {"torch": torch.__version__,
                     "peft": __import__("peft").__version__,
                     "transformers": __import__("transformers").__version__},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "train_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("MANIFEST", json.dumps(manifest, indent=1))


def main():
    args = build_parser().parse_args()
    if args.selftest:
        run_selftest(args)
        return
    assert args.arm, "--arm {mc,direct,mixed} is required unless --selftest"
    validate_args(args)
    run_train(args)


if __name__ == "__main__":
    main()
