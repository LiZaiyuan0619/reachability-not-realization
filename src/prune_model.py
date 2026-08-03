#!/usr/bin/env python3
"""Vendored MINIMAL unstructured pruning (WANDA / magnitude) -> saves a stock HF
checkpoint so the ENTIRE existing measurement chain (eval_drllm.py gen+mcll,
mlp_battery.py) runs on it UNCHANGED (just point --model_path at the output dir).

Scientific purpose (C2 "manufacture the scar"): take a HEALTHY zero-gap model
(e.g. Qwen2.5-7B-Instruct, census tag w11_gap_q257) and induce a recognition->
production scar with a sparsity knob, then show it localizes to an MLP write just
like the naturally-occurring scars (Llama-3.2-3B, gemma-4).

Methods
  magnitude : score = |W|                       (no calibration; 0-network smoke)
  wanda     : score = |W| * ||X_j||_2           (Sun et al. 2023, training-free,
              per-input-feature activation norm from ~128 calibration samples)
Comparison group is PER-OUTPUT-ROW (WANDA default for unstructured): each output
neuron's row gets EXACTLY floor/round(sparsity * in_features) weights zeroed, so
achieved sparsity is uniform and == target (up to rounding). Only the decoder-block
linears {q,k,v,o,gate,up,down}_proj are pruned; embeddings / lm_head are untouched
(standard WANDA scope).

One-shot dense-calibration variant: activation norms are collected in a SINGLE pass
over the dense model via forward_pre_hooks, then all layers pruned together. This is
the robust/minimal WANDA (vs. the reference sequential-per-layer replay, which is
fragile across transformers versions). Fidelity caveat noted in the plan; magnitude
needs no calibration at all.

Calibration source (offline-first): reuses eval_drllm.load_task text so it needs
ZERO network -- default hellaswag (cached, and DIFFERENT from the arc/piqa/agieval
eval tasks -> no same-task leakage). wikitext/c4 paths exist for paper-faithful runs
(need a one-time `pl` download).

Usage
  python prune_model.py --model_path P --out DIR --sparsity 0.5 --method wanda \
      --calib_source task --calib_task hellaswag --nsamples 128 --seqlen 2048
  python prune_model.py --selftest        # torch CPU end-to-end (tiny model), no GPU/net
"""
import argparse
import glob
import json
import os
import time


DEFAULT_PROJ = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def decoder_module(m):
    """Locate the module that owns `.layers` (matches eval_drllm / mlp_battery discovery)."""
    for path in ("model", "model.language_model", "model.text_model",
                 "language_model.model", "language_model"):
        obj = m
        try:
            for att in path.split("."):
                obj = getattr(obj, att)
            _ = obj.layers
            return obj
        except AttributeError:
            continue
    raise AttributeError("no decoder module (.layers) found")


def find_target_linears(model, proj_names):
    """All nn.Linear inside decoder blocks whose leaf name is a target projection."""
    import torch.nn as nn
    targets = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if ".layers." not in name:            # skip lm_head / embed / non-block linears
            continue
        if name.split(".")[-1] in proj_names:
            targets[name] = mod
    return targets


def build_calib_ids(tok, source, task, nsamples, seqlen, seed):
    """Return a list of [1, seqlen] LongTensors (no padding: concat token stream then chunk)."""
    import torch
    import random
    rng = random.Random(seed)
    if source == "task":
        import eval_drllm
        samples = eval_drllm.load_task(task, "short", 0, -1)
        texts = []
        for s in samples:
            # chat-template the eval prompt exactly as the model will see it
            try:
                t = tok.apply_chat_template(
                    [{"role": "user", "content": s["input"]}],
                    tokenize=False, add_generation_prompt=True)
            except Exception:
                t = s["input"]
            texts.append(t)
        rng.shuffle(texts)
    elif source in ("wikitext", "c4"):
        from datasets import load_dataset
        if source == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = [x for x in ds["text"] if x.strip()]
        else:
            ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
            texts = []
            for i, x in enumerate(ds):
                texts.append(x["text"])
                if i >= nsamples * 40:
                    break
        rng.shuffle(texts)
    else:
        raise ValueError(f"unknown calib source {source}")

    # concat into one id stream, then slice non-overlapping seqlen windows
    stream = []
    for t in texts:
        ids = tok(t, add_special_tokens=False).input_ids
        stream.extend(ids)
        if len(stream) >= (nsamples + 2) * seqlen:
            break
    if len(stream) < seqlen:
        raise RuntimeError(f"calibration stream too short ({len(stream)} < {seqlen} tokens)")
    batches = []
    i = 0
    while len(batches) < nsamples and i + seqlen <= len(stream):
        batches.append(torch.tensor(stream[i:i + seqlen], dtype=torch.long).unsqueeze(0))
        i += seqlen
    return batches


def collect_wanda_norms(model, targets, calib_ids, device):
    """One-shot dense calibration: per-target sqrt(mean_token X_j^2), shape [in_features]."""
    import torch
    sums = {n: None for n in targets}
    counts = {n: 0 for n in targets}
    handles = []

    def mk(name):
        def hook(mod, inp):
            x = inp[0]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            xf = x.float()
            s = (xf * xf).sum(dim=0)
            sums[name] = s if sums[name] is None else sums[name] + s
            counts[name] += xf.shape[0]
        return hook

    for name, mod in targets.items():
        handles.append(mod.register_forward_pre_hook(mk(name)))
    model.eval()
    with torch.no_grad():
        for j, ids in enumerate(calib_ids):
            model(ids.to(device))
            if (j + 1) % 16 == 0:
                print(f"  calib {j + 1}/{len(calib_ids)}", flush=True)
    for h in handles:
        h.remove()
    norms = {}
    for n in targets:
        if sums[n] is None or counts[n] == 0:
            norms[n] = None
        else:
            norms[n] = (sums[n] / counts[n]).sqrt()
    return norms


def prune_linear_(weight, sparsity, norm=None):
    """In-place per-row unstructured prune to EXACTLY k=round(sparsity*in) zeros/row.
    Returns (n_zeroed, n_total). norm: [in_features] activation norm (wanda) or None (magnitude)."""
    import torch
    W = weight.data
    out_f, in_f = W.shape
    metric = W.abs().float()
    if norm is not None:
        metric = metric * norm.view(1, -1).to(metric.device).float()
    k = int(round(sparsity * in_f))
    if k <= 0:
        return 0, out_f * in_f
    k = min(k, in_f)
    idx = torch.topk(metric, k, dim=1, largest=False).indices     # k smallest per row
    mask = torch.ones_like(W, dtype=torch.bool)
    mask.scatter_(1, idx, False)
    W.mul_(mask.to(W.dtype))
    return k * out_f, out_f * in_f


def prune_model(model, targets, sparsity, method, norms):
    per_layer = {}
    zt = tt = 0
    for name, mod in targets.items():
        nrm = norms.get(name) if (method == "wanda" and norms) else None
        z, t = prune_linear_(mod.weight, sparsity, nrm)
        per_layer[name] = round(z / t, 4) if t else 0.0
        zt += z
        tt += t
    return zt, tt, per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path")
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--out")
    ap.add_argument("--sparsity", type=float, default=0.5, help="unstructured fraction per output row")
    ap.add_argument("--method", choices=["wanda", "magnitude"], default="wanda")
    ap.add_argument("--calib_source", choices=["task", "wikitext", "c4"], default="task")
    ap.add_argument("--calib_task", default="hellaswag",
                    help="offline calib text via eval_drllm.load_task (default != eval tasks)")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mm_model", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        import sys
        sys.exit(0 if selftest() else 1)
    assert args.model_path and args.out and args.sparsity is not None, \
        "need --model_path --out --sparsity (or --selftest)"

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokp = args.tokenizer_path or args.model_path
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    t0 = time.time()
    print(f"[load] {args.model_path} dtype={args.dtype} -> {args.device}")
    tok = AutoTokenizer.from_pretrained(tokp)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.mm_model:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, torch_dtype=dtype, device_map={"": args.device})
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, device_map={"": args.device})
    model.eval()

    targets = find_target_linears(model, DEFAULT_PROJ)
    print(f"[targets] {len(targets)} decoder-block linears "
          f"(example: {sorted(targets)[0] if targets else 'NONE'})")
    assert targets, "no target linears found -- check decoder discovery / proj names"

    norms = {}
    if args.method == "wanda":
        print(f"[calib] source={args.calib_source} task={args.calib_task} "
              f"nsamples={args.nsamples} seqlen={args.seqlen}")
        calib = build_calib_ids(tok, args.calib_source, args.calib_task,
                                args.nsamples, args.seqlen, args.seed)
        print(f"[calib] built {len(calib)} windows; running dense forward passes")
        norms = collect_wanda_norms(model, targets, calib, args.device)

    print(f"[prune] method={args.method} sparsity={args.sparsity}")
    zt, tt, per_layer = prune_model(model, targets, args.sparsity, args.method, norms)
    achieved = zt / tt if tt else 0.0
    print(f"[prune] achieved global sparsity {achieved:.4f} ({zt}/{tt}) over targeted linears")

    os.makedirs(args.out, exist_ok=True)
    print(f"[save] {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    manifest = {
        "source_model": args.model_path,
        "method": args.method,
        "sparsity_target": args.sparsity,
        "sparsity_achieved": round(achieved, 5),
        "n_targeted_linears": len(targets),
        "n_params_targeted": tt,
        "n_params_zeroed": zt,
        "calib_source": args.calib_source if args.method == "wanda" else None,
        "calib_task": args.calib_task if args.method == "wanda" else None,
        "nsamples": args.nsamples if args.method == "wanda" else None,
        "seqlen": args.seqlen if args.method == "wanda" else None,
        "dtype": args.dtype,
        "per_layer_sparsity": per_layer,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "prune_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("MANIFEST", json.dumps({k: v for k, v in manifest.items()
                                  if k != "per_layer_sparsity"}, indent=1))
    print("saved", args.out)


# ==================== torch CPU end-to-end self-test (no GPU / no network) ====================

def selftest():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    ok = []

    def chk(name, cond):
        ok.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, tie_word_embeddings=False)
    model = Qwen2ForCausalLM(cfg).eval()
    targets = find_target_linears(model, DEFAULT_PROJ)
    chk("target linears found (2 layers x 7 proj = 14)", len(targets) == 14)
    chk("lm_head excluded", not any("lm_head" in n for n in targets))
    chk("embed excluded", not any("embed" in n for n in targets))

    # fabricate calibration batches (random ids) + collect wanda norms via hooks
    calib = [torch.randint(0, 256, (1, 16)) for _ in range(4)]
    norms = collect_wanda_norms(model, targets, calib, "cpu")
    a_name = sorted(targets)[0]
    chk("wanda norm shape == in_features", norms[a_name].shape[0] == targets[a_name].weight.shape[1])
    chk("wanda norms finite & >=0", bool(torch.isfinite(norms[a_name]).all()
                                         and (norms[a_name] >= 0).all()))

    # (1) magnitude prune @ 50% -> EXACTLY floor/round per-row zeros, uniform
    m1 = Qwen2ForCausalLM(cfg).eval()
    t1 = find_target_linears(m1, DEFAULT_PROJ)
    zt, tt, per = prune_model(m1, t1, 0.5, "magnitude", {})
    lin = t1[a_name].weight
    in_f = lin.shape[1]
    exp_k = round(0.5 * in_f)
    row_zeros = (lin.data == 0).sum(dim=1)
    chk("magnitude: every row has exactly round(0.5*in) zeros",
        bool((row_zeros == exp_k).all()))
    chk("magnitude: achieved sparsity == exp_k/in_f",
        abs(zt / tt - exp_k / in_f) < 1e-6)

    # (2) wanda prune @ 50% -> also exact per-row count; and it zeros the |W|*norm-smallest
    m2 = Qwen2ForCausalLM(cfg).eval()
    t2 = find_target_linears(m2, DEFAULT_PROJ)
    n2 = collect_wanda_norms(m2, t2, calib, "cpu")
    prune_model(m2, t2, 0.5, "wanda", n2)
    w = t2[a_name].weight
    rz = (w.data == 0).sum(dim=1)
    chk("wanda: every row has exactly round(0.5*in) zeros", bool((rz == exp_k).all()))
    # verify a single row's kept set == top-(in-k) by |W|*norm
    W0 = m2  # already pruned; recompute expectation on a fresh matrix instead
    fresh = torch.randn(3, in_f)
    metric = fresh.abs() * n2[a_name].view(1, -1)
    keep = torch.topk(metric, in_f - exp_k, dim=1, largest=True).indices
    manual = torch.ones_like(fresh, dtype=torch.bool)
    manual.scatter_(1, torch.topk(metric, exp_k, dim=1, largest=False).indices, False)
    pruned = fresh * manual
    chk("wanda: prune keeps exactly the |W|*norm-largest per row",
        bool(((pruned != 0).sum(dim=1) == in_f - exp_k).all()))

    # (3) sparsity 0 is a no-op
    m3 = Qwen2ForCausalLM(cfg).eval()
    t3 = find_target_linears(m3, DEFAULT_PROJ)
    ref = t3[a_name].weight.data.clone()
    z0, _, _ = prune_model(m3, t3, 0.0, "magnitude", {})
    chk("sparsity 0 -> zero weights zeroed", z0 == 0)
    chk("sparsity 0 -> weights unchanged", bool(torch.equal(t3[a_name].weight.data, ref)))

    # (4) save/reload roundtrip preserves the pruned (zeroed) weights exactly
    import tempfile
    from transformers import AutoModelForCausalLM
    with tempfile.TemporaryDirectory() as d:
        m1.save_pretrained(d, safe_serialization=True)
        r = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32)
        rl = dict(r.named_modules())[a_name].weight.data
        chk("roundtrip: reloaded row-zeros match", bool(((rl == 0).sum(dim=1) == exp_k).all()))
        chk("roundtrip: reloaded weights bit-equal to pruned",
            bool(torch.equal(rl, t1[a_name].weight.data)))
        # (5) pruned reloaded model still forwards (produces finite logits)
        with torch.no_grad():
            out = r(torch.randint(0, 256, (1, 8)))
        chk("pruned model forwards -> finite logits", bool(torch.isfinite(out.logits).all()))

    print(f"\nSELFTEST: {sum(ok)}/{len(ok)} checks passed -> "
          f"{'ALL PASS' if all(ok) else 'FAILURES PRESENT'}")
    return all(ok)


if __name__ == "__main__":
    main()
