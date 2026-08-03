"""Merge the password-locked sandbagging LoRA onto Mistral-7B-v0.1 (W-sandbag audit).

FelixHofstaetter/mistral-7b-sandbagging-new is a peft LoRA (r16 a16, all 7 proj
modules) trained to make Mistral-7B-v0.1 answer MCQA *wrong* unless a password
appears in the prompt (sandbagging / password-locking; Elicitation-Game family).
The audit question: can the capability ledger's non-G columns (R = MC letter
loglik, later S_K/P_K) see the hidden capability WITHOUT the password, i.e.
G collapses but R stays high on the locked model while base shows G~R?

Per the adapter README the tokenizer adds <PAD> (id 32000, padding only, never
appears in eval prompts), so loading order is:
  base -> config.pad_token_id = tok.pad_token_id -> resize_token_embeddings
  -> PeftModel.from_pretrained -> merge_and_unload
The adapter has NO embedding deltas (modules_to_save null), so the resized
<PAD> row is untrained noise either way; we seed it for determinism anyway.

Merge runs on CPU (bf16, ~15GB RAM; worker has 96G) so it needs no GPU and the
probe job can spend the GPU on evals. Output is a standard HF dir (+ tokenizer)
that eval_drllm.py loads with --model_path/--tokenizer_path.

  python merge_sandbag.py [--selftest]        # selftest = validate paths/args only
Writes <out>/merge_manifest.json last -> its existence is the idempotency key
(a killed half-written save leaves no manifest and gets redone).
"""
import argparse
import json
import os
import time

BASE_DEFAULT = "models/Mistral-7B-v0.1"
ADAPTER_DEFAULT = "models/mistral-7b-sandbagging-new"
OUT_DEFAULT = "models/sandbag_merged/locked"


def selftest(args):
    """Validate inputs without touching model weights (runs on dev in seconds)."""
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[ok]  " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    for f in ("config.json", "model.safetensors.index.json",
              "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        p = os.path.join(args.base, f)
        check(os.path.isfile(p), f"base file {p}")
    cfg_p = os.path.join(args.base, "config.json")
    if os.path.isfile(cfg_p):
        cfg = json.load(open(cfg_p))
        check(cfg.get("model_type") == "mistral", f"base model_type=mistral (got {cfg.get('model_type')})")
        check(cfg.get("vocab_size") == 32000, f"base vocab_size=32000 (got {cfg.get('vocab_size')})")

    for f in ("adapter_config.json", "adapter_model.safetensors",
              "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        p = os.path.join(args.adapter, f)
        check(os.path.isfile(p), f"adapter file {p}")
    acfg_p = os.path.join(args.adapter, "adapter_config.json")
    if os.path.isfile(acfg_p):
        acfg = json.load(open(acfg_p))
        check(acfg.get("base_model_name_or_path") == "mistralai/Mistral-7B-v0.1",
              f"adapter expects mistralai/Mistral-7B-v0.1 (got {acfg.get('base_model_name_or_path')})")
        check(acfg.get("peft_type") == "LORA", "adapter peft_type=LORA")

    try:
        import peft
        check(True, f"peft import ({peft.__version__})")
    except Exception as e:  # noqa: BLE001
        check(False, f"peft import: {e}")

    # tokenizer load is cheap (2MB json) and pins the resize target
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.adapter)
        check(len(tok) == 32001, f"adapter tokenizer len=32001 (got {len(tok)})")
        check(tok.pad_token == "<PAD>" and tok.pad_token_id == 32000,
              f"pad token <PAD>/32000 (got {tok.pad_token}/{tok.pad_token_id})")
    except Exception as e:  # noqa: BLE001
        check(False, f"adapter tokenizer load: {e}")

    check(os.path.isdir(os.path.dirname(args.out.rstrip("/"))) or True, "out parent creatable")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--adapter", default=ADAPTER_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--selftest", action="store_true",
                    help="validate paths/configs/tokenizer only; no model weights loaded")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest(args))

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft
    import transformers

    torch.manual_seed(0)  # determinism for the (unused) resized <PAD> embedding row
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.adapter)
    assert len(tok) == 32001 and tok.pad_token_id == 32000, \
        f"unexpected adapter tokenizer: len={len(tok)} pad={tok.pad_token_id}"

    print(f"[load] base {args.base} (cpu, bf16)")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    base.config.pad_token_id = tok.pad_token_id
    base.resize_token_embeddings(len(tok))  # 32000 -> 32001, per adapter README
    print(f"[resize] embeddings -> {base.get_input_embeddings().weight.shape[0]}")

    print(f"[load] adapter {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    print("[merge] merge_and_unload")
    merged = model.merge_and_unload()
    merged.config.use_cache = True

    os.makedirs(args.out, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    manifest = {
        "base": args.base,
        "adapter": args.adapter,
        "vocab_size": len(tok),
        "pad_token_id": tok.pad_token_id,
        "dtype": "bfloat16",
        "peft": peft.__version__,
        "transformers": transformers.__version__,
        "elapsed_s": round(time.time() - t0, 1),
        "note": "password-locked mistral-7b-sandbagging-new merged onto Mistral-7B-v0.1 "
                "(README order: pad_token_id -> resize -> attach -> merge_and_unload)",
    }
    with open(os.path.join(args.out, "merge_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("MANIFEST", json.dumps(manifest, indent=1))
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
