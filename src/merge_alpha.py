#!/usr/bin/env python3
"""ALPHA WEIGHT INTERPOLATION (07-12, PILLAR 6 = ledger as an INTERVENTION tool).
Linearly interpolate two checkpoints that share an architecture:
    theta(alpha) = (1 - alpha) * base + alpha * donor
and save the merged model so the ledger (G / R / oracle@K / majority@K) can be read at
each alpha. Applied to Olmo-3-1025-7B (base) <-> Olmo-3.1-7B-RL-Zero-Math (RL): if the
five ledger columns evolve CONTINUOUSLY and MONOTONICALLY along alpha, the entire RLVR
ledger effect is reproducible by a single weight direction -- upgrading "RL = distribution
sharpening" from an observation to a constructive proof. A knee (alpha* where produced has
risen but oracle@K has not yet collapsed) would be a free "realize-without-losing-ceiling"
merge point (model-merging x RLVR ledger, unoccupied).

Only merges tensors present with identical shape in BOTH checkpoints; asserts full
coverage (Olmo base<->rlzm verified 355/355 identical keys, same dims). Tokenizer copied
from base. Saves bf16.
    python merge_alpha.py --base P_base --donor P_rl --alpha 0.5 --out DIR
    python merge_alpha.py --selftest      # tiny 2-tensor CPU check, no model/GPU
"""
import argparse, json, os, re, shutil, sys

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def key_eligible(k, modules="all", layer_lo=None, layer_hi=None):
    """SURGERY FILTER (07-13, N1 separability). A tensor takes donor weight only if it
    passes BOTH filters; everything else stays pure base. Non-transformer-block tensors
    (embeddings, lm_head, final norm) never match a module/band filter -- surgery is
    deliberately restricted to block weights. modules=all + no band == old behavior."""
    m = _LAYER_RE.search(k)
    if modules == "all" and layer_lo is None and layer_hi is None:
        return True
    if m is None:
        return False               # non-block tensor under any restriction -> base
    li = int(m.group(1))
    if layer_lo is not None and li < layer_lo:
        return False
    if layer_hi is not None and li > layer_hi:
        return False
    if modules == "mlp" and ".mlp." not in k:
        return False
    if modules == "attn" and ".self_attn." not in k:
        return False
    return True


def interp_state(sd_base, sd_donor, alpha, modules="all", layer_lo=None, layer_hi=None):
    """(1-alpha)*base + alpha*donor over shared, shape-matched, FILTER-ELIGIBLE keys.
    Returns (merged_dict, n_merged, skipped_keys, n_kept_base). skipped = shape/coverage
    mismatch (arch guard); kept_base = deliberately filtered out (surgery), NOT an error."""
    out, skipped, kept = {}, [], 0
    for k, vb in sd_base.items():
        vd = sd_donor.get(k)
        if vd is None or tuple(vd.shape) != tuple(vb.shape):
            skipped.append(k)
            out[k] = vb            # keep base copy for any non-shared tensor
            continue
        if not key_eligible(k, modules, layer_lo, layer_hi):
            out[k] = vb; kept += 1
            continue
        # compute in fp32 for numerical safety, store back in the base tensor's dtype
        # (bf16) so the accumulated merged dict stays ~14G not ~28G -- keeps peak host
        # RAM well under the 96G job cap when several alpha jobs run concurrently.
        out[k] = ((1.0 - alpha) * vb.float() + alpha * vd.float()).to(vb.dtype)
    return out, len(out) - len(skipped) - kept, skipped, kept


def run_selftest():
    import torch
    base = {"w": torch.zeros(4), "b": torch.ones(2)}
    donor = {"w": torch.ones(4) * 4, "b": torch.ones(2) * 2}
    m, nmerged, skip, kept = interp_state(base, donor, 0.25)
    assert torch.allclose(m["w"], torch.ones(4)), m["w"]        # 0.75*0 + 0.25*4 = 1
    assert torch.allclose(m["b"], torch.ones(2) * 1.25), m["b"]  # 0.75*1 + 0.25*2 = 1.25
    assert nmerged == 2 and skip == [] and kept == 0
    # shape-mismatch skip path
    m2, n2, s2, _ = interp_state({"w": torch.zeros(4), "x": torch.zeros(3)},
                                 {"w": torch.ones(4), "x": torch.ones(5)}, 0.5)
    assert n2 == 1 and s2 == ["x"] and torch.allclose(m2["x"], torch.zeros(3))
    # endpoint sanity
    m0 = interp_state(base, donor, 0.0)[0]; m1 = interp_state(base, donor, 1.0)[0]
    assert torch.allclose(m0["w"], base["w"].float()) and torch.allclose(m1["w"], donor["w"].float())
    # ---- surgery filters (07-13) ----
    sdb = {"model.embed_tokens.weight": torch.zeros(2),
           "model.layers.0.mlp.gate_proj.weight": torch.zeros(2),
           "model.layers.0.self_attn.q_proj.weight": torch.zeros(2),
           "model.layers.5.mlp.up_proj.weight": torch.zeros(2),
           "model.layers.5.post_attention_layernorm.weight": torch.zeros(2)}
    sdd = {k: torch.ones_like(v) for k, v in sdb.items()}
    # mlp-only, alpha=1: only .mlp. keys take donor; embed/attn/norm stay base
    mm, nm, sk, kp = interp_state(sdb, sdd, 1.0, modules="mlp")
    assert nm == 2 and kp == 3 and sk == []
    assert mm["model.layers.0.mlp.gate_proj.weight"][0].item() == 1
    assert mm["model.layers.0.self_attn.q_proj.weight"][0].item() == 0
    assert mm["model.embed_tokens.weight"][0].item() == 0
    # band 0..4, all modules: layer 0 (both) merged, layer 5 + embed stay base
    mb, nb, _, kb = interp_state(sdb, sdd, 1.0, layer_lo=0, layer_hi=4)
    assert nb == 2 and kb == 3
    assert mb["model.layers.5.mlp.up_proj.weight"][0].item() == 0
    assert mb["model.layers.0.self_attn.q_proj.weight"][0].item() == 1
    # band + module combined: only layers.5.mlp
    mc, nc, _, kc = interp_state(sdb, sdd, 1.0, modules="mlp", layer_lo=5, layer_hi=31)
    assert nc == 1 and kc == 4 and mc["model.layers.5.mlp.up_proj.weight"][0].item() == 1
    # default filter == identity behavior (regression guard)
    assert key_eligible("model.embed_tokens.weight") and key_eligible("anything.at.all")
    print("[selftest] interp math + skip + endpoints + surgery filters OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--donor")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out")
    ap.add_argument("--modules", choices=["all", "mlp", "attn"], default="all",
                    help="surgery: donor weights only for this module family (block tensors)")
    ap.add_argument("--layer_lo", type=int, default=None)
    ap.add_argument("--layer_hi", type=int, default=None,
                    help="surgery: donor weights only for layers in [lo,hi] (inclusive)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        run_selftest(); return
    assert args.base and args.donor and args.out, "--base/--donor/--out required"
    assert 0.0 <= args.alpha <= 1.0
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading base   {args.base}", flush=True)
    mb = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    print(f"loading donor  {args.donor}", flush=True)
    md = AutoModelForCausalLM.from_pretrained(args.donor, torch_dtype=torch.bfloat16)
    sdb, sdd = mb.state_dict(), md.state_dict()
    merged, nmerged, skipped, kept = interp_state(sdb, sdd, args.alpha,
                                                  args.modules, args.layer_lo, args.layer_hi)
    print(f"merged {nmerged} tensors at alpha={args.alpha} "
          f"(modules={args.modules}, band={args.layer_lo}..{args.layer_hi}); "
          f"kept_base(filtered)={kept}; skipped(mismatch)={len(skipped)}")
    # base<->donor must share the arch; a large skip set means arch mismatch -> abort.
    assert len(skipped) <= 2, f"too many non-shared tensors ({len(skipped)}): {skipped[:8]}"
    if args.modules != "all" or args.layer_lo is not None or args.layer_hi is not None:
        assert nmerged > 0, "surgery filter matched zero tensors -- wrong key patterns?"
    mb.load_state_dict({k: v.to(torch.bfloat16) for k, v in merged.items()}, strict=True)
    os.makedirs(args.out, exist_ok=True)
    mb.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    # carry a standalone chat_template if the base ships one as a separate file
    for cand in ("chat_template.jinja", "chat_template.json"):
        p = os.path.join(args.base, cand)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.out, cand))
    with open(os.path.join(args.out, "MERGE_INFO.json"), "w") as f:
        json.dump({"base": args.base, "donor": args.donor, "alpha": args.alpha,
                   "modules": args.modules, "layer_lo": args.layer_lo, "layer_hi": args.layer_hi,
                   "n_merged": nmerged, "n_kept_base": kept, "n_skipped": len(skipped)}, f, indent=1)
    print(f"saved merged model -> {args.out}")


if __name__ == "__main__":
    main()
