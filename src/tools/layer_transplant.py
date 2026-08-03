"""Layer-transplant surgery for the alignment-scar hypothesis (W11 mechanism probe).

Swaps whole decoder layers (model.layers.N.*) from --src into --dst and saves the
hybrid to --out. Embeddings / norms / lm_head stay --dst's. Both models must share
config (Llama-3.2-3B base vs instruct do).

Readout logic:
  instruct <- base @L20 recovers the skip-L20 gain  => the lesion lives in L20's
      instruct weights (alignment overwrote a healthy layer) -- causal.
  instruct <- base @L20 stays at baseline           => interaction effect: L20 is
      only harmful in the context of the other aligned layers.
  base <- instruct @L20 drops base accuracy         => the damage is portable.
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True, help="recipient model (kept except layers)")
    ap.add_argument("--src", required=True, help="donor model (provides the layers)")
    ap.add_argument("--layers", required=True, help="comma list, e.g. '20' or '20,21,22,23'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    dst = AutoModelForCausalLM.from_pretrained(args.dst, torch_dtype=torch.float16)
    src = AutoModelForCausalLM.from_pretrained(args.src, torch_dtype=torch.float16)
    assert dst.config.num_hidden_layers == src.config.num_hidden_layers

    src_sd = src.state_dict()
    patch = {k: v for k, v in src_sd.items()
             if any(k.startswith(f"model.layers.{l}.") for l in layers)}
    assert patch, "no matching layer keys in donor state dict"
    dst.load_state_dict({**dst.state_dict(), **patch}, strict=True)
    print(f"transplanted {len(patch)} tensors for layers {layers}")

    os.makedirs(args.out, exist_ok=True)
    dst.save_pretrained(args.out, safe_serialization=True)
    print("saved hybrid to", args.out)


if __name__ == "__main__":
    main()
