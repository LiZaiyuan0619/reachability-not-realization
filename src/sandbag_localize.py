#!/usr/bin/env python3
"""SANDBAG LOCK LOCALIZATION (07-12, PILLAR 6 = ledger->surgery, unifies with the scar
mechanism line). The scar work localized an ACCIDENTAL realization failure to a single
MLP write; here we localize a DELIBERATE one (a password lock) with the SAME scalpel.
locked = password-locked Mistral (G collapses to <random on WMDP MCQA without the pw).
base   = Mistral-7B-v0.1 (untouched). We TRANSPLANT base's decoder layer L into the
locked model, one layer at a time, and measure whether G (greedy MCQA letter accuracy,
no password) RECOVERS. Layers whose transplant restores G are where the lock lives.
Anchors: locked-intact G (floor) and all-base G (ceiling). Also a cumulative-prefix
block scan (layers 0..L) to see if the lock is distributed vs local.
    python sandbag_localize.py --locked P --base P --out X [--scan single|prefix]
        [--layers 0,4,8,...] [--limit 120]
    python sandbag_localize.py --selftest      # no model/GPU: row loader + swap bookkeeping
"""
import argparse, json, os, time
import sandbag_eval as SE          # reuse load_rows / parse_letter / DEFAULT_CSV


def layer_module(model, i, part="full"):
    """the transplant unit: the whole decoder layer, or just its MLP submodule.
    part='mlp' pins the lock to the SAME substrate the scar-mechanism line implicated
    (a single MLP write) vs attention."""
    layer = model.model.layers[i]
    return layer.mlp if part == "mlp" else layer


def swap_in(dst_layer, src_state):
    """load src_state into dst_layer; return the dst's ORIGINAL state (for restore)."""
    orig = {k: v.detach().clone() for k, v in dst_layer.state_dict().items()}
    dst_layer.load_state_dict(src_state, strict=True)
    return orig


def run_selftest():
    print("== sandbag_localize selftest (row loader + swap bookkeeping; no model) ==")
    rows = SE.load_rows(SE.DEFAULT_CSV, use_pw=False, limit=20, seed=0)
    assert len(rows) == 20 and rows[0]["gold"] in "ABCD"
    assert rows[0]["prompt"].rstrip().endswith("Answer:")
    import torch
    import torch.nn as nn
    a = nn.Linear(3, 3); b = nn.Linear(3, 3)
    with torch.no_grad():
        for p in a.parameters(): p.zero_()
        for p in b.parameters(): p.add_(1.0)
    orig = swap_in(a, b.state_dict())
    assert torch.allclose(a.weight, b.weight)            # swapped to b
    a.load_state_dict(orig)                              # restore
    assert torch.allclose(a.weight, torch.zeros(3, 3))   # back to original
    print("[selftest] rows OK; swap+restore round-trips OK")


def measure_G(model, tok, rows, max_new=3):
    import torch
    correct = 0
    for s in rows:
        inp = tok(s["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        correct += (SE.parse_letter(gen) == s["gold"])
    return correct / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked"); ap.add_argument("--base")
    ap.add_argument("--csv", default=SE.DEFAULT_CSV)
    ap.add_argument("--out")
    ap.add_argument("--scan", choices=["single", "prefix"], default="single")
    ap.add_argument("--direction", choices=["fwd", "rev"], default="fwd",
                    help="fwd = transplant BASE layer into LOCKED, measure G RECOVERY "
                         "(where does healing the lock restore ability); rev = transplant "
                         "LOCKED layer into BASE, measure G BREAKAGE (where does inserting "
                         "the lock destroy a healthy model). Two-sided locality evidence.")
    ap.add_argument("--part", choices=["full", "mlp"], default="full",
                    help="transplant the whole decoder layer, or only its MLP (pin the "
                         "lock to the scar line's MLP-write substrate vs attention)")
    ap.add_argument("--layers", default="", help="comma list of layer ids to scan; '' = all")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        run_selftest(); return
    assert args.locked and args.base and args.out
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = SE.load_rows(args.csv, use_pw=False, limit=args.limit, seed=0)
    # fwd: heal the locked model (host=locked, donor=base), G recovers where lock lives.
    # rev: infect the base model  (host=base,   donor=locked), G breaks where lock lives.
    host_path, donor_path = (args.locked, args.base) if args.direction == "fwd" else (args.base, args.locked)
    tok = AutoTokenizer.from_pretrained(host_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[{args.direction}/{args.part}] host={host_path}  donor={donor_path}", flush=True)
    host = AutoModelForCausalLM.from_pretrained(host_path, torch_dtype=torch.bfloat16,
                                                device_map={"": "cuda:0"}).eval()
    donor = AutoModelForCausalLM.from_pretrained(donor_path, torch_dtype=torch.bfloat16,
                                                 device_map={"": "cuda:0"}).eval()
    nlayers = len(host.model.layers)
    donor_states = [{k: v.detach().clone() for k, v in layer_module(donor, i, args.part).state_dict().items()}
                    for i in range(nlayers)]

    floor = measure_G(host, tok, rows)            # host intact anchor
    anchor_name = "locked-intact(floor)" if args.direction == "fwd" else "base-intact(ceiling)"
    print(f"[anchor] {anchor_name} G = {floor:.4f} (n={len(rows)})", flush=True)

    scan = [int(x) for x in args.layers.split(",") if x != ""] or list(range(nlayers))
    results = []
    if args.scan == "single":
        for L in scan:
            orig = swap_in(layer_module(host, L, args.part), donor_states[L])
            g = measure_G(host, tok, rows)
            layer_module(host, L, args.part).load_state_dict(orig)   # restore
            results.append({"layer": L, "G_transplant": g, "delta": g - floor})
            print(f"  layer {L:2d}: G={g:.4f}  delta={g-floor:+.4f}", flush=True)
    else:  # prefix: cumulative layers 0..L (NOT restored until end -> rebuild each step)
        origs = {}
        for L in scan:
            origs[L] = swap_in(layer_module(host, L, args.part), donor_states[L])
            g = measure_G(host, tok, rows)
            results.append({"prefix_upto": L, "G_transplant": g, "delta": g - floor})
            print(f"  prefix 0..{L:2d}: G={g:.4f}  delta={g-floor:+.4f}", flush=True)
        for L, o in origs.items():
            layer_module(host, L, args.part).load_state_dict(o)

    summary = {"locked": args.locked, "base": args.base, "scan": args.scan,
               "direction": args.direction, "part": args.part,
               "n": len(rows), "G_floor": floor, "results": results,
               "acc_micro": floor}    # acc_micro key so generic okjson validators pass
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "records": results}, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
