"""
Clean router trainer for Dr.LLM reproduction.

Why not train.py-as-shipped: it hands prompt/completion + a `layer_decisions`
column to trl.SFTTrainer, but SFTTrainer re-tokenizes and rebuilds the dataset
to {input_ids, labels, attention_mask}, silently DROPPING layer_decisions before
the collator runs. The model then gets no router labels -> non-differentiable
argmax routing -> "element 0 does not require grad". This trainer pre-tokenizes
ourselves and uses a plain Trainer + custom collator so layer_decisions reaches
the model's forward (which computes the class-balanced focal router loss).

Only model.model.routers are trained; the base LM + lm_head are frozen.
"""
import os, json, glob
from argparse import ArgumentParser

import torch
from transformers import (AutoTokenizer, LlamaForCausalLM, Qwen2ForCausalLM,
                          Qwen3ForCausalLM, OlmoeForCausalLM, Trainer, TrainingArguments)
from datasets import Dataset
from tqdm import tqdm

SPLIT = {"llama": "<|start_header_id|>assistant<|end_header_id|>\n\n",
         "qwen":  "<|im_start|>assistant\n"}

def get_cls(name):
    n = name.lower()
    if "llama" in n: return LlamaForCausalLM, SPLIT["llama"]
    if "qwen3" in n: return Qwen3ForCausalLM, SPLIT["qwen"]
    if "qwen2" in n: return Qwen2ForCausalLM, SPLIT["qwen"]
    if "olmoe" in n: return OlmoeForCausalLM, SPLIT["qwen"]
    raise ValueError(name)

def main():
    ap = ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--data_folders", nargs="+", default=["data/train/target_30", "data/train/target_12", "data/train"])
    ap.add_argument("--run_name", default="drllm-router")
    ap.add_argument("--num_epochs", type=float, default=15)
    ap.add_argument("--gradient_accumulation", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--num_windows", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42,
                    help="training seed (multi-seed reruns close the "
                         "'reproduction failed by seed luck' objection)")
    args = ap.parse_args()

    from transformers import set_seed
    set_seed(args.seed)

    CLS, split_pat = get_cls(args.model_id)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if CLS is LlamaForCausalLM:
        tok.pad_token = "<|finetune_right_pad_id|>"; tok.padding_side = "right"
    elif tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = CLS.from_pretrained(args.model_id, dtype=torch.bfloat16, device_map="auto")
    model.model.init_routers()
    if args.num_windows:
        model.model.num_windows = args.num_windows
    nL = len(model.model.layers)
    is_instruct = "instruct" in args.model_id.lower() or ("qwen3" in args.model_id.lower())

    # ---- load MCTS label files, build pre-tokenized samples ----
    files = [f for d in args.data_folders for f in glob.glob(f"{d}/*.json")]
    data, qs = [], set()
    dist = {0: 0, 1: 0, 2: 0}
    for fp in tqdm(files, desc="load"):
        raw = json.load(open(fp))
        if raw.get("model_name") != args.model_id:
            continue
        for q, bp, ans in zip(raw["questions"], raw["best_path"], raw["best_response"]):
            if q in qs:
                continue
            qs.add(q)
            layer_dec = [min(bp.count(i), 2) for i in range(nL)]  # {0,1,2}, clamp stacked repeats
            for v in layer_dec:
                dist[v] += 1
            if is_instruct:
                msg = tok.apply_chat_template([{"role": "user", "content": q},
                                               {"role": "assistant", "content": str(ans)}], tokenize=False)
                if split_pat not in msg:
                    continue
                prompt, completion = msg.split(split_pat); prompt += split_pat
            else:
                prompt, completion = q, str(ans)
            pids = tok(prompt, add_special_tokens=not is_instruct).input_ids
            cids = tok(completion, add_special_tokens=False).input_ids
            if len(cids) == 0:
                cids = [tok.eos_token_id]
            input_ids = pids + cids
            labels = [-100] * len(pids) + cids
            data.append({"input_ids": input_ids, "labels": labels, "layer_decisions": layer_dec})
    print(f"loaded {len(data)} samples from {len(files)} files; label dist skip/exec/repeat = "
          f"{dist[0]}/{dist[1]}/{dist[2]}")
    assert data, "no training samples (check model_id matches data model_name)"
    ds = Dataset.from_list(data)

    class Collator:
        def __init__(self, pad_id): self.pad_id = pad_id
        def __call__(self, feats):
            L = max(len(f["input_ids"]) for f in feats)
            ii, lb, am, ld = [], [], [], []
            for f in feats:
                n = len(f["input_ids"]); p = L - n
                ii.append(f["input_ids"] + [self.pad_id] * p)
                lb.append(f["labels"] + [-100] * p)
                am.append([1] * n + [0] * p)
                ld.append(f["layer_decisions"])
            return {"input_ids": torch.tensor(ii), "attention_mask": torch.tensor(am),
                    "labels": torch.tensor(lb), "layer_decisions": torch.tensor(ld)}

    for p in model.parameters(): p.requires_grad = False
    for p in model.model.routers.parameters(): p.requires_grad = True
    model.model_accepts_loss_kwargs = False  # our forward returns the router loss directly

    targs = TrainingArguments(
        output_dir=f"checkpoints/{args.run_name}",
        per_device_train_batch_size=1,            # teacher forcing runs layers a per-sample count -> bs must be 1
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=2, save_steps=args.save_steps, save_total_limit=2,
        bf16=True, dataloader_num_workers=2,
        remove_unused_columns=False,              # keep layer_decisions -> reaches model.forward
        report_to="none", run_name=args.run_name,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=Collator(tok.pad_token_id), processing_class=tok)
    trainer.train()
    trainer.save_model(f"checkpoints/{args.run_name}")
    tok.save_pretrained(f"checkpoints/{args.run_name}")  # self-contained ckpt for lm_eval
    print("saved routers to", f"checkpoints/{args.run_name}")

if __name__ == "__main__":
    main()
