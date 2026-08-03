"""W3-e: UNFILTERED, full-sweep search supervision on FRESH pool queries.

Fixes the three defects of Dr.LLM's data-gen we established today:
  (1) no acceptance filtering -> supervision distribution == deployment-like
      distribution (hard samples stay, labeled "execute everything");
  (2) full sweep (no early break), ALL evaluated paths + rewards recorded ->
      per-layer marginals instead of one noisy first-hit path;
  (3) responses recorded per path -> teacher-forcing completions match paths.

Samples: pool queries per level, seeded shuffle, EXCLUDING (a) the 4001
training queries and (b) the eval held-out tail (200/level, same logic as
eval_drllm.load_task('dart_pool') plus a safety margin of 400).

Output: data/soft/soft_L<level>_s<shard>.json
  [{question, gt, level, paths: [{layers, r, resp}]}]
"""
import argparse
import glob
import json
import os
import random
import time

from datasets import load_dataset

from data_generation import MCTS, MCTSConfig, MCTSModel, MCTSNode
from prompts import answer_math

L = 28


class RecordingMCTS(MCTS):
    def full_search(self, input_text, correct_answer):
        root = MCTSNode(self.model.default_path, num_layers=self.model.num_layers)
        evaluated = {}
        for sim in range(self.config.num_simulations):
            node = self.select_node(root, sim + 1)
            if not node.is_fully_expanded() and node.visits > 0:
                node = node.expand()
            if node.path in evaluated:
                reward = evaluated[node.path][0]
            else:
                reward, resp = self.evaluate_path(input_text, correct_answer, node.path)
                evaluated[node.path] = (reward, resp)
            node.backpropagate(reward)
        return evaluated


def load_fresh_dart(level, per_level):
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    used = set()
    for f in glob.glob("data/train/dart-*.json"):
        used.update(json.load(open(f))["questions"])
    dart_dir = os.path.join(hf, "datasets/hkust-nlp__dart-math-pool-math/data")
    ds = load_dataset("parquet", data_files=os.path.join(dart_dir, "train-*.parquet"),
                      split="train")
    sub = ds.filter(lambda x: x["query_metadata"]["level"] == level, num_proc=16)
    # eval tail = last 200 unique queries (load_task logic); exclude 400 for safety
    rows = list(sub)
    tail_q, seen = set(), set()
    for item in reversed(rows):
        if item["query"] in seen:
            continue
        seen.add(item["query"])
        tail_q.add(item["query"])
        if len(tail_q) >= 400:
            break
    uniq, seen2 = [], set()
    for item in rows:
        q = item["query"]
        if q in seen2 or q in tail_q:
            continue
        seen2.add(q)
        inp = answer_math.format(question=q)
        if inp in used:
            continue
        uniq.append({"input": inp, "correct": item["gt_ans"], "level": level})
    random.Random(42).shuffle(uniq)
    return uniq[:per_level]


def load_fresh_arc(task, limit):
    from prompts import answer_letter_long, format_choices
    used = set()
    for f in glob.glob("data/train/arc_*.json"):
        used.update(json.load(open(f))["questions"])
    cfg = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge"}[task]
    ds = load_dataset("allenai/ai2_arc", cfg, split="train")
    fresh = []
    for item in ds:
        choices = item["choices"]
        labels = choices["label"]
        if item["answerKey"] not in labels:
            continue
        inp = answer_letter_long.format(question=item["question"],
                                        choices_text=format_choices(choices["text"]))
        if inp in used:
            continue
        fresh.append({"input": inp,
                      "correct": chr(65 + labels.index(item["answerKey"])),
                      "level": None})
    random.Random(42).shuffle(fresh)
    return fresh[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", default="dart", choices=["dart", "arc_easy", "arc_challenge"])
    ap.add_argument("--level", type=int, default=0, help="dart only")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num_shards", type=int, default=2)
    ap.add_argument("--per_level", type=int, default=800, help="total per level across shards")
    ap.add_argument("--limit", type=int, default=500, help="arc: total across shards")
    ap.add_argument("--num_simulations", type=int, default=100)
    ap.add_argument("--out_dir", default="data/soft")
    args = ap.parse_args()

    if args.task == "dart":
        quota = load_fresh_dart(args.level, args.per_level)
        tag = f"L{args.level}"
    else:
        quota = load_fresh_arc(args.task, args.limit)
        tag = args.task
    mine = [s for i, s in enumerate(quota) if i % args.num_shards == args.shard]
    print(f"{tag} shard {args.shard}/{args.num_shards}: {len(mine)} samples "
          f"(fresh {len(quota)}, sims={args.num_simulations})")

    cfg = MCTSConfig(num_simulations=args.num_simulations,
                     model_name=args.model_path,
                     dataset="dart" if args.task == "dart" else args.task)
    model = MCTSModel(args.model_path, rank=0)
    mcts = RecordingMCTS(model, cfg)

    out = []
    t0 = time.time()
    for i, s in enumerate(mine):
        evaluated = mcts.full_search(s["input"], s["correct"])
        paths = [{"layers": list(p.layers), "r": float(r), "resp": resp}
                 for p, (r, resp) in evaluated.items()]
        out.append({"question": s["input"], "gt": s["correct"],
                    "level": s["level"], "paths": paths})
        if (i + 1) % 10 == 0:
            solved = sum(1 for o in out if any(p["r"] > 0.5 for p in o["paths"]))
            print(f"{i+1}/{len(mine)} any-correct={solved/(i+1):.3f} "
                  f"({(time.time()-t0)/(i+1):.1f}s/sample)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    fp = os.path.join(args.out_dir, f"soft_{tag}_s{args.shard}.json")
    json.dump({"model_name": args.model_path, "records": out}, open(fp, "w"))
    print("saved", fp, len(out))


if __name__ == "__main__":
    main()
