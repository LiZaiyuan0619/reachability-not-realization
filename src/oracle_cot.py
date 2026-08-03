"""W4-d: oracle ceiling under the REALISTIC CoT protocol (dart).

All previous oracles used the paper's short-answer protocol. If the routing
ceiling vanishes under CoT, the gains story is protocol-bound; if it survives,
CoT is the battlefield worth fighting on. Costly per evaluation (up to 512
tokens), so small n and sims.
"""
import argparse
import json
import os
import time

import torch
from mathruler.grader import extract_boxed_content, grade_answer

from data_generation import MCTS, MCTSConfig, MCTSModel, MCTSNode
from eval_drllm import load_task, cot_math

L = 28


class CoTOracle(MCTS):
    def __init__(self, model, config, max_new=512):
        super().__init__(model, config)
        self.max_new = max_new

    def evaluate_path(self, input_text, correct_answer, path):
        resp = self.model.generate_with_path(input_text, path, max_new_tokens=self.max_new)
        pred = extract_boxed_content(resp.strip())
        return float(grade_answer(pred, correct_answer)), resp

    def full_search(self, input_text, correct_answer):
        root = MCTSNode(self.model.default_path, num_layers=self.model.num_layers)
        evaluated = {}
        for sim in range(self.config.num_simulations):
            node = self.select_node(root, sim + 1)
            if not node.is_fully_expanded() and node.visits > 0:
                node = node.expand()
            if node.path in evaluated:
                reward = evaluated[node.path]
            else:
                reward, _ = self.evaluate_path(input_text, correct_answer, node.path)
                evaluated[node.path] = reward
            node.backpropagate(reward)
        return evaluated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", default="dart_pool", choices=["dart", "dart_pool"])
    ap.add_argument("--per_level", type=int, default=20)
    ap.add_argument("--num_simulations", type=int, default=30)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    samples = load_task(args.task, "short", 0)
    by = {}
    for s in samples:
        by.setdefault(s["level"], []).append(s)
    samples = [s for lv in sorted(by) for s in by[lv][: args.per_level]]
    # rebuild prompts as CoT
    for s in samples:
        q = s["input"].split("Question: ", 1)[1].rsplit("\nThe final answer", 1)[0]
        s["input"] = cot_math.format(question=q)
    print(f"CoT oracle: {args.task} n={len(samples)} sims={args.num_simulations}")

    cfg = MCTSConfig(num_simulations=args.num_simulations,
                     model_name=args.model_path, dataset="dart")
    model = MCTSModel(args.model_path, rank=0)
    mcts = CoTOracle(model, cfg)

    recs = []
    t0 = time.time()
    for i, s in enumerate(samples):
        orig_r, _ = mcts.evaluate_path(s["input"], s["correct"], model.default_path)
        evaluated = mcts.full_search(s["input"], s["correct"])
        good = [(list(p.layers), float(r)) for p, r in evaluated.items() if r > 0.5]
        recs.append({"level": s["level"], "orig": float(orig_r > 0.5),
                     "oracle": float(orig_r > 0.5 or bool(good)), "n_good": len(good)})
        o = sum(r["orig"] for r in recs) / len(recs)
        c = sum(r["oracle"] for r in recs) / len(recs)
        print(f"{i+1}/{len(samples)} orig={o:.3f} oracle={c:.3f} "
              f"({(time.time()-t0)/(i+1):.0f}s/sample)", flush=True)

    n = len(recs)
    summary = {"task": args.task, "protocol": "cot", "n": n, "sims": args.num_simulations,
               "orig_acc": sum(r["orig"] for r in recs) / n,
               "oracle_acc": sum(r["oracle"] for r in recs) / n}
    summary["ceiling_gain"] = summary["oracle_acc"] - summary["orig_acc"]
    per = {}
    for lv in range(1, 6):
        rs = [r for r in recs if r["level"] == lv]
        if rs:
            per[lv] = {"orig": sum(r["orig"] for r in rs) / len(rs),
                       "oracle": sum(r["oracle"] for r in rs) / len(rs)}
    summary["per_level"] = per
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"oracle_cot_{args.task}.json")
    json.dump({"summary": summary, "records": recs}, open(out, "w"), indent=1)
    print("SUMMARY", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
