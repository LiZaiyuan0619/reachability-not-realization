"""W4-a: does EXPANDING the action space (or the starting point) raise the ceiling?

Two extensions over the Dr.LLM search space, same 200-sim budget (so any gain
is "per-budget", the honest comparison):
  --action_space backjump : adds INSERT actions — re-run an EARLIER layer at a
      later position (e.g. ... 5 6 7 [3] 8 ...). Tests the "no way back is a
      defect" hypothesis. Paths with duplicates auto-run with use_cache=False
      in generate_with_path, so this is execution-safe.
  --root static2023       : start the search from the all-exec path MINUS
      layers {20,21,22,23} (the static-prune sweet spot on ARC). Tests whether
      the static +7 and the per-sample oracle gain STACK.

Both reuse the recording full-sweep search; reports the same structure stats.
"""
import argparse
import json
import os
import random
import time

from data_generation import MCTS, MCTSConfig, MCTSModel, MCTSNode, LayerPath
from eval_drllm import load_task

L = 28
INSERT_SAMPLES = 60  # cap inserted-action candidates per node (branch control)


class ExtNode(MCTSNode):
    def __init__(self, path, parent=None, action="", num_layers=28, mode="drllm"):
        self.mode = mode  # must be set before super().__init__
        self.allow_insert = mode == "backjump"
        super().__init__(path, parent=parent, action=action, num_layers=num_layers)

    def _generate_possible_actions(self):
        actions = super()._generate_possible_actions()
        if self.mode == "skiponly":
            actions = [a for a in actions if a["type"] == "skip"]
        elif self.mode == "repeatonly":
            actions = [a for a in actions if a["type"] == "repeat"]
        if self.allow_insert:
            layers = self.path.layers
            cand = []
            for pos in range(len(layers)):
                seen_before = set(layers[: pos + 1])
                for j in seen_before:
                    if len(layers) + 1 <= self.num_layers * 2:
                        cand.append({"type": "insert", "pos": pos, "layer": j})
            random.shuffle(cand)
            actions += cand[:INSERT_SAMPLES]
            random.shuffle(actions)
        return actions

    def expand(self):
        if not self.untried_actions:
            return self
        action = self.untried_actions.pop()
        if action["type"] == "skip":
            new_path = self.path.skip_layers(action["start"], action["size"])
            desc = f"skip_{action['size']}_at_{action['start']}"
        elif action["type"] == "repeat":
            new_path = self.path.repeat_layers(action["start"], action["size"], action["count"])
            desc = f"repeat_{action['size']}_x{action['count']}_at_{action['start']}"
        else:  # insert (back-jump)
            ls = self.path.layers
            new_path = LayerPath(ls[: action["pos"] + 1] + [action["layer"]] + ls[action["pos"] + 1:])
            desc = f"insert_{action['layer']}_after_{action['pos']}"
        child = ExtNode(new_path, parent=self, action=desc,
                        num_layers=self.num_layers, mode=self.mode)
        self.children.append(child)
        return child


class ExtMCTS(MCTS):
    def __init__(self, model, config, root_path, mode):
        super().__init__(model, config)
        self.root_path = root_path
        self.mode = mode

    def full_search(self, input_text, correct_answer):
        root = ExtNode(LayerPath(list(self.root_path)), num_layers=self.model.num_layers,
                       mode=self.mode)
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
    ap.add_argument("--task", required=True,
                    choices=["arc_easy", "arc_challenge", "dart", "dart_pool"])
    ap.add_argument("--action_space",
                    choices=["drllm", "backjump", "skiponly", "repeatonly"], default="drllm")
    ap.add_argument("--root", choices=["full", "static2023"], default="full")
    ap.add_argument("--per_level", type=int, default=60)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--num_simulations", type=int, default=200)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if args.task.startswith("dart"):
        samples = load_task(args.task, "short",
                            args.per_level if args.task == "dart" else 0)
        if args.task == "dart_pool":
            by = {}
            for s in samples:
                by.setdefault(s["level"], []).append(s)
            samples = [s for lv in sorted(by) for s in by[lv][: args.per_level]]
    else:
        samples = load_task(args.task)[: args.limit]

    root_path = list(range(L)) if args.root == "full" else [l for l in range(L)
                                                            if l not in {20, 21, 22, 23}]
    tag = f"{args.task}_{args.action_space}_{args.root}"
    print(f"oracle v3: {tag} n={len(samples)} sims={args.num_simulations}")

    cfg = MCTSConfig(num_simulations=args.num_simulations, model_name=args.model_path,
                     dataset="dart" if args.task.startswith("dart") else args.task)
    model = MCTSModel(args.model_path, rank=0)
    mcts = ExtMCTS(model, cfg, root_path, mode=args.action_space)

    recs = []
    t0 = time.time()
    for i, s in enumerate(samples):
        evaluated = mcts.full_search(s["input"], s["correct"])
        orig_r, _ = mcts.evaluate_path(s["input"], s["correct"], model.default_path)
        root_r = evaluated.get(LayerPath(list(root_path)))
        good = [(list(p.layers), float(r)) for p, r in evaluated.items() if r > 0.5]
        used_insert = sum(1 for pl, _ in good
                          if any(pl[k] < max(pl[:k + 1]) for k in range(1, len(pl))))
        recs.append({"level": s.get("level"), "orig": float(orig_r > 0.5),
                     "root_ok": float((root_r or 0) > 0.5),
                     "oracle": float(bool(good)), "n_good": len(good),
                     "n_good_nonmono": used_insert})
        if (i + 1) % 20 == 0:
            o = sum(r["orig"] for r in recs) / len(recs)
            c = sum(r["oracle"] for r in recs) / len(recs)
            print(f"{i+1}/{len(samples)} orig={o:.3f} oracle={c:.3f} "
                  f"({(time.time()-t0)/(i+1):.1f}s/sample)", flush=True)

    n = len(recs)
    summary = {"task": args.task, "action_space": args.action_space, "root": args.root,
               "n": n, "sims": args.num_simulations,
               "orig_acc": sum(r["orig"] for r in recs) / n,
               "root_acc": sum(r["root_ok"] for r in recs) / n,
               "oracle_acc": sum(r["oracle"] for r in recs) / n,
               "rescued_with_nonmono_path": sum(1 for r in recs
                                                if r["orig"] == 0 and r["oracle"] == 1
                                                and r["n_good_nonmono"] > 0)}
    summary["ceiling_gain_vs_full"] = summary["oracle_acc"] - summary["orig_acc"]
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"oraclev3_{tag}.json")
    json.dump({"summary": summary, "records": recs}, open(out, "w"), indent=1)
    print("SUMMARY", json.dumps(summary, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
