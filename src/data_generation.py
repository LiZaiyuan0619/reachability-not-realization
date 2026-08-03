import os
import re
import math
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import torch
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    MistralForCausalLM,
    OlmoeForCausalLM,
    Qwen3ForCausalLM,
    Qwen2ForCausalLM
)
from datasets import load_dataset
from tqdm import tqdm
import json
import time
from argparse import ArgumentParser
from prompts import (
    answer_letter_base, 
    answer_letter_long, 
    answer_math, 
    answer_math_base, 
    format_choices, 
    format_choices_base,
)
from mathruler.grader import extract_boxed_content, grade_answer
import torch.multiprocessing as mp


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCTSConfig:
    """Configuration for MCTS"""
    num_simulations: int = 200  # MCTS simulations per input
    exploration_constant: float = 1.8  # UCB exploration constant
    length_penalty: float = 3.0  # Lambda for path length penalty
    random_prob: float = 0.1  # Probability of random selection during exploration
    max_path_length: int = 40  # Maximum allowed path length
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"  # Model name
    dataset: str = "arc_easy"
    

SKIP_SIZE = [1, 2]  # Sizes for skip actions
REPEAT_SIZE = [1]  # Sizes for repeat actions
REPEAT_COUNT = [0, 1]  # Counts for repeat actions

def get_is_instruct(model_name: str) -> bool:
    m = model_name.lower()
    return "instruct" in m or ("qwen3" in m and "base" not in m)

class LayerPath:
    def __init__(self, layers: List[int]):
        self.layers = layers
        self.length = len(layers)
        self.unique_layers = len(set(layers))
    
    def copy(self):
        return LayerPath(self.layers.copy())
    
    def skip_layers(self, start_idx: int, num_layers: int) -> 'LayerPath':
        new_layers = self.layers[:start_idx] + self.layers[start_idx + num_layers:]
        return LayerPath(new_layers)
    
    def repeat_layers(self, start_idx: int, num_layers: int, repeats: int) -> 'LayerPath':
        segment = self.layers[start_idx:start_idx + num_layers]
        repeated_segment = segment * (repeats + 1)
        new_layers = (self.layers[:start_idx] + 
                     repeated_segment + 
                     self.layers[start_idx + num_layers:])
        return LayerPath(new_layers)
    
    def __str__(self):
        return f"LayerPath(layers={self.layers}, length={self.length})"
    
    def __hash__(self):
        return hash(tuple(self.layers))
    
    def __eq__(self, other):
        return isinstance(other, LayerPath) and self.layers == other.layers

class MCTSNode:
    def __init__(self, path: LayerPath, parent: Optional['MCTSNode'] = None, action: str = "", num_layers: int = 28):
        self.path = path
        self.parent = parent
        self.action = action 
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.rewards = 0.0
        self.num_layers = num_layers
        self.untried_actions = self._generate_possible_actions()
        
    def _generate_possible_actions(self) -> List[Dict[str, Any]]:
        actions = []
        path_len = len(self.path.layers)
        for start in range(path_len):
            for skip_size in SKIP_SIZE:
                if start + skip_size <= path_len:
                    actions.append({
                        'type': 'skip',
                        'start': start,
                        'size': skip_size
                    })
        
        for start in range(path_len):
            for repeat_size in REPEAT_SIZE:
                for repeat_count in REPEAT_COUNT:
                    if start + repeat_size <= path_len:
                        additional_layers = repeat_size * repeat_count
                        if len(self.path.layers) + additional_layers <= self.num_layers * 2:
                            actions.append({
                                'type': 'repeat',
                                'start': start,
                                'size': repeat_size,
                                'count': repeat_count
                            })
        random.shuffle(actions)
        return actions
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def expand(self) -> 'MCTSNode':
        if not self.untried_actions:
            return self
        
        action = self.untried_actions.pop()
        
        if action['type'] == 'skip':
            new_path = self.path.skip_layers(action['start'], action['size'])
            action_desc = f"skip_{action['size']}_at_{action['start']}"
        else:  # repeat
            new_path = self.path.repeat_layers(
                action['start'], action['size'], action['count']
            )
            action_desc = f"repeat_{action['size']}_x{action['count']}_at_{action['start']}"
        
        child = MCTSNode(new_path, parent=self, action=action_desc)
        self.children.append(child)
        return child
    
    def ucb_score(self, exploration_constant: float, total_visits: int, length_penalty: float) -> float:
        """Calculate UCB score for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.rewards / self.visits
        exploration = exploration_constant * math.sqrt(math.log(total_visits) / self.visits)
        penalty = length_penalty * (self.path.length / self.num_layers)  # Normalize by typical layer count
        
        return exploitation + exploration - penalty
    
    def best_child(self, exploration_constant: float, total_visits: int, length_penalty: float) -> 'MCTSNode':
        return max(self.children, 
                  key=lambda child: child.ucb_score(exploration_constant, total_visits, length_penalty))
    
    def backpropagate(self, reward: float):
        self.visits += 1
        self.rewards += reward
        if self.parent:
            self.parent.backpropagate(reward)
            
def get_model_cls(name):
    if "llama" in name.lower():
        return LlamaForCausalLM
    elif "qwen3" in name.lower():
        return Qwen3ForCausalLM
    elif "qwen2" in name.lower():
        return Qwen2ForCausalLM
    elif "olmoe" in name.lower():
        return OlmoeForCausalLM
    elif "mistral" in name.lower():
        return MistralForCausalLM
    else:
        raise ValueError(f"Unknown model type for name: {name}. Supported models are Llama, Qwen, Olmoe, and Mistral.")
    

class MCTSModel:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", rank=0):
        self.model_name = model_name
        logger.info(f"[GPU {torch.cuda.current_device()}] Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.rank = rank
        CLS = get_model_cls(model_name)
        self.model = CLS.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={ "": f"cuda:{rank}" },
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.num_layers = len(self.model.model.layers)
        logger.info(f"Model loaded with {self.num_layers} layers")
        
        # Default path (all layers in order)
        self.default_path = LayerPath(list(range(self.num_layers)))
        
    def prepare_prompt(self, query: str, tokenizer: AutoTokenizer) -> str:
        if not get_is_instruct(self.model_name): return query
        messages = [{"role": "user", "content": query}]
        kwargs = {}
        if "qwen3" in self.model_name.lower(): kwargs['enable_thinking'] = False
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    
    def generate_with_path(self, query: str, path: LayerPath, 
                          max_new_tokens: int = 10, temperature: float = 0.0) -> str:
        original_layers = self.model.model.layer_indices
        self.model.model.layer_indices = path.layers  # Set active layers for generation
        prompt = self.prepare_prompt(query, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            min_length=input_len + 2,
            # A REPEAT (a layer appearing >1x in the path) runs that layer twice per
            # forward, but the KV cache is keyed by fixed layer_idx with a single mask
            # built from the reference layer's length -> the doubled layer overflows the
            # mask by 1 on decode ("expanded size (N) must match (N-1)"). So disable the
            # cache ONLY for repeat paths; skip-only paths are cache-safe and much faster.
            # Greedy output is identical either way. See LayerRouting/debug/verify_fix.sh.
            use_cache=(len(path.layers) == len(set(path.layers))),
        )
        new_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        self.model.model.layer_indices = original_layers  # Restore original layers
        return response.strip()

class MCTS:
    def __init__(self, model: MCTSModel, config: MCTSConfig):
        self.model = model
        self.config = config

    def evaluate_path(self, input_text: str, correct_answer: str, path: LayerPath) -> float:
        is_dart = "dart" in self.config.dataset
        is_instruct = get_is_instruct(self.config.model_name)
        max_new_tokens = 15 if is_dart else (10 if is_instruct else 2)
        raw_response = self.model.generate_with_path(
            input_text, 
            path,
            max_new_tokens=max_new_tokens,
        )
        # print(f"Evaluating path: {path}\nResponse: {response} -- Correct: {correct_answer}")
        if is_dart:
            # HACK: add boxed for base models
            if "boxed" in input_text and not is_instruct: raw_response = "\\boxed{" + raw_response
            pred_answer = extract_boxed_content(raw_response.strip())
            return float(grade_answer(pred_answer, correct_answer)), raw_response
        pred_answer = re.match(r"^(?:Answer:\s*)?([A-Da-d])\.?$", raw_response.strip())
        if not pred_answer:
            return 0.0, raw_response
        matched_group = pred_answer.group(1) or pred_answer.group(2)
        response = matched_group.strip()[0]
        return float(correct_answer.lower().strip()[0] == response.lower()), raw_response

    def select_node(self, node: MCTSNode, total_visits: int) -> MCTSNode:
        while not node.is_leaf():
            if random.random() < self.config.random_prob:
                node = random.choice(node.children)
            else:
                node = node.best_child(
                    self.config.exploration_constant,
                    total_visits,
                    self.config.length_penalty
                )
        return node
    
    def search(self, input_text: str, correct_answer: str, original_correct: float) -> Tuple[LayerPath, float]:
        """Run MCTS to find optimal path for given input"""
        # Initialize root with default path
        root = MCTSNode(self.model.default_path, num_layers=self.model.num_layers)
        evaluated_paths = {}
        best_length = len(self.model.default_path.layers)
        for simulation in range(self.config.num_simulations):
            # (1) Selection
            node = self.select_node(root, simulation + 1)
            
            # (2) Expansion
            if not node.is_fully_expanded() and node.visits > 0:
                node = node.expand()
            
            # (3) Simulation (evaluation)
            if node.path in evaluated_paths:
                reward = evaluated_paths[node.path]
            else:
                reward, _ = self.evaluate_path(input_text, correct_answer, node.path)
                evaluated_paths[node.path] = reward
            if reward > 0.5 and node.path.length < best_length:
                best_length = node.path.length
                
            # (4) Backpropagation
            node.backpropagate(reward)
            if reward > original_correct or (reward > 0.5 and best_length+2 <= self.model.default_path.length):
                break
            # if simulation % 100 == 0:
            #     logger.info(f"[GPU {torch.cuda.current_device()}] MCTS simulation {simulation}/{self.config.num_simulations}")
        good_paths = [p for p, r in evaluated_paths.items() if r > 0.5]
        # logger.info(f"[GPU {torch.cuda.current_device()}] MCTS simulation solution found at {simulation}/{self.config.num_simulations} with GT accuracy {evaluated_paths[self.model.default_path]}")
        return good_paths

def prepare_arc_data(dataset_name: str = "arc_easy", is_instruct: bool=True) -> List[Dict[str, Any]]:
    # put a seed of 42
    random.seed(42)
    if dataset_name == "arc_easy":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    elif dataset_name =="arc_challenge":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    elif "dart" in dataset_name:
        level = int(dataset_name.split("-")[-1])
        # The DART pool is cached as a hub snapshot (parquet), not in the datasets-lib
        # cache, so load_dataset("hkust-nlp/...") fails under HF_HUB_OFFLINE. Load the
        # local parquet files directly (offline-safe). Override dir via DART_POOL_DIR.
        _hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        dart_dir = os.environ.get(
            "DART_POOL_DIR",
            os.path.join(_hf, "datasets/hkust-nlp__dart-math-pool-math/data"))
        try:
            dataset = load_dataset("hkust-nlp/dart-math-pool-math", split="train")
        except Exception:
            dataset = load_dataset("parquet", data_files=os.path.join(dart_dir, "train-*.parquet"), split="train")
        dataset = dataset.filter(lambda x: x['query_metadata']['level'] == level, num_proc=32)
        
    def prepare_arc_sample(item):
        question = item["question"]
        choices = item["choices"]
        answer_key = item["answerKey"]
        choices_text = format_choices(choices["text"]) if is_instruct else format_choices_base(choices["text"])
        prompt_template = answer_letter_long if is_instruct else answer_letter_base
        input_text = prompt_template.format(question=question, choices_text=choices_text)
        labels = choices["label"]
        correct_idx = labels.index(answer_key)
        answer_key = chr(65 + correct_idx)  # Convert index to letter (A, B, C, D)
        return {"input": input_text, "correct": answer_key}
    
    def prepare_dart_sample(item):
        question = item["query"]
        answer_key = item["gt_ans"]
        prompt_template = answer_math if is_instruct else answer_math_base
        input_text = prompt_template.format(question=question)
        return {"input": input_text, "correct": answer_key}
    
    samples = []
    prepare_func = prepare_dart_sample if "dart" in dataset_name else prepare_arc_sample
    for item in tqdm(dataset, desc="Preparing Eval samples"):
        sample = prepare_func(item)
        samples.append(sample)
    
    return samples



def worker_evaluate(rank: int, samples: List[Dict], required_samples, is_train: bool, config: MCTSConfig):
    torch.cuda.set_device(rank)
    model = MCTSModel(config.model_name, rank=rank)
    mcts = MCTS(model, config)

    local_results = {
        "original_correct": [],
        "mcts_correct": [],
        "improved_cases": [],
        "visited_samples": 0,
        "path_lengths": [],
        "unique_layers": [],
        "best_path": [],
        "best_response": [],
        "good_paths": [],
        "questions": []
    }
    # target_accuracy = random.randint(8,20) / 100
    target_accuracy = 0.30
    for i, sample in enumerate(tqdm(samples, desc=f"[GPU {rank}]")):
        original_correct, _ = mcts.evaluate_path(sample["input"], sample["correct"], model.default_path)
        best_paths = mcts.search(sample["input"], sample["correct"], original_correct)
        local_results["visited_samples"] += 1
        is_correct = len(best_paths) > 0 
        if is_train and not is_correct: continue
        if not (float(is_correct) > original_correct) and len(local_results["improved_cases"]) >= required_samples:
            continue
        best_path = sorted(best_paths, key=lambda p: (p.length, p.unique_layers))[0] if best_paths else model.default_path
        if len(local_results["improved_cases"]) < required_samples:
            local_results["original_correct"].append(int(original_correct > 0.5))
            local_results['mcts_correct'].append(int(is_correct))
            local_results["improved_cases"].append(float(is_correct) > original_correct)
            local_results["path_lengths"].append(best_path.length)
            local_results["unique_layers"].append(best_path.unique_layers)
            local_results["best_path"].append(best_path.layers)
            local_results["good_paths"].append([p.layers for p in best_paths])
            local_results["best_response"].append(sample["correct"])
            local_results["questions"].append(sample["input"])
        else:
            first_noncorrect_index = next((i for i, v in enumerate(local_results["improved_cases"]) if not v), None)
            if first_noncorrect_index is None: break
            local_results["original_correct"][first_noncorrect_index] = int(original_correct > 0.5)
            local_results['mcts_correct'][first_noncorrect_index] = int(is_correct)
            local_results["improved_cases"][first_noncorrect_index] = float(is_correct) > original_correct
            local_results["path_lengths"][first_noncorrect_index] = best_path.length 
            local_results["unique_layers"][first_noncorrect_index] = best_path.unique_layers
            local_results["best_path"][first_noncorrect_index] = best_path.layers
            local_results["good_paths"][first_noncorrect_index] = [p.layers for p in best_paths]
            local_results["best_response"][first_noncorrect_index] = sample["correct"]
            local_results["questions"][first_noncorrect_index] = sample["input"]
                
        if len(local_results["improved_cases"]) >= required_samples and \
           sum(local_results["improved_cases"]) / len(local_results["improved_cases"]) >= target_accuracy:
            logger.info(f"[GPU {rank}] Reached target accuracy: {target_accuracy * 100:.1f}%")
            break

    return local_results
    

def evaluate_mcts(args):
    logger.info("Starting parallel MCTS evaluation")
    config = MCTSConfig(num_simulations=args.num_simulations,
                        exploration_constant=args.exploration_constant,
                        length_penalty=args.length_penalty,
                        random_prob=args.random_prob,
                        max_path_length=args.max_path_length,
                        model_name=args.model_name,
                        dataset=args.dataset)
    is_instruct = get_is_instruct(config.model_name)
    all_samples = prepare_arc_data(args.dataset, is_instruct)
    if args.num_shards > 1:
        all_samples = all_samples[args.shard::args.num_shards]  # disjoint strided subset
        logger.info(f"Shard {args.shard}/{args.num_shards}: {len(all_samples)} samples in this shard")

    world_size = torch.cuda.device_count()
    chunk_size = math.ceil(len(all_samples) / world_size)
    chunks = [all_samples[i:i+chunk_size] for i in range(0, len(all_samples), chunk_size)]
    required_samples = math.ceil(args.num_samples // world_size)
    with mp.Pool(world_size) as pool:
        worker_args = [(rank, chunks[rank], required_samples, args.is_train, config) for rank in range(world_size)]
        all_results = pool.starmap(worker_evaluate, worker_args)

    # Aggregate results
    results = {
        "exp": args.exp,
        "original_accuracy": 0.0,
        "mcts_accuracy": 0.0,
        "improved_cases_accuracy": 0.0,
        "average_path_length": 0.0,
        "average_unique_layers": 0.0,
        "num_simulations": config.num_simulations,
        "exploration_constant": config.exploration_constant,
        "length_penalty": config.length_penalty,
        "random_prob": config.random_prob,
        "max_path_length": config.max_path_length,
        "world_size": world_size,
        "num_samples": args.num_samples,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": config.model_name,
        "dataset": args.dataset,
        "visited_samples": 0,
        "skip_size": SKIP_SIZE,
        "repeat_size": REPEAT_SIZE,
        "repeat_count": REPEAT_COUNT,
        "original_correct": [],
        "mcts_correct": [],
        "improved_cases": [],
        "path_lengths": [],
        "unique_layers": [],
        "best_path": [],
        "good_paths": [],
        "best_response": [],
        "questions": []
    }

    for partial in all_results:
        results["original_correct"].extend(partial["original_correct"])
        results["mcts_correct"].extend(partial["mcts_correct"])
        results["improved_cases"].extend(partial["improved_cases"])
        results["path_lengths"].extend(partial["path_lengths"])
        results["unique_layers"].extend(partial["unique_layers"])
        results["best_path"].extend(partial["best_path"])
        results["good_paths"].extend(partial["good_paths"])
        results["best_response"].extend(partial["best_response"])
        results["questions"].extend(partial["questions"])
        results["visited_samples"] += partial["visited_samples"]
    
    num_samples = len(results["best_path"])
    results['original_accuracy'] = sum(results['original_correct']) / num_samples * 100
    results['mcts_accuracy'] = sum(results['mcts_correct']) / num_samples * 100
    results['improved_cases_accuracy'] = sum(results['improved_cases']) / num_samples * 100
    results['average_path_length'] = np.mean(results['path_lengths'])
    results['average_unique_layers'] = np.mean(results['unique_layers'])

    print("\n" + "="*50)
    print("MCTS MCTS EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {len(all_samples)}")
    print(f"Original model accuracy: {results['original_accuracy']:.1f}% ({results['original_correct']}/{len(all_samples)})")
    print(f"MCTS model accuracy: {results['mcts_accuracy']:.1f}% ({results['mcts_correct']}/{len(all_samples)})")
    print(f"Improved cases: {results['improved_cases_accuracy']:.1f}% ({results['improved_cases']}/{len(all_samples)})")
    print(f"Average path length: {results['average_path_length']:.1f}")
    print(f"Average unique layers: {results['average_unique_layers']:.1f}")
    print(f"Number of simulations: {config.num_simulations}")
    print(f"Skip sizes: {SKIP_SIZE}")
    print(f"Repeat sizes: {REPEAT_SIZE}")
    print(f"Repeat counts: {REPEAT_COUNT}")
    print("="*50)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = "data/train" if args.is_train else "predictions"
    os.makedirs(output_dir, exist_ok=True)
    shard_tag = f"_s{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""
    with open(f"{output_dir}/{args.dataset}_results{shard_tag}_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=4)
        
        
def parse_args():
    available_ds = ["arc_easy", "arc_challenge"]
    available_ds += [f"dart-{i}" for i in range(1,6)]
    parser = ArgumentParser(description="Run MCTS evaluation")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--num_simulations", type=int, default=200)
    parser.add_argument("--exploration_constant", type=float, default=1.8)
    parser.add_argument("--length_penalty", type=float, default=3.0)
    parser.add_argument("--random_prob", type=float, default=0.1)
    parser.add_argument("--max_path_length", type=int, default=40)
    parser.add_argument("--dataset", type=str, default="arc_easy", choices=available_ds)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--exp", type=str, default="mcts_parallel")
    parser.add_argument("--is_train", action="store_true")
    # sharding: split one dataset's pool across parallel jobs by a strided slice,
    # so N shards collect disjoint samples (grabs more GPUs, shortens long tails).
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    return parser.parse_args()
    


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    try:
        evaluate_mcts(args)
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()