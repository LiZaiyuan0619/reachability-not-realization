"""
we put a router over each layer of the model,
the router takes in the mean if the input hidden states and determines once at the inference for each layer
whether to skip it, execute it once, loop once over the layer
router_l = MlP(hidden_size, 3)
decision_l = argmax(softmax(router_l(hidden_states.mean(dim=1))))

the training data has (question, optimal_layer_configuration, answer)
sample ("What is 12*3?", [1, 2, 2, 6, 9], "36")

training objective for each router is just the cross entropy loss between the decision and the optimal_layer_configuration
"""

import torch
from transformers import AutoTokenizer, LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, OlmoeForCausalLM
import json
from glob import glob
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from tabulate import tabulate
import os
from argparse import ArgumentParser
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
parser.add_argument("--run_name", type=str, default="drllm-llama-3b-singleloss-2ep")
parser.add_argument("--gradient_accumulation", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--num_epochs", type=int, default=15)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_steps", type=int, default=500)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("-w", "--num_windows", type=int, default=None)
parser.add_argument("--with_squad", action="store_true", default=False)
parser.add_argument("--with_commonsense", action="store_true", default=False)
args = parser.parse_args()

model_id = args.model_id
run_name = args.run_name
data_folders = ["data/train/target_30", "data/train/target_12", "data/train"]
batch_size = 1
gradient_accumulation = args.gradient_accumulation
num_epochs = args.num_epochs
learning_rate = args.learning_rate
weight_decay = args.weight_decay
save_steps = args.save_steps
warmup_steps = args.warmup_steps
output_dir = f"checkpoints/{run_name}"
eval_size_per_type = 10
os.makedirs(output_dir, exist_ok=True)
cached_train_data_path = f"cached_data/train_data_{os.path.basename(model_id)}"
cached_eval_data_path = f"cached_data/eval_data_{os.path.basename(model_id)}"

def get_model_cls(name):
    if "llama" in name.lower():
        return LlamaForCausalLM
    elif "qwen3" in name.lower():
        return Qwen3ForCausalLM
    elif "qwen2" in name.lower():
        return Qwen2ForCausalLM
    elif "olmoe" in name.lower():
        return OlmoeForCausalLM
    else:
        raise ValueError(f"Model {name} not supported")
MODEL_CLS = get_model_cls(model_id)
model = MODEL_CLS.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
model.model.init_routers()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if MODEL_CLS == LlamaForCausalLM:
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"
if args.num_windows:
    model.model.num_windows = args.num_windows
if os.path.exists(cached_train_data_path) and os.path.exists(cached_eval_data_path) and False:
    train_data = Dataset.load_from_disk(cached_train_data_path)
    eval_data = Dataset.load_from_disk(cached_eval_data_path)
    print(f"Loaded cached data: {len(train_data)} training samples, {len(eval_data)} eval samples.")
else:
    data_files = [f for data_folder in data_folders for f in glob(f"{data_folder}/*.json")]
    questions, layer_configs, answers, ds_types, orig_corrects = [], [], [], [], []
    for ds_path in tqdm(data_files, desc="Loading data files"):
        with open(ds_path, "r") as f:
            raw_data = json.load(f)
        if model_id != raw_data['model_name']: continue
        if "squadv2" in raw_data['dataset'].lower() and not args.with_squad: continue
        if "commonsense" in raw_data['dataset'].lower() and not args.with_commonsense: continue
        questions.extend(raw_data["questions"])
        # layer_configs.extend(raw_data["good_paths"])
        layer_configs.extend(raw_data["best_path"])
        if "squadv2" in raw_data['dataset'].lower():
            raw_data["best_response"] =[a['text'][0] if len(a['text']) > 0 else "unanswerable" for a in raw_data["best_response"]]
        answers.extend(raw_data["best_response"])
        ds_types.extend([raw_data['dataset']] * len(raw_data["questions"]))
        orig_corrects.extend(raw_data['original_correct'])
    print(f"Loaded {len(questions)} samples from {len(data_files)} files.")

    data = []
    eval_data = {}
    repeated, skipped, executed, total = 0, 0, 0, 0
    if MODEL_CLS == LlamaForCausalLM:
        split_pattern = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif MODEL_CLS == Qwen2ForCausalLM or MODEL_CLS == Qwen3ForCausalLM:
        split_pattern = "<|im_start|>assistant\n"
    qs = set()
    for question, layer_config, answer, ds_type, orig_correct in zip(questions, layer_configs, answers, ds_types, orig_corrects):
        if question in qs: continue
        else: qs.add(question)
        if "instruct" in model_id.lower():
            conversations = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            msg = tokenizer.apply_chat_template(conversations, tokenize=False)
            prompt, completion = msg.split(split_pattern)
            prompt += split_pattern
        else:
            prompt, completion = question, answer
        # for layer_config in layer_configs:
        layer_decisions = [layer_config.count(idx) for idx in range(len(model.model.layers))]
        sample = {"prompt": prompt, "completion": completion, "layer_decisions": layer_decisions}
        
        
        if len(eval_data.get(ds_type, [])) < 1 or (len(eval_data.get(ds_type, [])) < eval_size_per_type and eval_data[ds_type][-1]['prompt'] != prompt):
            eval_data[ds_type] = eval_data.get(ds_type, [])
            eval_data[ds_type].append(sample)
        else:
            data.append(sample)
            total += len(layer_decisions)
            skipped += layer_decisions.count(0)
            executed += layer_decisions.count(1)
            repeated += layer_decisions.count(2)
            
    print(f"Data distribution is {skipped=}, {executed=}, {repeated=}")
    eval_data = [item for sublist in eval_data.values() for item in sublist]
    train_data = Dataset.from_list(data)
    eval_data = Dataset.from_list(eval_data)
    train_data.save_to_disk(cached_train_data_path)
    eval_data.save_to_disk(cached_eval_data_path)
    print(f"Training on {len(train_data)} samples, evaluating on {len(eval_data)} samples.")


training_args = SFTConfig(
    do_train=True,
    do_eval=False,
    do_predict=False,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation,
    per_device_eval_batch_size=1,
    eval_strategy="no",
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    logging_steps=2,
    save_steps=save_steps,
    output_dir=output_dir,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    fp16=False,
    bf16=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    run_name=run_name,
    report_to="wandb",
    prediction_loss_only=False,
)
for p in model.model.parameters():
    p.requires_grad = False
for p in model.model.routers.parameters():
    p.requires_grad = True
for p in model.lm_head.parameters():
    p.requires_grad = False

stat = []
for i, (name, p) in enumerate(model.named_parameters()):
    stat.append([i, name, p.shape, p.requires_grad])
print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))
model.model_accepts_loss_kwargs = False
trainer = SFTTrainer(
    args=training_args,
    model=model,
    train_dataset=train_data,
    processing_class=tokenizer,
)
trainer.train()

