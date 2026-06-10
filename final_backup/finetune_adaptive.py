"""Adaptive Self-Rehearsal fine-tuning script for Qwen2.5-3B.

Mirrors finetune.py exactly EXCEPT:
- Replaces uniform dataset shuffling with weighted sampling
- Adds probe-based domain monitoring every PROBE_EVAL_EVERY steps
- When per-domain accuracy drops more than FORGETTING_THRESHOLD pp below
  baseline for 2 consecutive evaluations, multiplies the sample weights of
  in-domain instructions by REWEIGHT_FACTOR.

Environment variables (same defaults as finetune.py where applicable):
- SEED                 : random seed (default 42)
- EPOCHS               : number of epochs (default 4)
- OUTPUT_DIR_OVERRIDE  : override output directory
- PROBE_EVAL_EVERY     : probe eval frequency in steps (default 25)
- FORGETTING_THRESHOLD : forgetting trigger threshold in pp (default 10.0)
- REWEIGHT_FACTOR      : multiplier per intervention (default 1.5)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, TrainerCallback,
)

# =============================================================================
# Configuration
# =============================================================================
SEED   = int(os.environ.get("SEED", 42))
EPOCHS = int(os.environ.get("EPOCHS", 4))
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR_OVERRIDE",
    f"./qwen-sft-seed{SEED}-ep{EPOCHS}-adaptive"
)

PROBE_EVAL_EVERY     = int(os.environ.get("PROBE_EVAL_EVERY", 25))
FORGETTING_THRESHOLD = float(os.environ.get("FORGETTING_THRESHOLD", 10.0))
REWEIGHT_FACTOR      = float(os.environ.get("REWEIGHT_FACTOR", 1.5))

MODEL_NAME        = "Qwen/Qwen2.5-3B"
INSTRUCTIONS_PATH = "generated_data/self_instruct_generated.json"
DOMAINS_PATH      = "instruction_domains.json"
PROBES_PATH       = "probes/probe_set.json"
MAX_LENGTH        = 512  # matches finetune.py

print(f"Adaptive SFT")
print(f"  SEED={SEED} EPOCHS={EPOCHS}")
print(f"  OUTPUT_DIR={OUTPUT_DIR}")
print(f"  PROBE_EVAL_EVERY={PROBE_EVAL_EVERY}")
print(f"  FORGETTING_THRESHOLD={FORGETTING_THRESHOLD}pp")
print(f"  REWEIGHT_FACTOR={REWEIGHT_FACTOR}")

# =============================================================================
# Model + tokenizer
# =============================================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

# =============================================================================
# Load instructions + domain labels
# =============================================================================
with open(INSTRUCTIONS_PATH) as f:
    instructions = json.load(f)
print(f"Loaded {len(instructions)} instructions")

with open(DOMAINS_PATH) as f:
    domain_labels = json.load(f)["labels"]
assert len(domain_labels) == len(instructions), \
    f"Domain labels ({len(domain_labels)}) != instructions ({len(instructions)})"

# =============================================================================
# Tokenize
#
# Format matches finetune.py exactly — Alpaca-style template with
# "Below is an instruction..." preamble, flat input/output fields.
# =============================================================================
def format_text(instr):
    """Format one instruction record into a training string.

    Matches finetune.py format exactly for a clean ablation.
    """
    if instr.get("input", "").strip():
        return (
            f"Below is an instruction that describes a task, paired with an input. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instr['instruction']}\n\n"
            f"### Input:\n{instr['input']}\n\n"
            f"### Response:\n{instr['output']}{tokenizer.eos_token}"
        )
    else:
        return (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instr['instruction']}\n\n"
            f"### Response:\n{instr['output']}{tokenizer.eos_token}"
        )

def tokenize_one(text):
    enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    enc["labels"] = [
        -100 if tok == tokenizer.pad_token_id else tok
        for tok in enc["input_ids"]
    ]
    return enc

tokenized = [tokenize_one(format_text(instr)) for instr in instructions]
print(f"Tokenized {len(tokenized)} examples")

# =============================================================================
# Adaptive dataset
#
# Trick: Trainer requests indices 0..N-1, but our __getitem__ ignores the
# requested index and returns a weighted-random sample instead. The weights
# tensor is mutated in place by the callback when forgetting is detected.
# =============================================================================
class AdaptiveDataset(Dataset):
    def __init__(self, examples, labels, seed):
        self.examples = examples
        self.labels   = labels
        self.weights  = np.ones(len(examples), dtype=np.float64)
        self.rng      = np.random.default_rng(seed)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, _idx):
        # Ignore requested idx — sample by weights
        p = self.weights / self.weights.sum()
        sampled_idx = int(self.rng.choice(len(self.examples), p=p))
        return self.examples[sampled_idx]

dataset = AdaptiveDataset(tokenized, domain_labels, seed=SEED)

# =============================================================================
# Load probes + define evaluator
# =============================================================================
with open(PROBES_PATH) as f:
    probes = json.load(f)
total_probes = sum(len(qs) for qs in probes.values())
print(f"Loaded {total_probes} probes across {len(probes)} domains")

@torch.no_grad()
def evaluate_probes(model, tokenizer, probes, device):
    """Per-domain accuracy via per-choice log-likelihood scoring."""
    model.eval()
    domain_accs = {}
    for domain, questions in probes.items():
        correct = 0
        for q in questions:
            scores = []
            for choice in q["choices"]:
                text = q["question"] + " " + str(choice)
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                out = model(**enc, labels=enc["input_ids"])
                # -loss * seq_len is proportional to log-likelihood
                scores.append(-out.loss.item() * enc["input_ids"].size(1))
            pred = int(np.argmax(scores))
            if pred == int(q["answer"]):
                correct += 1
        domain_accs[domain] = correct / max(len(questions), 1)
    model.train()
    return domain_accs

# =============================================================================
# Adaptive callback
# =============================================================================
class AdaptiveCallback(TrainerCallback):
    def __init__(self, dataset, probes, tokenizer,
                 eval_every=25, threshold_pp=10.0, reweight=1.5):
        self.dataset       = dataset
        self.probes        = probes
        self.tokenizer     = tokenizer
        self.eval_every    = eval_every
        self.threshold     = threshold_pp / 100.0  # store as fraction
        self.reweight      = reweight
        self.baseline_accs = None
        self.consecutive   = {d: 0 for d in probes}
        self.history       = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_every != 0:
            return

        device = next(model.parameters()).device
        accs = evaluate_probes(model, self.tokenizer, self.probes, device)

        record = {
            "step": int(state.global_step),
            "accs": {d: float(a) for d, a in accs.items()},
            "interventions": [],
            "weights_min": float(self.dataset.weights.min()),
            "weights_max": float(self.dataset.weights.max()),
            "weights_mean": float(self.dataset.weights.mean()),
        }

        if self.baseline_accs is None:
            self.baseline_accs = dict(accs)
            print(f"[Adaptive] Step {state.global_step}: baseline {accs}")
            self.history.append(record)
            return

        for domain, baseline in self.baseline_accs.items():
            drop = baseline - accs[domain]
            if drop > self.threshold:
                self.consecutive[domain] += 1
                if self.consecutive[domain] >= 2:
                    indices = [i for i, d in enumerate(self.dataset.labels) if d == domain]
                    if indices:
                        self.dataset.weights[indices] *= self.reweight
                        # Renormalize so weights.mean() stays ~1 (avoids unbounded drift)
                        self.dataset.weights *= (len(self.dataset.weights) / self.dataset.weights.sum())
                        record["interventions"].append({
                            "domain": domain,
                            "drop_pp": float(drop * 100),
                            "factor": self.reweight,
                            "n_boosted": len(indices),
                        })
                        print(f"[Adaptive] Step {state.global_step}: INTERVENE on '{domain}' "
                              f"(drop {drop*100:.1f}pp, boosted {len(indices)} examples by {self.reweight}x)")
                        self.consecutive[domain] = 0  # reset after intervention
            else:
                self.consecutive[domain] = 0

        self.history.append(record)

# =============================================================================
# Training args — matches finetune.py exactly (server_backup version)
# =============================================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, seed=SEED,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=25,
    save_steps=10000,
    save_total_limit=1,
    report_to="none",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    dataloader_pin_memory=True,
)

callback = AdaptiveCallback(
    dataset=dataset,
    probes=probes,
    tokenizer=tokenizer,
    eval_every=PROBE_EVAL_EVERY,
    threshold_pp=FORGETTING_THRESHOLD,
    reweight=REWEIGHT_FACTOR,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    callbacks=[callback],
)

print("\nStarting training...\n")
trainer.train()

# =============================================================================
# Save outputs
# =============================================================================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

trajectory_path = os.path.join(OUTPUT_DIR, "probe_trajectory.json")
with open(trajectory_path, "w") as f:
    json.dump(callback.history, f, indent=2)

n_interventions = sum(len(h["interventions"]) for h in callback.history)
print(f"\nDone.")
print(f"  Model:      {OUTPUT_DIR}")
print(f"  Trajectory: {trajectory_path}")
print(f"  Interventions fired: {n_interventions}")
print(f"  Final weights: min={dataset.weights.min():.3f} "
      f"max={dataset.weights.max():.3f} mean={dataset.weights.mean():.3f}")
