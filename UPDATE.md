# UPDATE: Adaptive Self-Rehearsal Implementation

## Context for the implementer

This update adds a **feasibility study** of Adaptive Self-Rehearsal to an existing forgetting-experiment codebase. The existing repository contains a working vanilla Self-Instruct SFT pipeline that has already produced multi-seed motivation data on Qwen2.5-3B showing selective domain-specific forgetting. This update layers an adaptive mechanism on top: probe-based per-domain monitoring during training, with sampler reweighting when forgetting is detected.

**The goal:** produce paired (same-seed) results comparing vanilla SFT vs adaptive SFT to demonstrate whether the adaptive mechanism preserves the worst-forgotten capabilities without sacrificing the gains.

**Do not modify existing files.** This entire update is *additive* — all new files. The existing `finetune.py`, `compare.py`, `visualize.py`, `benchmark_instruct.sh`, etc. must remain unchanged so the vanilla baseline stays clean and reproducible.

---

## Project state assumed

The existing repository structure (after Claude Code reads it):

```
~/self_instruct/server_backup/   (local mirror of server)
├── finetune.py                  ← EXISTING vanilla SFT. DO NOT MODIFY.
├── benchmark_instruct.sh        ← EXISTING. Reusable for adaptive models.
├── compare.py                   ← EXISTING. Motivation-phase comparison.
├── visualize.py                 ← EXISTING. Motivation-phase figures.
├── run_seeds.sh                 ← EXISTING. Vanilla multi-seed wrapper.
├── bootstrap.py
├── benchmark_base.sh
├── check_gpu.sh
├── setup.sh
├── generated_data/
│   └── self_instruct_generated.json    ← 2000 Self-Instruct-generated instructions
├── results/
│   ├── base/Qwen__Qwen2.5-3B/           ← base benchmarks (existing)
│   ├── instruct/seed{42,1337,2025}-ep4/ ← vanilla SFT results (existing)
│   └── figures/                          ← motivation-phase figures (existing)
└── logs/
```

**Required reading before writing code:** Open `finetune.py` and note (a) the prompt template format used to combine instruction + input + output into a training example, and (b) the `max_length` used for tokenization. The adaptive script must match these exactly so the only methodological difference between vanilla and adaptive is the adaptive mechanism itself.

---

## Files to create

All files are new. None modify existing code.

| File | Purpose |
|---|---|
| `finetune_adaptive.py` | Adaptive SFT training script (mirrors `finetune.py` + adds probe monitoring + sampler reweighting) |
| `build_probe_set.py` | One-time script to construct the 150-question held-out probe set from HF dataset train splits |
| `instruction_classifier.py` | One-time script to classify each of the 2000 self-instruct instructions into probe domains via keyword regex |
| `run_adaptive_seeds.sh` | Wrapper to run 3 adaptive seeds + benchmark each (mirrors `run_seeds.sh` pattern) |
| `compare_ablation.py` | Paired comparison of vanilla vs adaptive results |
| `visualize_ablation.py` | Ablation comparison figures (4 plots) |
| `probes/.gitkeep` | Placeholder for the `probes/` directory |
| `results/adaptive/.gitkeep` | Placeholder for the adaptive results directory |

---

## File 1: `finetune_adaptive.py`

This is the largest new file. **Before writing it, read `finetune.py` and copy the prompt formatting / max_length exactly** — adjust the `format_text` and `MAX_LENGTH` placeholders below to match.

```python
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
MAX_LENGTH        = 512   # MUST match finetune.py — verify and adjust if needed

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
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

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
# IMPORTANT: This format MUST match finetune.py exactly.
# Look at finetune.py and adapt the format_text() function below if needed.
# The standard Self-Instruct format is shown here (Alpaca-style).
# =============================================================================
def format_text(instr):
    """Format one instruction record into a training string.
    
    MUST match the format used in finetune.py. The format below is the
    standard Self-Instruct (Alpaca-style) template; verify against finetune.py
    and adjust if it uses a different template.
    """
    instruction = instr["instruction"]
    input_text  = instr.get("instances", [{}])[0].get("input", "") if instr.get("instances") else ""
    output_text = instr.get("instances", [{}])[0].get("output", "") if instr.get("instances") else ""
    
    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )
    return prompt + output_text

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
        self.threshold     = threshold_pp / 100.0   # store as fraction
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
# Training args — MUST MATCH finetune.py exactly for clean ablation
# =============================================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    seed=SEED,
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
```

---

## File 2: `build_probe_set.py`

```python
"""Build the held-out probe set for adaptive monitoring.

Samples 30 multiple-choice questions per domain from HF dataset TRAIN splits
(the lm-eval benchmark suite uses test/validation, so no overlap).

5 domains × 30 questions = 150 probes total.

Datasets pulled:
- formal_quant         : allenai/ai2_arc ARC-Challenge train, filtered for math/science keywords
- math_words           : gsm8k train (with MC conversion: correct answer + 3 numeric distractors)
- causal_commonsense   : Rowan/hellaswag train (4-choice MC)
- comprehension        : boolq train (2-choice yes/no)
- factual              : sciq train (4-choice science Q&A)
"""

import json, os, random
from datasets import load_dataset

random.seed(0)
NUM_PER_DOMAIN = 30
OUTPUT_PATH = "probes/probe_set.json"
os.makedirs("probes", exist_ok=True)

def build_formal_quant():
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    keywords = ["calculate", "equation", "force", "mass", "energy", "atom", "molecule",
                "physics", "chemistry", "rate", "ratio", "fraction", "function",
                "speed", "velocity", "current", "voltage", "wave", "acid", "electron"]
    candidates = [it for it in ds if any(kw in it["question"].lower() for kw in keywords)]
    if len(candidates) < NUM_PER_DOMAIN:
        candidates = list(ds)
    random.shuffle(candidates)
    out = []
    for it in candidates:
        if len(out) >= NUM_PER_DOMAIN: break
        choices = it["choices"]["text"]
        labels  = it["choices"]["label"]
        try:
            ans_idx = labels.index(it["answerKey"])
        except ValueError:
            continue
        out.append({"question": it["question"], "choices": choices, "answer": ans_idx})
    return out

def build_math_words():
    ds = load_dataset("gsm8k", "main", split="train")
    items = random.sample(list(ds), NUM_PER_DOMAIN * 3)
    out = []
    for it in items:
        if len(out) >= NUM_PER_DOMAIN: break
        ans_str = it["answer"].split("####")[-1].strip().replace(",", "").replace("$", "")
        try:
            correct = int(float(ans_str))
        except ValueError:
            continue
        # Build 3 numeric distractors
        distractors = set()
        while len(distractors) < 3:
            delta = random.choice([-1, 1, -2, 2, -5, 5, -10, 10, -3, 3])
            cand = correct + delta
            if cand != correct and cand >= 0:
                distractors.add(cand)
        choices = [str(correct)] + [str(d) for d in distractors]
        random.shuffle(choices)
        ans_idx = choices.index(str(correct))
        out.append({
            "question": it["question"] + "\nWhat is the answer?",
            "choices": choices,
            "answer": ans_idx,
        })
    return out

def build_causal_commonsense():
    ds = load_dataset("Rowan/hellaswag", split="train")
    items = random.sample(list(ds), NUM_PER_DOMAIN)
    return [
        {
            "question": it["ctx"] + " What happens next?",
            "choices": it["endings"],
            "answer": int(it["label"]),
        }
        for it in items
    ]

def build_comprehension():
    ds = load_dataset("boolq", split="train")
    items = random.sample(list(ds), NUM_PER_DOMAIN)
    return [
        {
            "question": it["passage"] + "\n\nQuestion: " + it["question"],
            "choices": ["No", "Yes"],
            "answer": int(it["answer"]),
        }
        for it in items
    ]

def build_factual():
    ds = load_dataset("sciq", split="train")
    items = random.sample(list(ds), NUM_PER_DOMAIN)
    out = []
    for it in items:
        choices = [it["distractor1"], it["distractor2"], it["distractor3"], it["correct_answer"]]
        random.shuffle(choices)
        out.append({
            "question": it["question"],
            "choices": choices,
            "answer": choices.index(it["correct_answer"]),
        })
    return out

probes = {
    "formal_quant":       build_formal_quant(),
    "math_words":         build_math_words(),
    "causal_commonsense": build_causal_commonsense(),
    "comprehension":      build_comprehension(),
    "factual":            build_factual(),
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(probes, f, indent=2)

print(f"Built probe set: {OUTPUT_PATH}")
for d, qs in probes.items():
    print(f"  {d:<22}: {len(qs)} questions")
```

---

## File 3: `instruction_classifier.py`

```python
"""Classify each self-instruct instruction into a probe domain via regex.

Output: instruction_domains.json with {"labels": [str * N]} parallel to the
instruction list. Order matches generated_data/self_instruct_generated.json.

Heuristic classification — fast and defensible but imperfect. Acknowledged
as a limitation in the methodology writeup.
"""

import json, re

INSTRUCTIONS_PATH = "generated_data/self_instruct_generated.json"
OUTPUT_PATH       = "instruction_domains.json"

# Patterns ordered by specificity — first match wins.
PATTERNS = [
    ("formal_quant", [
        r"\b(prove|theorem|axiom|lemma|differentiate|integrate|integral|derivative)\b",
        r"\b(algebra|topology|set theory|group theory|polynomial|matrix|determinant)\b",
        r"\b(quantum|relativity|kinetic|potential energy|electric field|magnetic field)\b",
        r"\b(formal logic|propositional|first-order|predicate logic|inference rule|truth table)\b",
        r"\b(force|momentum|wavelength|frequency|voltage|current|circuit)\b",
    ]),
    ("math_words", [
        r"\b(calculate|compute|how many|how much|what is the (total|sum|product|average|difference))\b",
        r"\b(percentage|percent of|fraction of|ratio of)\b",
        r"\b\d+\s*(plus|minus|times|divided by|\+|\-|\*|/)\s*\d+",
        r"\bif (he|she|they|the|a|each) (has|have|had|bought|sold|gives|gave)\b",
    ]),
    ("causal_commonsense", [
        r"\b(because|therefore|hence|thus|as a result|leads to|causes|caused by)\b",
        r"\b(what would happen if|what happens when|what's the (result|effect|consequence))\b",
        r"\b(why does|why is|why would|why did)\b",
    ]),
    ("comprehension", [
        r"\b(read the following|given the passage|according to the (text|passage|article))\b",
        r"\b(summarize|paraphrase|explain in your own words)\b",
        r"\b(is the following statement (true|false))\b",
        r"\b(what does (the author|this) (mean|imply|suggest))\b",
    ]),
    ("factual", [
        r"\b(who (is|was|invented|wrote|discovered|founded))\b",
        r"\b(what (is|was) the (capital|population|currency|language|name) of)\b",
        r"\b(when (did|was|is)|in what year|in which year)\b",
        r"\b(where (is|was|did))\b",
        r"\b(list (the|some|all))\b",
        r"\b(name (the|a|some|several))\b",
    ]),
]

def classify(text):
    text_lower = text.lower()
    for domain, patterns in PATTERNS:
        for p in patterns:
            if re.search(p, text_lower):
                return domain
    return "other"

with open(INSTRUCTIONS_PATH) as f:
    instructions = json.load(f)

labels = []
counts = {"formal_quant": 0, "math_words": 0, "causal_commonsense": 0,
          "comprehension": 0, "factual": 0, "other": 0}

for instr in instructions:
    text = instr.get("instruction", "")
    if "instances" in instr and instr["instances"]:
        text += " " + str(instr["instances"][0].get("input", ""))
    label = classify(text)
    labels.append(label)
    counts[label] += 1

with open(OUTPUT_PATH, "w") as f:
    json.dump({"labels": labels}, f)

print(f"Classified {len(labels)} instructions into domains:")
for d, c in counts.items():
    print(f"  {d:<22}: {c:>4}  ({100*c/len(labels):.1f}%)")
print(f"\nSaved to {OUTPUT_PATH}")
```

---

## File 4: `run_adaptive_seeds.sh`

```bash
#!/bin/bash
# Run adaptive SFT across 3 paired seeds (same seeds as vanilla) and benchmark each.

set -e

SEEDS=(42 1337 2025)
EPOCHS=${EPOCHS:-4}

mkdir -p logs results/adaptive

for SEED in "${SEEDS[@]}"; do
    OUTPUT_DIR="./qwen-sft-seed${SEED}-ep${EPOCHS}-adaptive"
    BENCHMARK_OUT="results/adaptive/seed${SEED}-ep${EPOCHS}-adaptive"

    echo "================================================================"
    echo "Seed $SEED: ADAPTIVE SFT training"
    echo "================================================================"
    SEED=$SEED EPOCHS=$EPOCHS OUTPUT_DIR_OVERRIDE="$OUTPUT_DIR" \
        python finetune_adaptive.py 2>&1 | tee "logs/finetune_adaptive_seed${SEED}.log"

    echo "================================================================"
    echo "Seed $SEED: Benchmarking adaptive model"
    echo "================================================================"
    MODEL="$OUTPUT_DIR" OUT="$BENCHMARK_OUT" \
        bash benchmark_instruct.sh 2>&1 | tee "logs/benchmark_adaptive_seed${SEED}.log"

    # Save trajectory alongside benchmarks for easy access
    if [ -f "$OUTPUT_DIR/probe_trajectory.json" ]; then
        cp "$OUTPUT_DIR/probe_trajectory.json" "$BENCHMARK_OUT/"
    fi
done

echo "All adaptive seeds complete."
```

---

## File 5: `compare_ablation.py`

```python
"""Compare vanilla SFT vs adaptive SFT — paired across same seeds.

Loads:
- results/base/                            : base model benchmarks
- results/instruct/seed*-ep4/              : vanilla SFT (existing)
- results/adaptive/seed*-ep4-adaptive/     : adaptive SFT (this update)

Computes paired per-task differences (adaptive_i - vanilla_i) where
seeds match, then aggregates with mean ± SE.
"""

import json, os, glob
import numpy as np

METRIC_KEYS = [
    "acc_norm,none", "acc,none", "exact_match,none",
    "exact_match,strict-match", "exact_match,flexible-extract",
    "exact_match,remove_whitespace",
]

def load_scores_from_dir(results_dir):
    scores = {}
    for root, _, files in os.walk(results_dir):
        for fn in files:
            if fn.startswith("results_") and fn.endswith(".json"):
                with open(os.path.join(root, fn)) as f:
                    data = json.load(f)
                if "results" in data:
                    for task, metrics in data["results"].items():
                        for key in METRIC_KEYS:
                            if key in metrics:
                                scores[task] = round(metrics[key] * 100, 2)
                                break
    return scores

def load_multi(parent, pattern, strip_suffix=""):
    per_seed = {}
    for d in sorted(glob.glob(os.path.join(parent, pattern))):
        if os.path.isdir(d):
            key = os.path.basename(d).replace(strip_suffix, "")
            scores = load_scores_from_dir(d)
            if scores:
                per_seed[key] = scores
    return per_seed

def mean_se(values):
    arr = np.array(values, dtype=float)
    m = float(arr.mean()) if len(arr) else 0.0
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return m, se

base     = load_scores_from_dir("results/base")
vanilla  = load_multi("results/instruct", "seed*-ep4")
adaptive = load_multi("results/adaptive", "seed*-ep4-adaptive", strip_suffix="-adaptive")

print(f"Base: {len(base)} tasks")
print(f"Vanilla seeds:  {list(vanilla.keys())}")
print(f"Adaptive seeds: {list(adaptive.keys())}")

common = sorted(set(vanilla) & set(adaptive))
print(f"Paired seeds: {len(common)} — {common}")
if len(common) < 3:
    print(f"WARNING: fewer than 3 paired seeds; results will be noisier.")

# Top-level table
all_tasks = set().union(*[s.keys() for s in vanilla.values()])
top_tasks = sorted(t for t in all_tasks if not t.startswith("mmlu_") or t == "mmlu")

print(f"\n{'='*120}")
print(f"TOP-LEVEL TASKS: vanilla vs adaptive (paired, n={len(common)})")
print(f"{'='*120}")
print(f"{'Task':<22} {'Base':>7}  {'Vanilla':>18}  {'Adaptive':>18}  {'Δ (paired)':>16}  {'flag':>5}")
print("-" * 120)

rows = []
for task in top_tasks:
    vv = [vanilla[s].get(task)  for s in common if task in vanilla[s]]
    av = [adaptive[s].get(task) for s in common if task in adaptive[s]]
    if not vv or not av: continue
    
    vm, vse = mean_se(vv)
    am, ase = mean_se(av)
    pairs   = [adaptive[s][task] - vanilla[s][task]
               for s in common if task in vanilla[s] and task in adaptive[s]]
    dm, dse = mean_se(pairs)
    
    flag = ""
    if dse > 0 and abs(dm) > 2 * dse:
        flag = "★"
    
    b = base.get(task, 0.0)
    print(f"{task:<22} {b:>7.2f}  {vm:>9.2f} ± {vse:<5.2f}   {am:>9.2f} ± {ase:<5.2f}   {dm:>+7.2f} ± {dse:<5.2f}   {flag:>5}")
    rows.append({"task": task, "base": b,
                 "vanilla_mean": vm, "vanilla_se": vse,
                 "adaptive_mean": am, "adaptive_se": ase,
                 "diff_mean": dm, "diff_se": dse})

# Key MMLU subjects — these are what should improve if the method works
KEY = ["mmlu_abstract_algebra", "mmlu_college_physics", "mmlu_formal_logic",
       "mmlu_college_chemistry", "mmlu_electrical_engineering"]

print(f"\n{'='*120}")
print(f"KEY MMLU SUBJECTS (worst-forgotten in vanilla — adaptive should recover these)")
print(f"{'='*120}")
print(f"{'Subject':<32} {'Base':>7}  {'Vanilla':>18}  {'Adaptive':>18}  {'Δ (paired)':>16}")
print("-" * 120)

for task in KEY:
    vv = [vanilla[s].get(task)  for s in common if task in vanilla[s]]
    av = [adaptive[s].get(task) for s in common if task in adaptive[s]]
    if not vv or not av: continue
    
    vm, vse = mean_se(vv); am, ase = mean_se(av)
    pairs   = [adaptive[s][task] - vanilla[s][task]
               for s in common if task in vanilla[s] and task in adaptive[s]]
    dm, dse = mean_se(pairs)
    b = base.get(task, 0.0)
    print(f"{task:<32} {b:>7.2f}  {vm:>9.2f} ± {vse:<5.2f}   {am:>9.2f} ± {ase:<5.2f}   {dm:>+7.2f} ± {dse:<5.2f}")

with open("results/ablation_table.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nFull table: results/ablation_table.json")
print(f"\n★ flag = paired Δ exceeds 2 SE (suggestive but not formal significance at n=3)")
```

---

## File 6: `visualize_ablation.py`

```python
"""Ablation figures: base / vanilla / adaptive three-way comparison."""

import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRIC_KEYS = [
    "acc_norm,none", "acc,none", "exact_match,none",
    "exact_match,strict-match", "exact_match,flexible-extract",
    "exact_match,remove_whitespace",
]

def load_scores_from_dir(results_dir):
    scores = {}
    for root, _, files in os.walk(results_dir):
        for fn in files:
            if fn.startswith("results_") and fn.endswith(".json"):
                with open(os.path.join(root, fn)) as f:
                    data = json.load(f)
                if "results" in data:
                    for task, metrics in data["results"].items():
                        for key in METRIC_KEYS:
                            if key in metrics:
                                scores[task] = round(metrics[key] * 100, 2)
                                break
    return scores

def load_multi(parent, pattern, strip_suffix=""):
    per_seed = {}
    for d in sorted(glob.glob(os.path.join(parent, pattern))):
        if os.path.isdir(d):
            key = os.path.basename(d).replace(strip_suffix, "")
            scores = load_scores_from_dir(d)
            if scores:
                per_seed[key] = scores
    return per_seed

def mean_se(values):
    arr = np.array(values, dtype=float)
    m = float(arr.mean()) if len(arr) else 0.0
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return m, se

base     = load_scores_from_dir("results/base")
vanilla  = load_multi("results/instruct", "seed*-ep4")
adaptive = load_multi("results/adaptive", "seed*-ep4-adaptive", strip_suffix="-adaptive")
common = sorted(set(vanilla) & set(adaptive))
n = len(common)
print(f"Common seeds: {common}")

os.makedirs("results/figures", exist_ok=True)

# -----------------------------------------------------------------------------
# Figure 1: top-level 3-way (base / vanilla / adaptive)
# -----------------------------------------------------------------------------
TOP = ["arc_easy", "arc_challenge", "boolq", "rte", "copa", "gsm8k",
       "lambada_openai", "piqa", "hellaswag", "winogrande", "openbookqa",
       "triviaqa", "mmlu"]
TOP = [t for t in TOP if t in base]

bv = [base[t] for t in TOP]
vm = []; vse = []; am = []; ase = []
for t in TOP:
    m, s = mean_se([vanilla[k][t]  for k in common if t in vanilla[k]]);  vm.append(m); vse.append(s)
    m, s = mean_se([adaptive[k][t] for k in common if t in adaptive[k]]); am.append(m); ase.append(s)

fig, ax = plt.subplots(figsize=(15, 6))
x = np.arange(len(TOP)); w = 0.27
ax.bar(x - w, bv, w, label="Base",            color="#3274A1")
ax.bar(x,     vm, w, yerr=vse, label=f"Vanilla SFT (n={n})",  color="#E1812C",
       error_kw=dict(ecolor='black', capsize=3))
ax.bar(x + w, am, w, yerr=ase, label=f"Adaptive SFT (n={n})", color="#3A923A",
       error_kw=dict(ecolor='black', capsize=3))
ax.set_xticks(x)
ax.set_xticklabels([t.replace("_", " ").title() for t in TOP], rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Ablation: Base vs Vanilla SFT vs Adaptive SFT (mean ± SE, n={n})", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("results/figures/ablation_toplevel.png", dpi=150); plt.close()
print("Saved: ablation_toplevel.png")

# -----------------------------------------------------------------------------
# Figure 2: paired (adaptive - vanilla) per task
# -----------------------------------------------------------------------------
diffs = []; ds = []
for t in TOP:
    pairs = [adaptive[k][t] - vanilla[k][t]
             for k in common if t in vanilla[k] and t in adaptive[k]]
    m, s = mean_se(pairs); diffs.append(m); ds.append(s)

fig, ax = plt.subplots(figsize=(14, 5))
colors = ["#27AE60" if d > 0 else "#E74C3C" for d in diffs]
ax.bar(range(len(TOP)), diffs, yerr=ds, color=colors,
       error_kw=dict(ecolor='black', capsize=3))
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(TOP)))
ax.set_xticklabels([t.replace("_", " ").title() for t in TOP], rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Δ Accuracy (Adaptive − Vanilla, pp)")
ax.set_title(f"Paired Improvement from Adaptive Mechanism (n={n} seeds)", fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("results/figures/ablation_paired_diff.png", dpi=150); plt.close()
print("Saved: ablation_paired_diff.png")

# -----------------------------------------------------------------------------
# Figure 3: key MMLU subjects 3-way
# -----------------------------------------------------------------------------
KEY = ["mmlu_abstract_algebra", "mmlu_college_physics", "mmlu_formal_logic",
       "mmlu_college_chemistry", "mmlu_electrical_engineering"]
KEY = [t for t in KEY if t in base]

bv = [base[t] for t in KEY]
vm = []; vse = []; am = []; ase = []
for t in KEY:
    m, s = mean_se([vanilla[k][t]  for k in common if t in vanilla[k]]);  vm.append(m); vse.append(s)
    m, s = mean_se([adaptive[k][t] for k in common if t in adaptive[k]]); am.append(m); ase.append(s)

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(KEY)); w = 0.27
ax.bar(x - w, bv, w, label="Base",            color="#3274A1")
ax.bar(x,     vm, w, yerr=vse, label="Vanilla SFT", color="#E1812C",
       error_kw=dict(ecolor='black', capsize=3))
ax.bar(x + w, am, w, yerr=ase, label="Adaptive SFT", color="#3A923A",
       error_kw=dict(ecolor='black', capsize=3))
ax.set_xticks(x)
ax.set_xticklabels([t.replace("mmlu_", "").replace("_", " ").title() for t in KEY],
                   rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Recovery of Worst-Forgotten MMLU Subjects (Vanilla → Adaptive)", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("results/figures/ablation_key_subjects.png", dpi=150); plt.close()
print("Saved: ablation_key_subjects.png")

# -----------------------------------------------------------------------------
# Figure 4: probe trajectory during training (uses any one seed)
# -----------------------------------------------------------------------------
traj_files = sorted(glob.glob("results/adaptive/*/probe_trajectory.json"))
if not traj_files:
    traj_files = sorted(glob.glob("qwen-sft-seed*-ep*-adaptive/probe_trajectory.json"))

if traj_files:
    with open(traj_files[0]) as f:
        history = json.load(f)
    
    steps = [h["step"] for h in history]
    domains = list(history[0]["accs"].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(domains)))
    
    fig, ax = plt.subplots(figsize=(13, 6))
    for d, c in zip(domains, colors):
        ys = [h["accs"][d] * 100 for h in history]
        ax.plot(steps, ys, marker='o', label=d, color=c, lw=2)
    
    for h in history:
        for itv in h.get("interventions", []):
            ax.axvline(h["step"], color=colors[domains.index(itv["domain"])],
                       ls="--", alpha=0.4, lw=1)
    
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Probe Accuracy (%)")
    ax.set_title(f"Per-Domain Probe Accuracy During Adaptive Training\n"
                 f"(dashed lines mark interventions; source: {os.path.basename(os.path.dirname(traj_files[0]))})",
                 fontweight="bold")
    ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("results/figures/ablation_probe_trajectory.png", dpi=150); plt.close()
    print("Saved: ablation_probe_trajectory.png")
else:
    print("(no probe trajectory found; skipping trajectory figure)")

print("\nDone.")
```

---

## Files 7 + 8: directory placeholders

Create two empty placeholder files so the directories exist in version control:

- `probes/.gitkeep` — empty file
- `results/adaptive/.gitkeep` — empty file

---

## Execution sequence (after files are on the server)

The user will scp these files to the server, then run them in this order:

```bash
# On server, in ~/forgetting_experiment/, with venv active:

# 1. One-time setup: build probe set and classify instructions
python build_probe_set.py           # ~2 min; creates probes/probe_set.json
python instruction_classifier.py    # ~1 sec; creates instruction_domains.json

# 2. Run all 3 adaptive seeds (~10-14 hours total in tmux)
chmod +x run_adaptive_seeds.sh
bash run_adaptive_seeds.sh 2>&1 | tee logs/all_adaptive.log

# 3. Compare vanilla vs adaptive
python compare_ablation.py | tee logs/compare_ablation.log
python visualize_ablation.py | tee logs/visualize_ablation.log

# 4. Copy ablation figures to local laptop (commands provided separately)
```

---

## Verification checklist for Claude Code

After creating all files, verify:

- [ ] `finetune_adaptive.py` exists and is syntactically valid Python (`python -c "import ast; ast.parse(open('finetune_adaptive.py').read())"`)
- [ ] `build_probe_set.py`, `instruction_classifier.py`, `compare_ablation.py`, `visualize_ablation.py` all parse cleanly
- [ ] `run_adaptive_seeds.sh` has executable bit set
- [ ] `probes/` and `results/adaptive/` directories exist (via .gitkeep)
- [ ] The `format_text` function in `finetune_adaptive.py` matches the prompt format used in the existing `finetune.py` (read `finetune.py` to verify)
- [ ] `MAX_LENGTH` in `finetune_adaptive.py` matches the value used in `finetune.py`
- [ ] `TrainingArguments` in `finetune_adaptive.py` is byte-for-byte identical to `finetune.py` (except for `OUTPUT_DIR` and `callbacks`)

**Critical:** The whole ablation is meaningful only if vanilla and adaptive use identical training configurations except for the adaptive mechanism. Any unintended divergence in prompt format, max_length, or TrainingArguments invalidates the comparison.

---

## Notes on design choices (for the writeup)

These defaults are calibrated for an undergraduate feasibility study with a tight deadline. Each has a known limitation worth mentioning in the writeup:

- **Probe set: 5 domains × 30 questions.** 30 questions per domain reduces binomial sampling noise to ~9pp standard deviation, making a 10pp absolute threshold defensible. Smaller probe sets would be noise-dominated.
- **Threshold: 10pp absolute, confirmed across 2 consecutive evals.** Above the noise floor; two-eval confirmation rules out single-step spikes. A more rigorous design would use noise-calibrated thresholds (k × σ_binomial) — future work.
- **Reweight factor: 1.5×.** Conservative; can re-trigger and compound. 2× risks over-correction; 1.2× may not move the needle.
- **Probe eval every 25 steps.** With ~250 total steps, gives 10 evaluation points. Probe evaluation cost is ~10-15 seconds, negligible.
- **Instruction classification: keyword regex.** Fast and reproducible but imperfect. A more sophisticated approach would use the base model itself as a one-shot classifier — viable future work.
- **No decay of weight boosts.** Once a domain is boosted, it stays boosted. This may bias the final epochs toward earlier-detected forgotten domains. Future work could add weight decay back toward 1.0 over time.

These limitations are honest and defensible. Acknowledging them in the writeup pre-empts methodological pushback from a reviewer or examining committee.
