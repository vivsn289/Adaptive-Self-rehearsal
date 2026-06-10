# CLAUDE.md — Self-Instruct Forgetting Experiment

## Instructions for Claude Code

You are running on the USER'S LOCAL MACHINE (WSL on Windows). You CANNOT access the
remote GPU server. Your only job is to **create all the script files** listed below in
the local project folder. The user will then copy them to the server via scp and run
them there manually.

When the user says "create all scripts", generate every file in the FILES section below,
exactly as specified. Do not run them. Do not try to SSH anywhere. Just write the files.

After creating the files, tell the user the scp command and the server run order.

---

## Project Context

- **Goal:** Replicate Self-Instruct (Wang et al. 2023) on Mistral-7B and measure catastrophic forgetting.
- **Base model:** mistralai/Mistral-7B-v0.1
- **Pipeline:** Model generates 10K instructions via iterative bootstrap → full SFT on that data → benchmark before/after.
- **Server:** llm@192.168.51.62 (IIT Hyderabad CSE, shared account, RTX 3090 24GB, Ubuntu 22.04)
- **Critical constraint:** Shared server. Processes must survive disconnects (tmux + nohup) and resume after interruption (checkpointing).

---

## Robustness Requirements (why the scripts are built this way)

The server is shared and the user may be disconnected or interrupted. Every long-running
script is designed to be RESUMABLE:

- `bootstrap.py` saves generated instructions every 100 accepted samples and RESUMES from
  the existing file on restart. If killed at 6,200 instructions, restarting continues from 6,200.
- `finetune.py` saves checkpoints every 200 steps and auto-resumes from the latest checkpoint.
- Benchmarks write per-task results; if interrupted, completed tasks are preserved and only
  the incomplete run needs rerunning.
- All long commands are run under tmux AND logged to files, so progress is visible even after disconnect.

---

# FILES TO CREATE

Create each of these files in the project folder exactly as written.

---

## FILE 1: `setup.sh`

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Self-Instruct Forgetting Experiment Setup"
echo "=========================================="

# Create virtual environment
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env
fi
source env/bin/activate

# Install dependencies
echo "Installing dependencies (this may take several minutes)..."
pip install --upgrade pip --quiet
pip install torch transformers accelerate datasets trl peft lm-eval rouge-score matplotlib pandas tabulate bitsandbytes sentencepiece protobuf --quiet

# Clone Self-Instruct repo for seed tasks
if [ ! -d "self-instruct" ]; then
    echo "Cloning Self-Instruct repo for seed tasks..."
    git clone https://github.com/yizhongw/self-instruct.git
fi

# Create directory structure
mkdir -p results/base results/instruct results/figures generated_data logs

# Verify GPU
echo ""
echo "=========================================="
echo "GPU Check:"
python -c "import torch; print('  CUDA available:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('  VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB') if torch.cuda.is_available() else None"
echo "=========================================="
echo ""
echo "Setup complete. Next: bash check_gpu.sh to verify the GPU is free before starting."
```

---

## FILE 2: `check_gpu.sh`

```bash
#!/bin/bash
# Run this BEFORE starting any job on the shared server.
echo "=========================================="
echo "Current GPU usage (make sure it's free):"
echo "=========================================="
nvidia-smi
echo ""
echo "=========================================="
echo "Users currently logged in:"
echo "=========================================="
who
echo ""
echo "If the GPU shows significant memory in use by another process,"
echo "DO NOT start your job. Wait or coordinate with the other user."
```

---

## FILE 3: `benchmark_base.sh`

```bash
#!/bin/bash
set -e
source env/bin/activate

MODEL="mistralai/Mistral-7B-v0.1"
OUT="results/base"

echo "Benchmarking BASE model: $MODEL"
echo "Logging to logs/benchmark_base.log"

# Fast multiple-choice benchmarks
lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=float16 \
    --tasks arc_challenge,arc_easy,hellaswag,boolq,mmlu,winogrande,piqa,openbookqa,copa,rte,lambada_openai \
    --batch_size 4 \
    --output_path $OUT 2>&1 | tee logs/benchmark_base_mc.log

# Generation-heavy benchmarks (slower, smaller batch)
lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=float16 \
    --tasks gsm8k,triviaqa \
    --batch_size 1 \
    --output_path $OUT 2>&1 | tee logs/benchmark_base_gen.log

echo "BASE benchmarks complete. Results in $OUT"
```

---

## FILE 4: `bootstrap.py`

```python
"""
Self-Instruct Bootstrap Pipeline for Mistral-7B.
Generates 10K instruction-following samples from the model itself.

RESUMABLE: On restart, loads existing generated_data/self_instruct_generated.json
and continues from where it left off. Safe to kill and restart anytime.
"""

import json
import random
import re
import os
import torch
import time
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# CONFIG
# ============================================================
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
SEED_TASKS_PATH = "self-instruct/data/seed_tasks.jsonl"
OUTPUT_DIR = "generated_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "self_instruct_generated.json")
STATS_FILE = os.path.join(OUTPUT_DIR, "generation_stats.json")
TARGET_INSTRUCTIONS = 10000
ROUGE_THRESHOLD = 0.7
MAX_GENERATION_ATTEMPTS = TARGET_INSTRUCTIONS * 10
SAVE_EVERY = 100  # Save checkpoint every N accepted instructions

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================
print(f"Loading {BASE_MODEL}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto",
)
model.eval()
print("Model loaded.", flush=True)

def generate(prompt, max_new_tokens=256, temperature=0.7, top_p=0.5, stop_sequences=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, do_sample=temperature > 0, pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    if stop_sequences:
        for stop in stop_sequences:
            if stop in text:
                text = text[:text.index(stop)]
    return text.strip()

# ============================================================
# LOAD SEED TASKS
# ============================================================
print("Loading seed tasks...", flush=True)
seed_tasks = []
with open(SEED_TASKS_PATH, "r") as f:
    for line in f:
        task = json.loads(line)
        instances = task.get("instances", [{}])
        seed_tasks.append({
            "instruction": task["instruction"],
            "input": instances[0].get("input", "") if instances else "",
            "output": instances[0].get("output", "") if instances else "",
            "is_classification": task.get("is_classification", False),
            "source": "seed",
        })

# ============================================================
# RESUME: Load existing generated data if present
# ============================================================
all_generated_data = []
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        existing = json.load(f)
    for item in existing:
        all_generated_data.append({
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item.get("output", ""),
            "is_classification": item.get("is_classification", False),
            "source": "generated",
        })
    print(f"RESUMING: Loaded {len(all_generated_data)} previously generated instructions.", flush=True)

task_pool = list(seed_tasks) + list(all_generated_data)
all_instructions = [t["instruction"] for t in task_pool]
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
print(f"Pool size: {len(task_pool)} ({len(seed_tasks)} seed + {len(all_generated_data)} generated)", flush=True)

# ============================================================
# PIPELINE FUNCTIONS
# ============================================================
def generate_new_instructions(task_pool):
    seed_pool = [t for t in task_pool if t["source"] == "seed"]
    gen_pool = [t for t in task_pool if t["source"] == "generated"]
    if len(gen_pool) >= 2:
        sampled = random.sample(seed_pool, 6) + random.sample(gen_pool, 2)
    else:
        sampled = random.sample(seed_pool, min(8, len(seed_pool)))
    random.shuffle(sampled)

    prompt = "Come up with a series of tasks:\n\n"
    for i, task in enumerate(sampled, 1):
        prompt += f"Task {i}: {task['instruction']}\n"
    prompt += f"Task {len(sampled) + 1}:"

    response = generate(prompt, max_new_tokens=512, temperature=0.7, top_p=0.5,
                        stop_sequences=["Task 16", "\n\n\n"])
    new_instructions = []
    lines = (f"Task {len(sampled) + 1}:" + response).split("\n")
    for line in lines:
        match = re.match(r"^Task\s*\d+\s*[:.]?\s*(.+)", line.strip())
        if match:
            inst = match.group(1).strip()
            if len(inst) > 5:
                new_instructions.append(inst)
    return new_instructions

def classify_instruction(instruction):
    prompt = (
        "Can the following task be regarded as a classification task with finite output labels?\n\n"
        "Task: Given my personality and the job, tell me if I would be suitable.\n"
        "Is it classification? Yes\n\n"
        "Task: Give me an example of a time when you had to use your sense of humor.\n"
        "Is it classification? No\n\n"
        "Task: Fact checking - tell me if the statement is true, false, or unknown.\n"
        "Is it classification? Yes\n\n"
        "Task: Given the name of an exercise, explain how to do it.\n"
        "Is it classification? No\n\n"
        f"Task: {instruction}\n"
        "Is it classification?"
    )
    response = generate(prompt, max_new_tokens=5, temperature=0.0, top_p=1.0)
    return "yes" in response.lower()[:10]

def generate_instance_input_first(instruction):
    prompt = (
        "Come up with examples for the following tasks. Try to generate multiple examples when possible.\n"
        "If the task doesn't require additional input, you can generate the output directly.\n\n"
        "Task: Which exercises are best for reducing belly fat at home?\n"
        "Output:\n- Lying Leg Raises\n- Plank\n- Sit-ups\n\n"
        "Task: Sort the given list ascendingly.\n"
        "Input: [10, 92, 2, 5, -4]\nOutput: [-4, 2, 5, 10, 92]\n\n"
        f"Task: {instruction}\n"
    )
    response = generate(prompt, max_new_tokens=300, temperature=0.0, top_p=1.0,
                        stop_sequences=["Task:", "\n\n\n"])
    input_text, output_text = "", ""
    if "Output:" in response:
        parts = response.split("Output:", 1)
        before = parts[0]
        output_text = parts[1].strip()
        if "Input:" in before:
            input_text = before.split("Input:", 1)[1].strip()
        input_text = re.sub(r"^Example\s*\d+\s*\n?", "", input_text).strip()
    else:
        output_text = response.strip()
    return input_text, output_text

def generate_instance_output_first(instruction):
    prompt = (
        "Given the classification task definition and the class labels, "
        "generate an input that corresponds to each of the class labels.\n\n"
        "Task: Classify the sentiment into positive, negative, or mixed.\n"
        "Class label: mixed\n"
        "Sentence: I enjoy the food but the service is slow.\n"
        "Class label: Positive\n"
        "Sentence: I had a great day today.\n\n"
        f"Task: {instruction}\n"
        "Class label:"
    )
    response = generate(prompt, max_new_tokens=200, temperature=0.0, top_p=1.0,
                        stop_sequences=["Task:", "\n\n\n"])
    output_text, input_text = "", ""
    lines = response.strip().split("\n")
    if lines:
        output_text = lines[0].strip()
        rest = "\n".join(lines[1:]).strip()
        for prefix in ["Sentence:", "Input:", "Text:", "Email:", "Document:"]:
            if rest.startswith(prefix):
                rest = rest[len(prefix):].strip()
                break
        input_text = rest
    return input_text, output_text

def is_valid_instruction(instruction, existing):
    if len(instruction.split()) < 3 or len(instruction.split()) > 150:
        return False
    blacklist = ["image", "picture", "graph", "plot", "draw", "figure", "photo",
                 "video", "audio", "recording", "listen", "watch", "diagram"]
    if any(w in instruction.lower() for w in blacklist):
        return False
    check = existing
    if len(existing) > 500:
        check = random.sample(existing, 500) + existing[-100:]
    for ex in check:
        if scorer.score(ex, instruction)["rougeL"].fmeasure > ROUGE_THRESHOLD:
            return False
    return True

def is_valid_instance(input_text, output_text):
    if len(output_text.strip()) < 2:
        return False
    if output_text.strip() == input_text.strip() and input_text.strip():
        return False
    if len(output_text.split()) > 500:
        return False
    return True

def save_checkpoint():
    save_data = [{"instruction": t["instruction"], "input": t["input"],
                  "output": t["output"], "is_classification": t["is_classification"]}
                 for t in all_generated_data]
    with open(OUTPUT_FILE, "w") as f:
        json.dump(save_data, f, indent=2)

# ============================================================
# MAIN LOOP
# ============================================================
print(f"\n{'='*60}\nTarget: {TARGET_INSTRUCTIONS} | Current: {len(all_generated_data)}\n{'='*60}\n", flush=True)

attempt = 0
start_time = time.time()
start_count = len(all_generated_data)

while len(all_generated_data) < TARGET_INSTRUCTIONS and attempt < MAX_GENERATION_ATTEMPTS:
    attempt += 1
    if attempt % 50 == 0:
        elapsed = (time.time() - start_time) / 3600
        new_count = len(all_generated_data) - start_count
        rate = new_count / elapsed if elapsed > 0 else 0
        remaining = TARGET_INSTRUCTIONS - len(all_generated_data)
        eta = remaining / rate if rate > 0 else float('inf')
        print(f"[Attempt {attempt}] Generated: {len(all_generated_data)}/{TARGET_INSTRUCTIONS} | "
              f"Rate: {rate:.0f}/hr | ETA: {eta:.1f}hr", flush=True)

    try:
        new_instructions = generate_new_instructions(task_pool)
    except Exception as e:
        print(f"  Gen error: {e}", flush=True)
        continue

    for instruction in new_instructions:
        if len(all_generated_data) >= TARGET_INSTRUCTIONS:
            break
        if not is_valid_instruction(instruction, all_instructions):
            continue
        try:
            is_clf = classify_instruction(instruction)
        except Exception:
            is_clf = False
        try:
            if is_clf:
                inp, out = generate_instance_output_first(instruction)
            else:
                inp, out = generate_instance_input_first(instruction)
        except Exception:
            continue
        if not is_valid_instance(inp, out):
            continue

        new_task = {"instruction": instruction, "input": inp, "output": out,
                    "is_classification": is_clf, "source": "generated"}
        task_pool.append(new_task)
        all_instructions.append(instruction)
        all_generated_data.append(new_task)

        if len(all_generated_data) % 100 == 0:
            elapsed = (time.time() - start_time) / 3600
            print(f"  >>> {len(all_generated_data)} instructions ({elapsed:.1f}hr)", flush=True)
        if len(all_generated_data) % SAVE_EVERY == 0:
            save_checkpoint()
            print(f"  [Checkpoint saved: {len(all_generated_data)}]", flush=True)

# Final save
save_checkpoint()
elapsed = (time.time() - start_time) / 3600
clf_count = sum(1 for t in all_generated_data if t["is_classification"])
stats = {
    "total_generated": len(all_generated_data),
    "classification_tasks": clf_count,
    "non_classification_tasks": len(all_generated_data) - clf_count,
    "total_attempts": attempt,
    "session_time_hours": round(elapsed, 1),
}
with open(STATS_FILE, "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n{'='*60}\nDONE. {len(all_generated_data)} instructions. {elapsed:.1f}hr this session.\n{'='*60}", flush=True)
print(json.dumps(stats, indent=2), flush=True)
```

---

## FILE 5: `finetune.py`

```python
"""
Full SFT (not LoRA) of Mistral-7B on self-generated instruction data.
Optimized for RTX 3090 24GB.

RESUMABLE: Auto-resumes from the latest checkpoint in the output dir if present.
"""
import os
import json
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
                          Trainer, DataCollatorForLanguageModeling)

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
DATA_PATH = "generated_data/self_instruct_generated.json"
OUTPUT_DIR = "./mistral-self-instruct-sft"

# Load data
with open(DATA_PATH) as f:
    data = json.load(f)
print(f"Training on {len(data)} self-generated samples", flush=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

def format_and_tokenize(sample):
    if sample.get("input", "").strip():
        text = (f"Below is an instruction that describes a task, paired with an input. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{sample['instruction']}\n\n"
                f"### Input:\n{sample['input']}\n\n"
                f"### Response:\n{sample['output']}{tokenizer.eos_token}")
    else:
        text = (f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{sample['instruction']}\n\n"
                f"### Response:\n{sample['output']}{tokenizer.eos_token}")
    tokens = tokenizer(text, truncation=True, max_length=512, padding=False)
    return tokens

dataset = Dataset.from_list(data)
tokenized = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

# Load model with gradient checkpointing
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto",
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=True,
    logging_steps=25,
    save_steps=200,            # Frequent checkpoints for resumability
    save_total_limit=3,
    report_to="none",
    optim="adamw_torch",
    gradient_checkpointing=True,
    dataloader_pin_memory=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model, args=training_args, train_dataset=tokenized,
    data_collator=data_collator, tokenizer=tokenizer,
)

# Auto-resume from checkpoint if one exists
resume = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        resume = True
        print(f"RESUMING from existing checkpoint in {OUTPUT_DIR}", flush=True)

print("Starting full SFT (gradient checkpointing ON, effective batch 32, 2 epochs)...", flush=True)
trainer.train(resume_from_checkpoint=resume)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Model saved to {OUTPUT_DIR}", flush=True)
```

---

## FILE 6: `benchmark_instruct.sh`

```bash
#!/bin/bash
set -e
source env/bin/activate

MODEL="./mistral-self-instruct-sft"
OUT="results/instruct"

echo "Benchmarking FINE-TUNED model: $MODEL"

lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=float16 \
    --tasks arc_challenge,arc_easy,hellaswag,boolq,mmlu,winogrande,piqa,openbookqa,copa,rte,lambada_openai \
    --batch_size 4 \
    --output_path $OUT 2>&1 | tee logs/benchmark_instruct_mc.log

lm_eval --model hf \
    --model_args pretrained=$MODEL,dtype=float16 \
    --tasks gsm8k,triviaqa \
    --batch_size 1 \
    --output_path $OUT 2>&1 | tee logs/benchmark_instruct_gen.log

echo "FINE-TUNED benchmarks complete. Results in $OUT"
```

---

## FILE 7: `compare.py`

```python
import json, glob, os

def load_scores(results_dir):
    scores = {}
    for jf in glob.glob(os.path.join(results_dir, "**", "results.json"), recursive=True):
        with open(jf) as f:
            data = json.load(f)
        if "results" in data:
            for task, metrics in data["results"].items():
                for key in ["acc_norm,none", "acc,none", "exact_match,none"]:
                    if key in metrics:
                        scores[task] = round(metrics[key] * 100, 2)
                        break
    return scores

base = load_scores("results/base")
inst = load_scores("results/instruct")

print(f"\n{'Task':<28} {'Base':>8} {'Instruct':>10} {'Delta':>8} {'Forgetting':>12}")
print("-" * 70)
for task in sorted(base.keys()):
    if task in inst and not task.startswith("mmlu_"):
        b, i = base[task], inst[task]
        delta = round(i - b, 2)
        fgt = round((b - i) / b * 100, 2) if b > 0 else 0
        flag = " <<<" if fgt > 5 else (" ++" if fgt < -5 else "")
        print(f"{task:<28} {b:>8} {i:>10} {delta:>+8} {fgt:>10}%{flag}")
print("\n<<< = significant forgetting (>5%)   ++ = significant improvement (>5%)")
```

---

## FILE 8: `visualize.py`

```python
import json, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_scores(results_dir):
    scores = {}
    for jf in glob.glob(os.path.join(results_dir, "**", "results.json"), recursive=True):
        with open(jf) as f:
            data = json.load(f)
        if "results" in data:
            for task, metrics in data["results"].items():
                for key in ["acc_norm,none", "acc,none", "exact_match,none"]:
                    if key in metrics:
                        scores[task] = round(metrics[key] * 100, 2)
                        break
    return scores

base = load_scores("results/base")
inst = load_scores("results/instruct")

CATEGORIES = {
    "Factual Recall": ["triviaqa", "mmlu"],
    "Reasoning": ["arc_challenge", "arc_easy", "openbookqa", "copa"],
    "Commonsense": ["hellaswag", "winogrande", "piqa"],
    "Comprehension": ["boolq", "rte"],
    "Language Modeling": ["lambada_openai"],
    "Math": ["gsm8k"],
}

top_tasks = [t for t in sorted(base.keys()) if t in inst and not t.startswith("mmlu_")]
mmlu_tasks = [t for t in sorted(base.keys()) if t in inst and t.startswith("mmlu_") and t != "mmlu"]
os.makedirs("results/figures", exist_ok=True)

def disp(t):
    return t.replace("lambada_openai", "LAMBADA").replace("_", " ").title()

# FIG 1: Comparison
tasks = top_tasks
labels = [disp(t) for t in tasks]
bv = [base[t] for t in tasks]; iv = [inst[t] for t in tasks]
fig, ax = plt.subplots(figsize=(16, 6))
x = range(len(tasks)); w = 0.35
b1 = ax.bar([i-w/2 for i in x], bv, w, label="Base (Mistral-7B)", color="#3274A1")
b2 = ax.bar([i+w/2 for i in x], iv, w, label="Self-Instruct SFT", color="#E1812C")
ax.set_ylabel("Accuracy (%)"); ax.set_title("Base vs Self-Instruct Fine-Tuned: Per-Benchmark Performance", fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
ax.legend(); ax.grid(axis="y", alpha=0.3)
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout(); plt.savefig("results/figures/comparison.png", dpi=150); plt.close()
print("Saved: comparison.png")

# FIG 2: Forgetting scores
fgt = [(base[t]-inst[t])/base[t]*100 if base[t]>0 else 0 for t in tasks]
fig, ax = plt.subplots(figsize=(11, 6))
colors = ["#E74C3C" if f>5 else ("#27AE60" if f<-5 else "#95A5A6") for f in fgt]
bars = ax.barh(labels, fgt, color=colors, edgecolor="white")
ax.set_xlabel("Forgetting Score (%) — Positive = Lost | Negative = Gained")
ax.set_title("Per-Domain Forgetting After Self-Instruct Fine-Tuning", fontweight="bold")
ax.axvline(0, color="black", lw=0.8); ax.axvline(5, color="#E74C3C", lw=0.8, ls="--", alpha=0.4); ax.axvline(-5, color="#27AE60", lw=0.8, ls="--", alpha=0.4)
for bar, v in zip(bars, fgt):
    xp = bar.get_width()+0.3 if bar.get_width()>=0 else bar.get_width()-0.3
    ax.text(xp, bar.get_y()+bar.get_height()/2, f"{v:.1f}%", ha="left" if v>=0 else "right", va="center", fontsize=9)
ax.legend(handles=[mpatches.Patch(color="#E74C3C", label="Forgetting (>5%)"),
                   mpatches.Patch(color="#95A5A6", label="Stable (±5%)"),
                   mpatches.Patch(color="#27AE60", label="Improvement (>5%)")], loc="lower right")
plt.tight_layout(); plt.savefig("results/figures/forgetting.png", dpi=150); plt.close()
print("Saved: forgetting.png")

# FIG 3: Delta waterfall
deltas = [inst[t]-base[t] for t in tasks]
fig, ax = plt.subplots(figsize=(14, 5))
colors = ["#27AE60" if d>=0 else "#E74C3C" for d in deltas]
ax.bar(labels, deltas, color=colors, edgecolor="white")
ax.set_ylabel("Accuracy Change (pp)"); ax.set_title("Impact of Self-Instruct Fine-Tuning: Per-Benchmark Change", fontweight="bold")
ax.axhline(0, color="black", lw=0.8); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
for i, d in enumerate(deltas):
    ax.text(i, d+(0.3 if d>=0 else -0.3), f"{d:+.1f}", ha="center", va="bottom" if d>=0 else "top", fontsize=8, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("results/figures/delta_waterfall.png", dpi=150); plt.close()
print("Saved: delta_waterfall.png")

# FIG 4: MMLU per-subject
if mmlu_tasks:
    mf = [(t, (base[t]-inst[t])/base[t]*100 if base[t]>0 else 0) for t in mmlu_tasks]
    mf.sort(key=lambda x: x[1], reverse=True)
    sel = mf[:15] + mf[-15:]
    lm = [t[0].replace("mmlu_","").replace("_"," ").title() for t in sel]
    vm = [t[1] for t in sel]
    cm = ["#E74C3C" if v>5 else ("#27AE60" if v<-5 else "#95A5A6") for v in vm]
    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(lm, vm, color=cm, edgecolor="white")
    ax.set_xlabel("Forgetting Score (%)"); ax.set_title("MMLU Per-Subject: Top 15 Forgotten + Top 15 Improved", fontweight="bold")
    ax.axvline(0, color="black", lw=0.8)
    for bar, v in zip(bars, vm):
        xp = bar.get_width()+0.5 if bar.get_width()>=0 else bar.get_width()-0.5
        ax.text(xp, bar.get_y()+bar.get_height()/2, f"{v:.1f}%", ha="left" if v>=0 else "right", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig("results/figures/mmlu_subject_forgetting.png", dpi=150); plt.close()
    print("Saved: mmlu_subject_forgetting.png")
    print(f"MMLU spread: {mf[0][1]-mf[-1][1]:.1f}pp | Most forgotten: {mf[0][0]} ({mf[0][1]:+.1f}%) | Most improved: {mf[-1][0]} ({mf[-1][1]:+.1f}%)")

# FIG 5: Category-level (pitch chart)
cat_scores = {}
for cn, ct in CATEGORIES.items():
    vals = [(base[t]-inst[t])/base[t]*100 for t in ct if t in base and t in inst and base[t]>0]
    if vals:
        cat_scores[cn] = round(sum(vals)/len(vals), 2)
if cat_scores:
    cn = list(cat_scores.keys()); cv = list(cat_scores.values())
    cc = ["#E74C3C" if v>5 else ("#27AE60" if v<-5 else "#95A5A6") for v in cv]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cn, cv, color=cc, edgecolor="white", height=0.6)
    ax.set_xlabel("Average Forgetting Score (%) — Positive = Lost | Negative = Gained")
    ax.set_title("Forgetting by Capability Category", fontweight="bold")
    ax.axvline(0, color="black", lw=0.8)
    for bar, v in zip(bars, cv):
        xp = bar.get_width()+0.3 if bar.get_width()>=0 else bar.get_width()-0.3
        ax.text(xp, bar.get_y()+bar.get_height()/2, f"{v:+.1f}%", ha="left" if v>=0 else "right", va="center", fontsize=10, fontweight="bold")
    plt.tight_layout(); plt.savefig("results/figures/category_forgetting.png", dpi=150); plt.close()
    print("Saved: category_forgetting.png")
    print("\nCategory forgetting:")
    for n, v in sorted(cat_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {n:<20} {v:+.2f}%")

# VERDICT
print(f"\n{'='*60}\nVERDICT\n{'='*60}")
if cat_scores:
    spread = max(cat_scores.values()) - min(cat_scores.values())
    print(f"Category spread: {spread:.1f}pp")
    if spread > 10:
        print("STRONG domain-specific forgetting. Adaptive self-rehearsal well-motivated.")
    elif spread > 5:
        print("MODERATE domain-specific forgetting. Reasonable motivation.")
    else:
        print("Uniform forgetting. Simple data mixing may suffice.")
print(f"{'='*60}")
```

---

# SERVER RUN ORDER (give this to the user after creating files)

After Claude Code creates all 8 files locally, the user runs:

```bash
# === ON LOCAL MACHINE ===
# Copy the whole folder to the server (one command, takes seconds)
scp -r ~/forgetting_experiment llm@192.168.51.62:/home/llm/

# SSH into the server
ssh llm@192.168.51.62

# === ON SERVER ===
cd forgetting_experiment

# IMPORTANT: check the GPU is free before starting (shared server!)
bash check_gpu.sh
# If another process is using significant VRAM, STOP and coordinate.

# Start a named tmux session (survives disconnects)
tmux new -s vivaswan_exp

# One-time setup
bash setup.sh

# Run each step. nohup + & means it keeps running even if tmux dies.
# Step A: Benchmark base (2-3 hr)
bash benchmark_base.sh

# Step B: Bootstrap 10K instructions (8-15 hr, RESUMABLE)
#   If killed, just run again — it resumes from the last checkpoint.
python bootstrap.py

# Step C: Full SFT (3-5 hr, RESUMABLE)
python finetune.py

# Step D: Benchmark fine-tuned (2-3 hr)
bash benchmark_instruct.sh

# Step E: Analysis (seconds)
python compare.py
python visualize.py

# Detach tmux anytime: Ctrl+B then D
# Reattach later: tmux attach -t vivaswan_exp
# Log out safely (processes in tmux keep running): exit

# === BACK ON LOCAL MACHINE (after everything finishes) ===
# Copy results back to look at the charts
scp -r llm@192.168.51.62:/home/llm/forgetting_experiment/results ~/forgetting-results
```

## Resumability Cheat-Sheet (if interrupted)

| If interrupted during... | What to do |
|---|---|
| Base benchmark | Rerun `bash benchmark_base.sh` (lm-eval re-runs cleanly) |
| Bootstrap | Rerun `python bootstrap.py` — resumes from last checkpoint automatically |
| Fine-tuning | Rerun `python finetune.py` — resumes from last checkpoint automatically |
| Instruct benchmark | Rerun `bash benchmark_instruct.sh` |
| Analysis | Rerun anytime, no state |

## Extra Safety: run jobs detached from the terminal entirely

To be doubly safe against disconnects (beyond tmux), prefix long jobs with nohup:

```bash
nohup python bootstrap.py > logs/bootstrap.log 2>&1 &
# Check progress: tail -f logs/bootstrap.log
# This keeps running even if tmux AND ssh both die.
```
