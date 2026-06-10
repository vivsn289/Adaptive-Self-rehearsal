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
TARGET_INSTRUCTIONS = 2000
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
