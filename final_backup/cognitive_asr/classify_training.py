"""Step 3: LLM-as-a-judge classification of 52K Alpaca training data into cognitive tiers.

Uses Qwen2.5-3B-Instruct as judge. Batched inference with left-padding for speed.
At ~4s/example naive, batching brings this down significantly.

RESUMABLE: If classifications/alpaca_52k_tiers.json already exists, it continues from
the last classified index rather than restarting from scratch.

Output:
  classifications/alpaca_52k_tiers.json   — [{instruction_index, tier, instruction_preview}]
  classifications/validation_sample.json  — 100 stratified examples for manual review
"""

import json
import os
import random
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    JUDGE_MODEL, ALPACA_DATA, CLASSIFICATIONS_DIR,
    ALPACA_TIERS_PATH, VALIDATION_SAMPLE_PATH,
    JUDGE_PROMPT_TEMPLATE,
)

SEED = 42
BATCH_SIZE = 8          # Tune down if OOM; tune up if GPU has headroom
MAX_NEW_TOKENS = 10     # Only need "TIER_1", "TIER_2", or "TIER_3"
SAVE_EVERY = 500        # Checkpoint frequency

random.seed(SEED)
os.makedirs(CLASSIFICATIONS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print(f"Loading training data: {ALPACA_DATA}", flush=True)
with open(ALPACA_DATA) as f:
    alpaca = json.load(f)
print(f"Loaded {len(alpaca)} Alpaca examples", flush=True)

# ---------------------------------------------------------------------------
# Load judge model
# ---------------------------------------------------------------------------

print(f"Loading judge model: {JUDGE_MODEL}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Judge model loaded.", flush=True)


# ---------------------------------------------------------------------------
# Resume: load existing classifications
# ---------------------------------------------------------------------------

existing_classifications = []
if os.path.exists(ALPACA_TIERS_PATH):
    with open(ALPACA_TIERS_PATH) as f:
        existing_classifications = json.load(f)
    print(f"RESUMING: {len(existing_classifications)} already classified", flush=True)

classified_indices = {c["instruction_index"] for c in existing_classifications}
remaining_indices = [i for i in range(len(alpaca)) if i not in classified_indices]
print(f"Remaining to classify: {len(remaining_indices)}", flush=True)


# ---------------------------------------------------------------------------
# Judge inference
# ---------------------------------------------------------------------------

def build_judge_prompt(item):
    instruction = item.get("instruction", "")[:300]
    output = item.get("output", "")[:200]
    return JUDGE_PROMPT_TEMPLATE.format(instruction=instruction, output=output)


def parse_tier(text: str) -> str:
    text = text.strip().upper()
    for tier in ["TIER_1", "TIER_2", "TIER_3"]:
        if tier in text:
            return tier
    # Fallback: check for T1/T2/T3 shorthand
    m = re.search(r"TIER[_ ]?([123])", text)
    if m:
        return f"TIER_{m.group(1)}"
    return "TIER_3"  # Default to least disruptive tier on parse failure


@torch.no_grad()
def classify_batch(items_batch):
    """Classify a batch of items and return tier labels."""
    prompts = [build_judge_prompt(item) for item in items_batch]

    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = []
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            try:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = p
            formatted.append(text)
    else:
        formatted = prompts

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    tiers = []
    prompt_len = inputs["input_ids"].shape[1]
    for seq in out:
        generated = seq[prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        tiers.append(parse_tier(text))
    return tiers


# ---------------------------------------------------------------------------
# Main classification loop
# ---------------------------------------------------------------------------

all_classifications = list(existing_classifications)
start_time = time.time()
n_processed = 0

print(f"\nStarting classification... batch_size={BATCH_SIZE}", flush=True)

for batch_start in range(0, len(remaining_indices), BATCH_SIZE):
    batch_idx = remaining_indices[batch_start: batch_start + BATCH_SIZE]
    batch_items = [alpaca[i] for i in batch_idx]

    try:
        tiers = classify_batch(batch_items)
    except Exception as e:
        print(f"  [ERROR] batch {batch_start}: {e}", flush=True)
        tiers = ["TIER_3"] * len(batch_items)

    for i, (alpaca_idx, tier) in enumerate(zip(batch_idx, tiers)):
        all_classifications.append({
            "instruction_index": alpaca_idx,
            "tier": tier,
            "instruction_preview": alpaca[alpaca_idx].get("instruction", "")[:100],
        })

    n_processed += len(batch_idx)

    if n_processed % SAVE_EVERY < BATCH_SIZE or batch_start + BATCH_SIZE >= len(remaining_indices):
        with open(ALPACA_TIERS_PATH, "w") as f:
            json.dump(all_classifications, f)
        elapsed = time.time() - start_time
        rate = n_processed / elapsed if elapsed > 0 else 1
        remaining = len(remaining_indices) - n_processed
        eta_h = remaining / rate / 3600
        print(
            f"  [{n_processed}/{len(remaining_indices)}] "
            f"total classified: {len(all_classifications)} | "
            f"rate: {rate:.1f}/s | ETA: {eta_h:.1f}h",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Distribution summary
# ---------------------------------------------------------------------------

tier_counts = {"TIER_1": 0, "TIER_2": 0, "TIER_3": 0}
for c in all_classifications:
    tier_counts[c.get("tier", "TIER_3")] = tier_counts.get(c.get("tier", "TIER_3"), 0) + 1

total = len(all_classifications)
print(f"\n{'='*60}\nClassification Distribution ({total} total)\n{'='*60}")
for tier, count in sorted(tier_counts.items()):
    pct = count / total * 100 if total > 0 else 0
    print(f"  {tier}: {count:>6} ({pct:.1f}%)")
print(f"{'='*60}")

# Expected: ~5-6% T1, ~15-25% T2, ~70-80% T3
# Warn if wildly off
t1_pct = tier_counts.get("TIER_1", 0) / total * 100 if total > 0 else 0
if t1_pct < 2 or t1_pct > 15:
    print(f"WARNING: TIER_1 at {t1_pct:.1f}% — expected 5-6%. Review judge prompt.", flush=True)


# ---------------------------------------------------------------------------
# Stratified validation sample
# ---------------------------------------------------------------------------

by_tier = {"TIER_1": [], "TIER_2": [], "TIER_3": []}
for c in all_classifications:
    t = c.get("tier", "TIER_3")
    if t in by_tier:
        by_tier[t].append(c)

sample = []
rng = random.Random(SEED)
for tier, items in by_tier.items():
    n = min(33, len(items))
    sampled = rng.sample(items, n)
    for s in sampled:
        idx = s["instruction_index"]
        row = dict(s)
        row["instruction_full"] = alpaca[idx].get("instruction", "")
        row["output_preview"] = alpaca[idx].get("output", "")[:200]
        sample.append(row)

rng.shuffle(sample)
sample = sample[:100]

with open(VALIDATION_SAMPLE_PATH, "w") as f:
    json.dump(sample, f, indent=2)

print(f"\nValidation sample (100 examples) saved → {VALIDATION_SAMPLE_PATH}")
print("Review this manually to verify tier quality before training.\n")

print("=== Stratified validation sample (first 20) ===")
for i, s in enumerate(sample[:20]):
    print(f"\n[{i+1}] {s['tier']} | {s['instruction_preview']}")

print(f"\nClassification complete. Results: {ALPACA_TIERS_PATH}")
print(f"Next step: python validate_rehearsal.py")
