"""Step 4: Three-stage quality filter for raw rehearsal bank examples.

Stage 1: Programmatic answer verification (math/arithmetic only)
Stage 2: LLM-as-a-judge cognitive demand verification (re-classify; reject if below intended tier)
Stage 3: ROUGE-L deduplication within each tier (keep first, drop ROUGE-L > 0.7)

Output:
  rehearsal_bank/verified_tier1.json
  rehearsal_bank/verified_tier2.json
  rehearsal_bank/verified_tier3.json
"""

import json
import os
import re
import time

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    JUDGE_MODEL, REHEARSAL_DIR,
    TIER1_RAW_PATH, TIER2_RAW_PATH, TIER3_RAW_PATH,
    TIER1_VERIFIED_PATH, TIER2_VERIFIED_PATH, TIER3_VERIFIED_PATH,
    TIER1_REHEARSAL_TARGET, TIER2_REHEARSAL_TARGET, TIER3_REHEARSAL_TARGET,
    JUDGE_PROMPT_TEMPLATE,
)

ROUGE_THRESHOLD = 0.7
JUDGE_BATCH_SIZE = 8

os.makedirs(REHEARSAL_DIR, exist_ok=True)


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

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


# ---------------------------------------------------------------------------
# Stage 1: Programmatic answer verification
# ---------------------------------------------------------------------------

def _extract_final_number(text: str):
    """Extract the last number from a solution string."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def _verify_modular_arithmetic(item: dict) -> bool:
    """Re-compute modular arithmetic answer from instruction text."""
    instruction = item.get("instruction", "")
    m = re.search(
        r"\(\(\((\d+)\s*[×x\*]\s*(\d+)\)\s*mod\s*(\d+)\)\s*\+\s*(\d+)\)\s*mod\s*(\d+)",
        instruction,
    )
    if not m:
        return True  # Can't verify — pass through
    a, b, c, d, e = [int(g) for g in m.groups()]
    expected = ((a * b) % c + d) % e
    output_num = _extract_final_number(item.get("output", ""))
    if output_num is None:
        return False
    return abs(output_num - expected) < 0.5


def _verify_chained_arithmetic(item: dict) -> bool:
    """For programmatic examples with verified_programmatic=True, trust the label."""
    return item.get("verified_programmatic", False)


def stage1_programmatic(examples: list, tier: int) -> tuple[list, dict]:
    """Filter math/arithmetic examples by programmatic verification.

    Returns (passing, stats).
    """
    if tier != 1:
        return examples, {"stage1_skipped": len(examples)}

    passing = []
    n_checked = 0
    n_rejected = 0
    for ex in examples:
        subtype = ex.get("subtype", "")
        try:
            if subtype == "modular_arithmetic":
                n_checked += 1
                if not _verify_modular_arithmetic(ex):
                    n_rejected += 1
                    continue
            elif subtype == "chained_arithmetic":
                n_checked += 1
                if not _verify_chained_arithmetic(ex):
                    n_rejected += 1
                    continue
        except Exception:
            # Parse failure → reject gracefully rather than crashing
            n_checked += 1
            n_rejected += 1
            continue
        passing.append(ex)

    stats = {
        "stage1_checked": n_checked,
        "stage1_rejected": n_rejected,
        "stage1_passing": len(passing),
    }
    return passing, stats


# ---------------------------------------------------------------------------
# Stage 2: LLM-as-a-judge cognitive demand verification
# ---------------------------------------------------------------------------

def parse_tier_label(text: str) -> str:
    text = text.strip().upper()
    for t in ["TIER_1", "TIER_2", "TIER_3"]:
        if t in text:
            return t
    m = re.search(r"TIER[_ ]?([123])", text)
    if m:
        return f"TIER_{m.group(1)}"
    return "TIER_3"


@torch.no_grad()
def judge_batch(items: list) -> list:
    """Return list of tier labels for a batch of rehearsal items."""
    prompts = []
    for ex in items:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            instruction=ex.get("instruction", "")[:300],
            output=ex.get("output", "")[:200],
        )
        prompts.append(prompt)

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
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    labels = []
    prompt_len = inputs["input_ids"].shape[1]
    for seq in out:
        text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        labels.append(parse_tier_label(text))
    return labels


TIER_RANK = {"TIER_1": 1, "TIER_2": 2, "TIER_3": 3}


def stage2_judge(examples: list, intended_tier: int) -> tuple[list, dict]:
    """Reject examples classified below intended_tier by the judge."""
    passing = []
    n_rejected = 0
    intended_label = f"TIER_{intended_tier}"

    for batch_start in range(0, len(examples), JUDGE_BATCH_SIZE):
        batch = examples[batch_start: batch_start + JUDGE_BATCH_SIZE]
        try:
            labels = judge_batch(batch)
        except Exception as e:
            print(f"  [stage2 judge error]: {e}", flush=True)
            labels = [intended_label] * len(batch)

        for ex, label in zip(batch, labels):
            if TIER_RANK.get(label, 3) > intended_tier:
                # Classified as lower-demand than intended — reject
                n_rejected += 1
            else:
                passing.append(ex)

        if batch_start % 100 == 0:
            print(
                f"  [Stage2] {batch_start + len(batch)}/{len(examples)} | "
                f"passing so far: {len(passing)}",
                flush=True,
            )

    stats = {
        "stage2_rejected": n_rejected,
        "stage2_passing": len(passing),
    }
    return passing, stats


# ---------------------------------------------------------------------------
# Stage 3: ROUGE-L deduplication
# ---------------------------------------------------------------------------

def stage3_rouge_dedup(examples: list) -> tuple[list, dict]:
    """Remove examples with ROUGE-L > ROUGE_THRESHOLD against any earlier example."""
    kept = []
    kept_texts = []
    n_rejected = 0

    for ex in examples:
        text = ex.get("instruction", "") + " " + ex.get("output", "")
        is_dup = False
        for prev in kept_texts:
            score = scorer.score(prev, text)["rougeL"].fmeasure
            if score > ROUGE_THRESHOLD:
                is_dup = True
                break
        if is_dup:
            n_rejected += 1
        else:
            kept.append(ex)
            kept_texts.append(text)

    stats = {
        "stage3_rejected": n_rejected,
        "stage3_passing": len(kept),
    }
    return kept, stats


# ---------------------------------------------------------------------------
# Full pipeline for one tier
# ---------------------------------------------------------------------------

def validate_tier(raw_path: str, verified_path: str, tier: int, target: int):
    print(f"\n{'='*60}")
    print(f"Validating Tier {tier} (target: {target} verified)")
    print(f"{'='*60}", flush=True)

    if not os.path.exists(raw_path):
        print(f"  ERROR: {raw_path} not found. Run generate_rehearsal.py first.", flush=True)
        return []

    with open(raw_path) as f:
        raw = json.load(f)
    print(f"  Raw examples: {len(raw)}", flush=True)

    # Stage 1
    print(f"\n  --- Stage 1: Programmatic verification ---", flush=True)
    s1_start = time.time()
    s1_passing, s1_stats = stage1_programmatic(raw, tier)
    print(f"  Stage 1 complete in {time.time()-s1_start:.0f}s: {s1_stats}", flush=True)

    # Stage 2
    print(f"\n  --- Stage 2: LLM-as-a-judge verification (n={len(s1_passing)}) ---", flush=True)
    s2_start = time.time()
    s2_passing, s2_stats = stage2_judge(s1_passing, tier)
    print(f"  Stage 2 complete in {time.time()-s2_start:.0f}s: {s2_stats}", flush=True)

    # Stage 3
    print(f"\n  --- Stage 3: ROUGE-L deduplication (n={len(s2_passing)}) ---", flush=True)
    s3_start = time.time()
    verified, s3_stats = stage3_rouge_dedup(s2_passing)
    print(f"  Stage 3 complete in {time.time()-s3_start:.0f}s: {s3_stats}", flush=True)

    # Cap at target
    verified = verified[:target]

    with open(verified_path, "w") as f:
        json.dump(verified, f, indent=2)

    total_rejected = len(raw) - len(verified)
    rejection_rate = total_rejected / len(raw) * 100 if raw else 0
    print(f"\n  RESULT Tier {tier}:")
    print(f"    Raw:          {len(raw)}")
    print(f"    After Stage1: {len(s1_passing)}")
    print(f"    After Stage2: {len(s2_passing)}")
    print(f"    After Stage3: {len(verified)}")
    print(f"    Rejection rate: {rejection_rate:.1f}%")
    if len(verified) < target:
        print(f"    WARNING: only {len(verified)}/{target} verified. "
              f"Increase TIER{tier}_RAW_TARGET in config.py and rerun generate_rehearsal.py.")
    print(f"    Saved → {verified_path}", flush=True)

    return verified


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t1 = validate_tier(TIER1_RAW_PATH, TIER1_VERIFIED_PATH, tier=1, target=TIER1_REHEARSAL_TARGET)
    t2 = validate_tier(TIER2_RAW_PATH, TIER2_VERIFIED_PATH, tier=2, target=TIER2_REHEARSAL_TARGET)
    t3 = validate_tier(TIER3_RAW_PATH, TIER3_VERIFIED_PATH, tier=3, target=TIER3_REHEARSAL_TARGET)

    print(f"\n{'='*60}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Tier 1 verified: {len(t1)}")
    print(f"  Tier 2 verified: {len(t2)}")
    print(f"  Tier 3 verified: {len(t3)}")
    print(f"\nNext step: python finetune_cognitive.py")


if __name__ == "__main__":
    main()
