"""Step 1: Build cognitive-tier probe pools from held-out benchmark test sets.

Reserves 20% of each benchmark's test set for probing during training and flags
the remaining 80% for final evaluation. Split indices are saved for reproducibility
so evaluate.py can exclude probe questions.

Output files:
  probes/tier1_pool.json     — 60-80 Tier 1 probes (multi-step derivation)
  probes/tier2_pool.json     — 40-50 Tier 2 probes (extended reasoning)
  probes/tier3_pool.json     — 30-40 Tier 3 probes (pattern recognition/recall)
  probes/split_indices.json  — {benchmark: {probe_indices: [...], eval_indices: [...]}}
"""

import json
import os
import random
import re

from datasets import load_dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROBES_DIR, TIER1_POOL_SIZE, TIER2_POOL_SIZE, TIER3_POOL_SIZE,
    TIER1_POOL_PATH, TIER2_POOL_PATH, TIER3_POOL_PATH, SPLIT_INDICES_PATH,
)

SEED = 42
random.seed(SEED)

os.makedirs(PROBES_DIR, exist_ok=True)

split_indices = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_20_80(n, seed=SEED):
    """Return (probe_indices, eval_indices) — 20% probe, 80% eval."""
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    cut = max(1, int(n * 0.20))
    return idxs[:cut], idxs[cut:]


def count_gsm8k_steps(solution: str) -> int:
    """Count reasoning steps in a GSM8K solution.

    GSM8K solutions separate computation lines with newlines before the #### marker.
    We count non-empty lines before the final answer as individual steps.
    """
    before_answer = solution.split("####")[0]
    lines = [l.strip() for l in before_answer.split("\n") if l.strip()]
    return len(lines)


# ---------------------------------------------------------------------------
# Tier 1: multi-step derivation
# ---------------------------------------------------------------------------

def build_tier1_gsm8k(target: int):
    """GSM8K test problems with 3+ reasoning steps."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    all_items = list(ds)
    probe_idx, eval_idx = split_20_80(len(all_items))
    split_indices["gsm8k"] = {"probe_indices": probe_idx, "eval_indices": eval_idx}

    probes = []
    for i in probe_idx:
        it = all_items[i]
        steps = count_gsm8k_steps(it["answer"])
        if steps < 3:
            continue
        ans_str = it["answer"].split("####")[-1].strip().replace(",", "").replace("$", "")
        try:
            correct = int(float(ans_str))
        except ValueError:
            continue
        rng = random.Random(i)
        distractors = set()
        attempts = 0
        while len(distractors) < 3 and attempts < 50:
            attempts += 1
            delta = rng.choice([-1, 1, -2, 2, -5, 5, -10, 10, -3, 3, -20, 20])
            cand = correct + delta
            if cand != correct and cand > 0:
                distractors.add(cand)
        if len(distractors) < 3:
            continue
        choices = [str(correct)] + [str(d) for d in distractors]
        rng.shuffle(choices)
        ans_idx = choices.index(str(correct))
        probes.append({
            "question": it["question"] + "\nWhat is the final numerical answer?",
            "choices": choices,
            "answer": ans_idx,
            "answer_raw": str(correct),
            "source_benchmark": "gsm8k",
            "tier": 1,
            "n_steps": steps,
            "cognitive_justification": (
                f"Multi-step arithmetic word problem requiring {steps} chained computation "
                "steps; each step's output feeds the next."
            ),
        })
        if len(probes) >= target:
            break
    return probes


def build_tier1_mmlu_subject(subject: str, target: int):
    """MMLU test questions for a high-derivation subject."""
    dataset_name = subject.replace("mmlu_", "")
    try:
        ds = load_dataset("cais/mmlu", dataset_name, split="test")
    except Exception:
        ds = load_dataset("lukaemon/mmlu", dataset_name, split="test")
    all_items = list(ds)
    probe_idx, eval_idx = split_20_80(len(all_items))
    split_indices[subject] = {"probe_indices": probe_idx, "eval_indices": eval_idx}

    probes = []
    for i in probe_idx:
        it = all_items[i]
        choices = it["choices"] if isinstance(it["choices"], list) else list(it["choices"])
        answer = it["answer"]
        if isinstance(answer, str):
            try:
                answer = int(answer)
            except ValueError:
                label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                answer = label_map.get(answer.strip(), 0)
        probes.append({
            "question": it["question"],
            "choices": choices,
            "answer": int(answer),
            "source_benchmark": subject,
            "tier": 1,
            "cognitive_justification": (
                f"MMLU {dataset_name.replace('_', ' ')} question requiring formal derivation "
                "or multi-step reasoning within a technical domain."
            ),
        })
        if len(probes) >= target:
            break
    return probes


def build_tier1_pool():
    pool = []
    gsm_target = min(30, TIER1_POOL_SIZE // 2)
    pool.extend(build_tier1_gsm8k(gsm_target))
    print(f"  GSM8K (≥3 steps): {len(pool)}")

    per_subject = max(6, (TIER1_POOL_SIZE - len(pool)) // 4)
    for subj in ["mmlu_abstract_algebra", "mmlu_college_physics",
                 "mmlu_formal_logic", "mmlu_college_chemistry"]:
        prev = len(pool)
        pool.extend(build_tier1_mmlu_subject(subj, per_subject))
        print(f"  {subj}: {len(pool) - prev}")
        if len(pool) >= TIER1_POOL_SIZE:
            break

    return pool[:TIER1_POOL_SIZE]


# ---------------------------------------------------------------------------
# Tier 2: extended reasoning
# ---------------------------------------------------------------------------

def build_tier2_hellaswag(target: int):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    all_items = list(ds)
    probe_idx, eval_idx = split_20_80(len(all_items))
    split_indices["hellaswag"] = {"probe_indices": probe_idx, "eval_indices": eval_idx}

    rng = random.Random(SEED)
    sampled_probe_idx = rng.sample(probe_idx, min(target, len(probe_idx)))
    probes = []
    for i in sampled_probe_idx:
        it = all_items[i]
        try:
            label = int(it["label"])
        except (ValueError, TypeError):
            continue
        probes.append({
            "question": it["ctx"] + "\nWhat happens next?",
            "choices": list(it["endings"]),
            "answer": label,
            "source_benchmark": "hellaswag",
            "tier": 2,
            "cognitive_justification": (
                "Commonsense/narrative reasoning: requires understanding the context "
                "across multiple sentences and inferring the most plausible continuation."
            ),
        })
    return probes


def build_tier2_piqa(target: int):
    ds = load_dataset("ybisk/piqa", split="validation")
    all_items = list(ds)
    probe_idx, eval_idx = split_20_80(len(all_items))
    split_indices["piqa"] = {"probe_indices": probe_idx, "eval_indices": eval_idx}

    rng = random.Random(SEED)
    sampled_probe_idx = rng.sample(probe_idx, min(target, len(probe_idx)))
    probes = []
    for i in sampled_probe_idx:
        it = all_items[i]
        probes.append({
            "question": it["goal"],
            "choices": [it["sol1"], it["sol2"]],
            "answer": int(it["label"]),
            "source_benchmark": "piqa",
            "tier": 2,
            "cognitive_justification": (
                "Physical intuition question: requires reasoning about real-world "
                "cause-and-effect relationships to select a valid procedure."
            ),
        })
    return probes


def build_tier2_lambada(target: int):
    """LAMBADA uses a fill-in-the-last-word format; convert to 4-choice MC."""
    try:
        ds = load_dataset("EleutherAI/lambada_openai", split="test")
    except Exception:
        ds = load_dataset("lambada", split="test")
    all_items = list(ds)
    probe_idx, eval_idx = split_20_80(len(all_items))
    split_indices["lambada_openai"] = {"probe_indices": probe_idx, "eval_indices": eval_idx}

    rng = random.Random(SEED)
    sampled_probe_idx = rng.sample(probe_idx, min(target * 3, len(probe_idx)))

    # Build a word pool for distractors
    word_pool = set()
    for it in all_items:
        text = it.get("text", "")
        words = re.findall(r"\b[a-z]{3,10}\b", text.lower())
        word_pool.update(words)
    word_pool = list(word_pool)

    probes = []
    for i in sampled_probe_idx:
        if len(probes) >= target:
            break
        it = all_items[i]
        text = it.get("text", "")
        words = text.rsplit(None, 1)
        if len(words) != 2:
            continue
        context, correct = words
        correct_clean = re.sub(r"[^a-zA-Z']", "", correct).lower()
        if not correct_clean or len(correct_clean) < 2:
            continue
        distractors = []
        rng2 = random.Random(i)
        candidates = [w for w in word_pool if w != correct_clean and len(w) >= 2]
        rng2.shuffle(candidates)
        distractors = candidates[:3]
        if len(distractors) < 3:
            continue
        choices = [correct_clean] + distractors
        rng2.shuffle(choices)
        ans_idx = choices.index(correct_clean)
        probes.append({
            "question": context + " _____\nWhat word fills in the blank?",
            "choices": choices,
            "answer": ans_idx,
            "source_benchmark": "lambada_openai",
            "tier": 2,
            "cognitive_justification": (
                "Long-range language coherence: requires tracking coreference and "
                "narrative thread across a full paragraph to predict the final word."
            ),
        })
    return probes


def build_tier2_pool():
    per = TIER2_POOL_SIZE // 3 + 1
    pool = []
    pool.extend(build_tier2_hellaswag(per))
    print(f"  HellaSwag: {len(pool)}")
    t2 = len(pool)
    #pool.extend(build_tier2_piqa(per))  # PIQA loading script unsupported
    print(f"  PIQA: {len(pool) - t2}")
    t3 = len(pool)
    pool.extend(build_tier2_lambada(per))
    print(f"  LAMBADA: {len(pool) - t3}")
    return pool[:TIER2_POOL_SIZE]


# ---------------------------------------------------------------------------
# Tier 3: pattern recognition / recall
# ---------------------------------------------------------------------------

def build_tier3_arc_easy(target: int):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    all_items = list(ds)
    probe_idx, eval_idx = split_20_80(len(all_items))
    split_indices["arc_easy"] = {"probe_indices": probe_idx, "eval_indices": eval_idx}

    rng = random.Random(SEED)
    sampled = rng.sample(probe_idx, min(target, len(probe_idx)))
    probes = []
    for i in sampled:
        it = all_items[i]
        choices = it["choices"]["text"]
        labels = it["choices"]["label"]
        try:
            ans_idx = labels.index(it["answerKey"])
        except ValueError:
            continue
        probes.append({
            "question": it["question"],
            "choices": choices,
            "answer": ans_idx,
            "source_benchmark": "arc_easy",
            "tier": 3,
            "cognitive_justification": (
                "Direct science fact recall or single-step elimination; does not "
                "require chained computation."
            ),
        })
    return probes


def build_tier3_boolq(target: int):
    ds = load_dataset("google/boolq", split="validation")
    all_items = list(ds)
    probe_idx, eval_idx = split_20_80(len(all_items))
    split_indices["boolq"] = {"probe_indices": probe_idx, "eval_indices": eval_idx}

    rng = random.Random(SEED)
    sampled = rng.sample(probe_idx, min(target, len(probe_idx)))
    probes = []
    for i in sampled:
        it = all_items[i]
        probes.append({
            "question": it["passage"] + "\n\nQuestion: " + it["question"],
            "choices": ["No", "Yes"],
            "answer": int(it["answer"]),
            "source_benchmark": "boolq",
            "tier": 3,
            "cognitive_justification": (
                "Binary yes/no reading comprehension; typically solvable by locating "
                "the key sentence in the passage without multi-step inference."
            ),
        })
    return probes


def build_tier3_mmlu_subject(subject: str, target: int):
    dataset_name = subject.replace("mmlu_", "")
    try:
        ds = load_dataset("cais/mmlu", dataset_name, split="test")
    except Exception:
        ds = load_dataset("lukaemon/mmlu", dataset_name, split="test")
    all_items = list(ds)
    probe_idx, eval_idx = split_20_80(len(all_items))
    split_indices[subject] = {"probe_indices": probe_idx, "eval_indices": eval_idx}

    probes = []
    for i in probe_idx[:target]:
        it = all_items[i]
        choices = it["choices"] if isinstance(it["choices"], list) else list(it["choices"])
        answer = it["answer"]
        if isinstance(answer, str):
            try:
                answer = int(answer)
            except ValueError:
                label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                answer = label_map.get(answer.strip(), 0)
        probes.append({
            "question": it["question"],
            "choices": choices,
            "answer": int(answer),
            "source_benchmark": subject,
            "tier": 3,
            "cognitive_justification": (
                f"MMLU {dataset_name.replace('_', ' ')} — empirically stable or improving "
                "after SFT; used as sanity-check probes."
            ),
        })
    return probes


def build_tier3_pool():
    per = TIER3_POOL_SIZE // 4 + 1
    pool = []
    pool.extend(build_tier3_arc_easy(per))
    print(f"  ARC-Easy: {len(pool)}")
    t2 = len(pool)
    pool.extend(build_tier3_boolq(per))
    print(f"  BoolQ: {len(pool) - t2}")
    t3 = len(pool)
    for subj in ["mmlu_high_school_physics", "mmlu_medical_genetics", "mmlu_econometrics"]:
        pool.extend(build_tier3_mmlu_subject(subj, max(3, per // 3)))
    print(f"  MMLU (stable subjects): {len(pool) - t3}")
    return pool[:TIER3_POOL_SIZE]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Building Tier 1 probe pool (multi-step derivation) ===")
    tier1 = build_tier1_pool()
    print(f"Tier 1 total: {len(tier1)}\n")

    print("=== Building Tier 2 probe pool (extended reasoning) ===")
    tier2 = build_tier2_pool()
    print(f"Tier 2 total: {len(tier2)}\n")

    print("=== Building Tier 3 probe pool (pattern recognition/recall) ===")
    tier3 = build_tier3_pool()
    print(f"Tier 3 total: {len(tier3)}\n")

    with open(TIER1_POOL_PATH, "w") as f:
        json.dump(tier1, f, indent=2)
    with open(TIER2_POOL_PATH, "w") as f:
        json.dump(tier2, f, indent=2)
    with open(TIER3_POOL_PATH, "w") as f:
        json.dump(tier3, f, indent=2)
    with open(SPLIT_INDICES_PATH, "w") as f:
        json.dump(split_indices, f, indent=2)

    print(f"Saved probe pools and split indices to {PROBES_DIR}/")
    print(f"  tier1_pool.json      : {len(tier1)} probes")
    print(f"  tier2_pool.json      : {len(tier2)} probes")
    print(f"  tier3_pool.json      : {len(tier3)} probes")
    print(f"  split_indices.json   : {len(split_indices)} benchmarks tracked")
    print("\nBenchmark 80% eval splits are recorded. Pass split_indices.json to")
    print("evaluate.py so it can exclude probe questions from final benchmarks.")


if __name__ == "__main__":
    main()
