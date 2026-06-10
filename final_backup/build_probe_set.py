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
    ds = load_dataset("openai/gsm8k", "main", split="train")
    items = random.sample(list(ds), NUM_PER_DOMAIN * 3)
    out = []
    for it in items:
        if len(out) >= NUM_PER_DOMAIN: break
        ans_str = it["answer"].split("####")[-1].strip().replace(",", "").replace("$", "")
        try:
            correct = int(float(ans_str))
        except ValueError:
            continue
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
    ds = load_dataset("google/boolq", split="train")
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
    ds = load_dataset("allenai/sciq", split="train")
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
