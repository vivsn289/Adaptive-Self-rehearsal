"""Step 2: Base model self-generation of rehearsal bank.

Uses Qwen2.5-3B BASE model (not instruct) to generate rehearsal examples while
its cognitive capabilities are intact. Generated examples are saved raw; quality
filtering happens in validate_rehearsal.py.

Output files:
  rehearsal_bank/raw_tier1.json  — multi-step math, symbolic, logical examples
  rehearsal_bank/raw_tier2.json  — causal reasoning, reading comprehension
  rehearsal_bank/raw_tier3.json  — factual recall, single-hop pattern matching

RESUMABLE: Checks for existing output files and skips completed tiers.
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
    BASE_MODEL, REHEARSAL_DIR,
    TIER1_RAW_PATH, TIER2_RAW_PATH, TIER3_RAW_PATH,
    TIER1_RAW_TARGET, TIER2_RAW_TARGET, TIER3_RAW_TARGET,
)

SEED = 42
random.seed(SEED)
os.makedirs(REHEARSAL_DIR, exist_ok=True)

# Partial checkpoint paths for within-tier resumability.
# If generation is interrupted mid-tier, restart picks up from these files.
CKPT_TIER1_MATH = os.path.join(REHEARSAL_DIR, "ckpt_tier1_math.json")
CKPT_TIER1_LOGIC = os.path.join(REHEARSAL_DIR, "ckpt_tier1_logic.json")
CKPT_TIER2_CAUSAL = os.path.join(REHEARSAL_DIR, "ckpt_tier2_causal.json")
CKPT_TIER2_COMP = os.path.join(REHEARSAL_DIR, "ckpt_tier2_comp.json")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

print(f"Loading {BASE_MODEL} (base, NOT instruct)...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Model loaded.", flush=True)


# ---------------------------------------------------------------------------
# Core generation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(prompts, max_new_tokens=400, temperature=0.8, top_p=0.9):
    """Generate responses for a list of prompts using left-padded batching."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    results = []
    for i, seq in enumerate(out):
        prompt_len = inputs["input_ids"].shape[1]
        generated = seq[prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        results.append(text)
    return results


# ---------------------------------------------------------------------------
# Programmatic math generation (guaranteed-correct answers)
# ---------------------------------------------------------------------------

def generate_programmatic_modular(n: int, rng: random.Random):
    """Generate ((a × b) mod c + d) mod e style problems with verified answers."""
    examples = []
    for _ in range(n * 3):
        if len(examples) >= n:
            break
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        c = rng.randint(7, 31)
        d = rng.randint(5, 50)
        e = rng.randint(7, 23)
        correct = ((a * b) % c + d) % e
        problem = (
            f"Compute ((({a} × {b}) mod {c}) + {d}) mod {e}. "
            f"Show each step."
        )
        solution = (
            f"Step 1: {a} × {b} = {a * b}\n"
            f"Step 2: {a * b} mod {c} = {(a * b) % c}\n"
            f"Step 3: {(a * b) % c} + {d} = {(a * b) % c + d}\n"
            f"Step 4: {(a * b) % c + d} mod {e} = {correct}\n"
            f"Answer: {correct}"
        )
        examples.append({
            "instruction": problem,
            "output": solution,
            "answer": correct,
            "subtype": "modular_arithmetic",
            "verified_programmatic": True,
        })
    return examples


def generate_programmatic_chained_arithmetic(n: int, rng: random.Random):
    """Generate 4-step chained arithmetic word problems with verified answers."""
    templates = [
        ("A store has {a} boxes with {b} items each. After selling {c} items, "
         "they restock {d} more. How many items total?",
         lambda a, b, c, d: a * b - c + d,
         lambda a, b, c, d: (
             f"Step 1: Total items = {a} × {b} = {a * b}\n"
             f"Step 2: After selling {c}: {a * b} - {c} = {a * b - c}\n"
             f"Step 3: After restocking {d}: {a * b - c} + {d} = {a * b - c + d}\n"
             f"Answer: {a * b - c + d}"
         )),
        ("A train travels {a} km at {b} km/h, then {c} km at {d} km/h. "
         "What is the total travel time in minutes?",
         lambda a, b, c, d: int((a / b + c / d) * 60),
         lambda a, b, c, d: (
             f"Step 1: Time for first leg = {a}/{b} = {a/b:.2f} hours\n"
             f"Step 2: Time for second leg = {c}/{d} = {c/d:.2f} hours\n"
             f"Step 3: Total time = {a/b:.2f} + {c/d:.2f} = {a/b + c/d:.2f} hours\n"
             f"Step 4: Convert to minutes = {a/b + c/d:.2f} × 60 = {int((a/b + c/d)*60)}\n"
             f"Answer: {int((a/b + c/d)*60)} minutes"
         )),
    ]
    examples = []
    for _ in range(n * 2):
        if len(examples) >= n:
            break
        tmpl_idx = rng.randint(0, len(templates) - 1)
        tmpl, fn, sol_fn = templates[tmpl_idx]
        a = rng.randint(3, 20)
        b = rng.randint(2, 15)
        c = rng.randint(1, a * b - 1)
        d = rng.randint(1, 30)
        try:
            ans = fn(a, b, c, d)
            if not isinstance(ans, int) or ans < 0:
                continue
            sol = sol_fn(a, b, c, d)
            problem = tmpl.format(a=a, b=b, c=c, d=d)
        except (ZeroDivisionError, ValueError):
            continue
        examples.append({
            "instruction": problem,
            "output": sol,
            "answer": ans,
            "subtype": "chained_arithmetic",
            "verified_programmatic": True,
        })
    return examples


# ---------------------------------------------------------------------------
# Few-shot prompts for base model generation
# ---------------------------------------------------------------------------

FEW_SHOT_MATH_WORDPROBLEM = """Generate a multi-step math word problem that requires EXACTLY 3 or 4 sequential arithmetic operations, where each step's result is needed for the next step. Include the full step-by-step solution.

Example 1:
Problem: A baker made 8 batches of cookies with 12 cookies each. She sold 3/4 of them and then baked 15 more. How many cookies does she have now?
Solution:
Step 1: Total cookies = 8 × 12 = 96
Step 2: Cookies sold = 3/4 × 96 = 72
Step 3: Cookies remaining = 96 - 72 = 24
Step 4: After baking more = 24 + 15 = 39
Answer: 39

Example 2:
Problem: A car travels 150 km on 12.5 liters of fuel. If fuel costs $1.80 per liter and the driver needs to travel 300 km, what is the total fuel cost?
Solution:
Step 1: Fuel efficiency = 150 / 12.5 = 12 km per liter
Step 2: Fuel needed = 300 / 12 = 25 liters
Step 3: Total cost = 25 × 1.80 = $45.00
Answer: $45.00

Example 3:
Problem: A rectangle has length 3 times its width. If the perimeter is 96 cm, what is the area?
Solution:
Step 1: Let width = w, then length = 3w
Step 2: Perimeter = 2(w + 3w) = 8w = 96, so w = 12
Step 3: Length = 3 × 12 = 36
Step 4: Area = 12 × 36 = 432 sq cm
Answer: 432 sq cm

Now generate a NEW multi-step word problem (different from the examples above) with full step-by-step solution:
Problem:"""

FEW_SHOT_LOGIC = """Generate a multi-step logical inference problem where the conclusion requires chaining at least 3 premises together. Include the full reasoning chain.

Example 1:
Premises:
1. All mammals are warm-blooded.
2. All warm-blooded animals regulate their body temperature internally.
3. Whales are mammals.
4. Animals that regulate temperature internally can survive in cold water.
Question: Can whales survive in cold water?
Reasoning:
Step 1: Whales are mammals (Premise 3).
Step 2: All mammals are warm-blooded (Premise 1), so whales are warm-blooded.
Step 3: Warm-blooded animals regulate temperature internally (Premise 2), so whales do too.
Step 4: Animals that regulate temperature internally can survive in cold water (Premise 4).
Conclusion: Yes, whales can survive in cold water.

Example 2:
Premises:
1. If it rains, the ground gets wet.
2. If the ground is wet, the soccer field becomes muddy.
3. If the field is muddy, the game gets cancelled.
4. It rained yesterday.
Question: Was the game cancelled?
Reasoning:
Step 1: It rained yesterday (Premise 4).
Step 2: Rain makes the ground wet (Premise 1), so the ground is wet.
Step 3: Wet ground makes the field muddy (Premise 2), so the field is muddy.
Step 4: A muddy field leads to cancellation (Premise 3).
Conclusion: Yes, the game was cancelled.

Now generate a NEW logical inference problem (different from above) with full reasoning chain:
Premises:"""

FEW_SHOT_CAUSAL = """Generate a causal reasoning scenario where understanding the final outcome requires tracing through a chain of 2-3 cause-and-effect relationships. Include the question and explanation.

Example 1:
Scenario: A city experiences a prolonged drought. As a result, the reservoir water level drops critically low. The water authority implements strict rationing, which causes local farmers to reduce irrigation. With less water, crop yields fall significantly.
Question: What is the likely effect of the drought on food availability in the region?
Explanation: The drought reduced reservoir levels → water rationing followed → farmers reduced irrigation → crop yields fell. Therefore, food availability in the region likely decreased.
Answer: Food availability in the region likely decreased due to lower crop yields caused by reduced irrigation from drought-induced water rationing.

Example 2:
Scenario: A major highway closes for construction. Commuters divert to local roads, increasing traffic significantly. Local businesses along the detour route see more customers, but residents suffer from noise and congestion.
Question: What are the direct economic effects on local businesses along the detour?
Explanation: Highway closure → traffic diverted to local roads → increased foot traffic past local businesses → higher customer volume → increased revenue for local businesses.
Answer: Local businesses along the detour likely see increased revenue due to higher foot traffic from diverted commuters.

Now generate a NEW causal reasoning scenario (different from above) with question and explanation:
Scenario:"""

FEW_SHOT_FACTUAL = """Generate a factual recall question with a clear, direct answer. The question should be answerable from general knowledge in a single step.

Example 1:
Question: What is the chemical symbol for gold?
Answer: Au

Example 2:
Question: In which year did World War II end?
Answer: 1945

Example 3:
Question: What is the capital city of Japan?
Answer: Tokyo

Now generate a NEW factual question (different from above) with its answer:
Question:"""

FEW_SHOT_COMPREHENSION = """Generate a short reading passage followed by a question that requires understanding the passage but can be answered directly from it.

Example:
Passage: The Great Barrier Reef, located off the coast of Queensland, Australia, is the world's largest coral reef system. It stretches over 2,300 kilometers and is composed of over 2,900 individual reefs. The reef supports an enormous diversity of marine life and is listed as a UNESCO World Heritage Site.
Question: Where is the Great Barrier Reef located?
Answer: Off the coast of Queensland, Australia.

Now generate a NEW passage with a question and answer (different from above):
Passage:"""


# ---------------------------------------------------------------------------
# Tier 1 generation
# ---------------------------------------------------------------------------

def parse_generated_math(text: str, prefix: str):
    """Extract (problem, solution) from model output."""
    full_text = prefix + text
    if "Solution:" in full_text and ("Step 1:" in full_text or "Step 2:" in full_text):
        parts = full_text.split("Solution:", 1)
        problem = parts[0].replace("Problem:", "").strip()
        solution = "Solution:" + parts[1].strip()
        if "Answer:" in solution:
            solution = solution[:solution.rfind("Answer:") + solution[solution.rfind("Answer:"):].find("\n") + 1].strip()
        return problem.strip(), solution.strip()
    return None, None


def generate_tier1_model_math(n_target: int, rng: random.Random, checkpoint_path: str = None):
    """Model-generated multi-step math word problems."""
    examples = []
    # Resume from partial checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            examples = json.load(f)
        print(f"  [Tier1-math] Resumed from checkpoint: {len(examples)} examples", flush=True)

    attempts = 0
    max_attempts = n_target * 6
    batch_size = 4
    last_ckpt_count = len(examples)

    while len(examples) < n_target and attempts < max_attempts:
        prompts = [FEW_SHOT_MATH_WORDPROBLEM] * min(batch_size, n_target - len(examples) + 4)
        try:
            responses = generate_batch(prompts, max_new_tokens=512, temperature=0.85, top_p=0.92)
        except Exception as e:
            print(f"  [generate_tier1_math] batch error: {e}", flush=True)
            attempts += batch_size
            continue

        for resp in responses:
            attempts += 1
            problem, solution = parse_generated_math(resp, "")
            if not problem or not solution:
                if "Step 1:" in resp:
                    lines = resp.split("\n")
                    problem_lines = [l for l in lines if not l.startswith("Step") and not l.startswith("Answer")]
                    solution_lines = [l for l in lines if l.startswith("Step") or l.startswith("Answer")]
                    if problem_lines and solution_lines:
                        problem = " ".join(problem_lines[:3]).strip()
                        solution = "\n".join(solution_lines).strip()
                    else:
                        continue
                else:
                    continue
            steps = len(re.findall(r"Step \d+:", solution))
            if steps < 3:
                continue
            examples.append({
                "instruction": problem,
                "output": solution,
                "subtype": "math_word_problem",
                "verified_programmatic": False,
            })

        # Save checkpoint every 100 new examples so interrupted runs can resume mid-tier
        if checkpoint_path and len(examples) - last_ckpt_count >= 100:
            with open(checkpoint_path, "w") as f:
                json.dump(examples, f)
            last_ckpt_count = len(examples)

        print(f"  [Tier1-math] {len(examples)}/{n_target} (attempts {attempts})", flush=True)
    return examples


def generate_tier1_logic(n_target: int, rng: random.Random, checkpoint_path: str = None):
    """Model-generated logical inference problems."""
    examples = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            examples = json.load(f)
        print(f"  [Tier1-logic] Resumed from checkpoint: {len(examples)} examples", flush=True)

    attempts = 0
    max_attempts = n_target * 6
    batch_size = 4
    last_ckpt_count = len(examples)

    while len(examples) < n_target and attempts < max_attempts:
        prompts = [FEW_SHOT_LOGIC] * min(batch_size, n_target - len(examples) + 4)
        try:
            responses = generate_batch(prompts, max_new_tokens=512, temperature=0.8, top_p=0.9)
        except Exception as e:
            print(f"  [generate_tier1_logic] batch error: {e}", flush=True)
            attempts += batch_size
            continue

        for resp in responses:
            attempts += 1
            full = "Premises:\n" + resp
            if "Conclusion:" not in full and "Answer:" not in full:
                continue
            if "Step 1:" not in full and "Step 2:" not in full:
                continue
            steps = len(re.findall(r"Step \d+:", full))
            if steps < 3:
                continue
            q_match = re.search(r"Question:\s*(.+?)(?:\n|Reasoning:)", full, re.DOTALL)
            question = q_match.group(1).strip() if q_match else "What can be concluded?"
            reasoning = full
            examples.append({
                "instruction": question + "\n\n" + full.split("Question:")[0],
                "output": reasoning,
                "subtype": "logical_inference",
                "verified_programmatic": False,
            })

        if checkpoint_path and len(examples) - last_ckpt_count >= 100:
            with open(checkpoint_path, "w") as f:
                json.dump(examples, f)
            last_ckpt_count = len(examples)

        print(f"  [Tier1-logic] {len(examples)}/{n_target} (attempts {attempts})", flush=True)
    return examples


def build_tier1_raw(rng: random.Random):
    """Build raw Tier 1 rehearsal examples."""
    print("\n=== Generating Tier 1 raw rehearsal bank ===", flush=True)

    # Programmatic (guaranteed correct)
    n_modular = TIER1_RAW_TARGET // 5
    n_chained = TIER1_RAW_TARGET // 5
    print(f"  Programmatic modular arithmetic: target {n_modular}", flush=True)
    modular = generate_programmatic_modular(n_modular, rng)
    print(f"  Programmatic chained arithmetic: target {n_chained}", flush=True)
    chained = generate_programmatic_chained_arithmetic(n_chained, rng)

    # Model-generated (with checkpoint resumability)
    n_math = TIER1_RAW_TARGET * 2 // 5
    n_logic = TIER1_RAW_TARGET - n_modular - n_chained - n_math
    print(f"  Model math word problems: target {n_math}", flush=True)
    model_math = generate_tier1_model_math(n_math, rng, checkpoint_path=CKPT_TIER1_MATH)
    print(f"  Model logical inference: target {n_logic}", flush=True)
    logic = generate_tier1_logic(n_logic, rng, checkpoint_path=CKPT_TIER1_LOGIC)

    all_raw = modular + chained + model_math + logic
    rng.shuffle(all_raw)
    print(f"Tier 1 raw total: {len(all_raw)}", flush=True)
    return all_raw


# ---------------------------------------------------------------------------
# Tier 2 generation
# ---------------------------------------------------------------------------

def generate_tier2_causal(n_target: int, rng: random.Random, checkpoint_path: str = None):
    examples = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            examples = json.load(f)
        print(f"  [Tier2-causal] Resumed from checkpoint: {len(examples)} examples", flush=True)

    attempts = 0
    batch_size = 4
    last_ckpt_count = len(examples)

    while len(examples) < n_target and attempts < n_target * 6:
        prompts = [FEW_SHOT_CAUSAL] * min(batch_size, n_target - len(examples) + 4)
        try:
            responses = generate_batch(prompts, max_new_tokens=300, temperature=0.8, top_p=0.9)
        except Exception as e:
            print(f"  [generate_tier2_causal] batch error: {e}", flush=True)
            attempts += batch_size
            continue
        for resp in responses:
            attempts += 1
            full = "Scenario: " + resp
            if "Question:" not in full or "Answer:" not in full:
                continue
            q_match = re.search(r"Question:\s*(.+?)(?:\n|Explanation:)", full, re.DOTALL)
            a_match = re.search(r"Answer:\s*(.+?)(?:\n\n|$)", full, re.DOTALL)
            if not q_match or not a_match:
                continue
            scenario = full.split("Question:")[0].replace("Scenario:", "").strip()
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()
            if len(answer) < 10:
                continue
            examples.append({
                "instruction": scenario + "\n\n" + question,
                "output": answer,
                "subtype": "causal_reasoning",
                "verified_programmatic": False,
            })

        if checkpoint_path and len(examples) - last_ckpt_count >= 100:
            with open(checkpoint_path, "w") as f:
                json.dump(examples, f)
            last_ckpt_count = len(examples)

        print(f"  [Tier2-causal] {len(examples)}/{n_target} (attempts {attempts})", flush=True)
    return examples


def generate_tier2_comprehension(n_target: int, rng: random.Random, checkpoint_path: str = None):
    examples = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            examples = json.load(f)
        print(f"  [Tier2-comp] Resumed from checkpoint: {len(examples)} examples", flush=True)

    attempts = 0
    batch_size = 4
    last_ckpt_count = len(examples)

    while len(examples) < n_target and attempts < n_target * 6:
        prompts = [FEW_SHOT_COMPREHENSION] * min(batch_size, n_target - len(examples) + 4)
        try:
            responses = generate_batch(prompts, max_new_tokens=300, temperature=0.75, top_p=0.9)
        except Exception as e:
            print(f"  [generate_tier2_comp] batch error: {e}", flush=True)
            attempts += batch_size
            continue
        for resp in responses:
            attempts += 1
            full = "Passage: " + resp
            if "Question:" not in full or "Answer:" not in full:
                continue
            p_match = re.search(r"Passage:\s*(.+?)Question:", full, re.DOTALL)
            q_match = re.search(r"Question:\s*(.+?)Answer:", full, re.DOTALL)
            a_match = re.search(r"Answer:\s*(.+?)(?:\n\n|$)", full, re.DOTALL)
            if not p_match or not q_match or not a_match:
                continue
            passage = p_match.group(1).strip()
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()
            if len(passage) < 50 or len(answer) < 5:
                continue
            examples.append({
                "instruction": passage + "\n\nQuestion: " + question,
                "output": answer,
                "subtype": "reading_comprehension",
                "verified_programmatic": False,
            })

        if checkpoint_path and len(examples) - last_ckpt_count >= 100:
            with open(checkpoint_path, "w") as f:
                json.dump(examples, f)
            last_ckpt_count = len(examples)

        print(f"  [Tier2-comp] {len(examples)}/{n_target} (attempts {attempts})", flush=True)
    return examples


def build_tier2_raw(rng: random.Random):
    print("\n=== Generating Tier 2 raw rehearsal bank ===", flush=True)
    half = TIER2_RAW_TARGET // 2
    causal = generate_tier2_causal(half, rng, checkpoint_path=CKPT_TIER2_CAUSAL)
    comp = generate_tier2_comprehension(TIER2_RAW_TARGET - half, rng, checkpoint_path=CKPT_TIER2_COMP)
    all_raw = causal + comp
    rng.shuffle(all_raw)
    print(f"Tier 2 raw total: {len(all_raw)}", flush=True)
    return all_raw


# ---------------------------------------------------------------------------
# Tier 3 generation
# ---------------------------------------------------------------------------

def generate_tier3_factual(n_target: int, rng: random.Random):
    examples = []
    attempts = 0
    batch_size = 6

    while len(examples) < n_target and attempts < n_target * 5:
        prompts = [FEW_SHOT_FACTUAL] * min(batch_size, n_target - len(examples) + 6)
        try:
            responses = generate_batch(prompts, max_new_tokens=100, temperature=0.7, top_p=0.9)
        except Exception as e:
            print(f"  [generate_tier3_factual] batch error: {e}", flush=True)
            attempts += batch_size
            continue
        for resp in responses:
            attempts += 1
            full = "Question: " + resp
            if "Answer:" not in full:
                continue
            q_match = re.search(r"Question:\s*(.+?)Answer:", full, re.DOTALL)
            a_match = re.search(r"Answer:\s*(.+?)(?:\n|$)", full)
            if not q_match or not a_match:
                continue
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()
            if len(question) < 10 or len(answer) < 1:
                continue
            examples.append({
                "instruction": question,
                "output": answer,
                "subtype": "factual_recall",
                "verified_programmatic": False,
            })
        print(f"  [Tier3-factual] {len(examples)}/{n_target} (attempts {attempts})", flush=True)
    return examples


def build_tier3_raw(rng: random.Random):
    print("\n=== Generating Tier 3 raw rehearsal bank ===", flush=True)
    factual = generate_tier3_factual(TIER3_RAW_TARGET, rng)
    print(f"Tier 3 raw total: {len(factual)}", flush=True)
    return factual


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _remove_ckpt(path: str):
    if path and os.path.exists(path):
        os.remove(path)


def main():
    rng = random.Random(SEED)

    if os.path.exists(TIER1_RAW_PATH):
        with open(TIER1_RAW_PATH) as f:
            t1 = json.load(f)
        print(f"SKIP Tier 1 — already exists ({len(t1)} examples)", flush=True)
    else:
        t1 = build_tier1_raw(rng)
        with open(TIER1_RAW_PATH, "w") as f:
            json.dump(t1, f, indent=2)
        print(f"Saved {len(t1)} Tier 1 raw examples → {TIER1_RAW_PATH}", flush=True)
        # Tier 1 complete — remove partial checkpoints
        _remove_ckpt(CKPT_TIER1_MATH)
        _remove_ckpt(CKPT_TIER1_LOGIC)

    if os.path.exists(TIER2_RAW_PATH):
        with open(TIER2_RAW_PATH) as f:
            t2 = json.load(f)
        print(f"SKIP Tier 2 — already exists ({len(t2)} examples)", flush=True)
    else:
        t2 = build_tier2_raw(rng)
        with open(TIER2_RAW_PATH, "w") as f:
            json.dump(t2, f, indent=2)
        print(f"Saved {len(t2)} Tier 2 raw examples → {TIER2_RAW_PATH}", flush=True)
        _remove_ckpt(CKPT_TIER2_CAUSAL)
        _remove_ckpt(CKPT_TIER2_COMP)

    if os.path.exists(TIER3_RAW_PATH):
        with open(TIER3_RAW_PATH) as f:
            t3 = json.load(f)
        print(f"SKIP Tier 3 — already exists ({len(t3)} examples)", flush=True)
    else:
        t3 = build_tier3_raw(rng)
        with open(TIER3_RAW_PATH, "w") as f:
            json.dump(t3, f, indent=2)
        print(f"Saved {len(t3)} Tier 3 raw examples → {TIER3_RAW_PATH}", flush=True)

    print(f"\n=== Raw generation complete ===")
    print(f"  Tier 1: {len(t1)}")
    print(f"  Tier 2: {len(t2)}")
    print(f"  Tier 3: {len(t3)}")
    print(f"\nNext step: python validate_rehearsal.py")


if __name__ == "__main__":
    main()
