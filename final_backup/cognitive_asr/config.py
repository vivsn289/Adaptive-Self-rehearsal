"""Shared constants and paths for Cognitive-Tier ASR v2."""

import os

# ---------------------------------------------------------------------------
# Tier source definitions
# ---------------------------------------------------------------------------
TIER1_SOURCES = [
    "gsm8k",
    "mmlu_abstract_algebra",
    "mmlu_college_physics",
    "mmlu_formal_logic",
    "mmlu_college_chemistry",
]
TIER2_SOURCES = ["hellaswag", "lambada_openai", "piqa"]
TIER3_SOURCES = [
    "arc_easy",
    "boolq",
    "mmlu_high_school_physics",
    "mmlu_medical_genetics",
    "mmlu_econometrics",
]

# ---------------------------------------------------------------------------
# Probe pool sizes (total questions in the held-out pool)
# ---------------------------------------------------------------------------
TIER1_POOL_SIZE = 70
TIER2_POOL_SIZE = 45
TIER3_POOL_SIZE = 35

# Per-checkpoint sample sizes drawn from the pool
TIER1_SAMPLE_SIZE = 18
TIER2_SAMPLE_SIZE = 12
TIER3_SAMPLE_SIZE = 9

# ---------------------------------------------------------------------------
# Rehearsal bank targets (verified examples after quality filtering)
# ---------------------------------------------------------------------------
TIER1_REHEARSAL_TARGET = 900
TIER2_REHEARSAL_TARGET = 250
TIER3_REHEARSAL_TARGET = 150

# Overgeneration multipliers (raw → verified ratios are rough targets)
# Tier1: ~30-40% rejection → generate 1400-1500
# Tier2: ~50% rejection → generate 400
# Tier3: ~40% rejection → generate 250
TIER1_RAW_TARGET = 1450
TIER2_RAW_TARGET = 400
TIER3_RAW_TARGET = 250

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
DEFAULT_PROBE_INTERVAL = 50          # gradient steps between probe evals
DEFAULT_TIER1_THRESHOLD = 5.0        # pp drop that triggers intervention
DEFAULT_TIER2_THRESHOLD = 5.0        # pp; needs 3 consecutive (not 2)
DEFAULT_REHEARSAL_FRACTION = 0.25    # fraction of each batch = rehearsal data
DEFAULT_REWEIGHT_FACTOR = 1.5
DEFAULT_CONSECUTIVE_DROPS_T1 = 2     # triggers after N consecutive drops
DEFAULT_CONSECUTIVE_DROPS_T2 = 3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-3B"
JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)

ALPACA_DATA = os.path.join(_PARENT, "generated_data", "alpaca_52k.json")
BASE_RESULTS = os.path.join(_PARENT, "results", "base")
VANILLA_52K_RESULTS = os.path.join(_PARENT, "results", "alpaca52k")

PROBES_DIR = os.path.join(_HERE, "probes")
REHEARSAL_DIR = os.path.join(_HERE, "rehearsal_bank")
CLASSIFICATIONS_DIR = os.path.join(_HERE, "classifications")
RESULTS_DIR = os.path.join(_HERE, "results")

TIER1_POOL_PATH = os.path.join(PROBES_DIR, "tier1_pool.json")
TIER2_POOL_PATH = os.path.join(PROBES_DIR, "tier2_pool.json")
TIER3_POOL_PATH = os.path.join(PROBES_DIR, "tier3_pool.json")
SPLIT_INDICES_PATH = os.path.join(PROBES_DIR, "split_indices.json")

TIER1_RAW_PATH = os.path.join(REHEARSAL_DIR, "raw_tier1.json")
TIER2_RAW_PATH = os.path.join(REHEARSAL_DIR, "raw_tier2.json")
TIER3_RAW_PATH = os.path.join(REHEARSAL_DIR, "raw_tier3.json")
TIER1_VERIFIED_PATH = os.path.join(REHEARSAL_DIR, "verified_tier1.json")
TIER2_VERIFIED_PATH = os.path.join(REHEARSAL_DIR, "verified_tier2.json")
TIER3_VERIFIED_PATH = os.path.join(REHEARSAL_DIR, "verified_tier3.json")

ALPACA_TIERS_PATH = os.path.join(CLASSIFICATIONS_DIR, "alpaca_52k_tiers.json")
VALIDATION_SAMPLE_PATH = os.path.join(CLASSIFICATIONS_DIR, "validation_sample.json")

# ---------------------------------------------------------------------------
# lm-eval metric keys — non-obvious cases must be listed explicitly
# ---------------------------------------------------------------------------
METRIC_KEYS = {
    "gsm8k": "exact_match,strict-match",
    "triviaqa": "exact_match,remove_whitespace",
}
METRIC_KEYS_FALLBACK = [
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,remove_whitespace",
]

# ---------------------------------------------------------------------------
# Judge prompt template (shared by classify_training.py and validate_rehearsal.py)
# ---------------------------------------------------------------------------
JUDGE_PROMPT_TEMPLATE = (
    "Given the following instruction and response, classify the cognitive demand "
    "required to correctly answer this instruction.\n\n"
    "Instruction: {instruction}\n"
    "Response: {output}\n\n"
    "Classification criteria:\n"
    "- TIER_1: Requires 3+ sequential reasoning steps where each step depends on "
    "the previous. Includes: multi-step math, symbolic manipulation, chained logical "
    "inference, formula derivation and application, multi-step proofs.\n"
    "- TIER_2: Requires 2 steps of reasoning or paragraph-level comprehension. "
    "Includes: moderate inference, cause-effect reasoning, multi-sentence understanding.\n"
    "- TIER_3: Answerable in a single step via recall, pattern matching, or direct "
    "retrieval. Includes: factual questions, definitions, classifications, formatting "
    "tasks, simple rewrites.\n\n"
    "Respond with exactly one of: TIER_1, TIER_2, TIER_3"
)
