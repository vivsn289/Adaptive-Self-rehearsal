"""Step 5: Cognitive-Tier Adaptive Self-Rehearsal (ASR v2) training loop.

Key improvements over v1:
- Probe pools sourced from held-out benchmark TEST splits (not HF train splits)
- LLM-as-a-judge tier classification replaces keyword-regex domain classifier
- Three cognitive tiers replace five subject-domain buckets
- Cascade rule: Tier 1 trigger injects ALL Tier 1 subtypes (math + logic + symbolic)
- Rehearsal injection is temporary: gradually reduced once recovery is confirmed
- Per-tier forgetting score F_d = (B_d - C_d) / B_d logged at every checkpoint

Environment variables:
  SEED                 : random seed (default 42)
  PROBE_INTERVAL       : gradient steps between probe evals (default 50)
  TIER1_THRESHOLD      : forgetting pp that triggers intervention (default 5.0)
  REHEARSAL_FRACTION   : fraction of batch replaced with rehearsal (default 0.25)
  MODEL_PATH           : base model id (default Qwen/Qwen2.5-3B)
  INSTRUCTIONS_PATH    : path to training data json (default ../generated_data/alpaca_52k.json)
"""

import json
import os
import random
import re
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
    Trainer,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BASE_MODEL, ALPACA_DATA, RESULTS_DIR,
    TIER1_POOL_PATH, TIER2_POOL_PATH, TIER3_POOL_PATH,
    TIER1_VERIFIED_PATH, TIER2_VERIFIED_PATH, TIER3_VERIFIED_PATH,
    ALPACA_TIERS_PATH,
    TIER1_SAMPLE_SIZE, TIER2_SAMPLE_SIZE, TIER3_SAMPLE_SIZE,
    DEFAULT_PROBE_INTERVAL, DEFAULT_TIER1_THRESHOLD, DEFAULT_TIER2_THRESHOLD,
    DEFAULT_REHEARSAL_FRACTION, DEFAULT_REWEIGHT_FACTOR,
    DEFAULT_CONSECUTIVE_DROPS_T1, DEFAULT_CONSECUTIVE_DROPS_T2,
)

# ---------------------------------------------------------------------------
# Configuration from env
# ---------------------------------------------------------------------------

SEED = int(os.environ.get("SEED", 42))
PROBE_INTERVAL = int(os.environ.get("PROBE_INTERVAL", DEFAULT_PROBE_INTERVAL))
TIER1_THRESHOLD = float(os.environ.get("TIER1_THRESHOLD", DEFAULT_TIER1_THRESHOLD))
REHEARSAL_FRACTION = float(os.environ.get("REHEARSAL_FRACTION", DEFAULT_REHEARSAL_FRACTION))
MODEL_PATH = os.environ.get("MODEL_PATH", BASE_MODEL)
INSTRUCTIONS_PATH = os.environ.get("INSTRUCTIONS_PATH", ALPACA_DATA)

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR_OVERRIDE",
    f"./qwen-cognitive-asr-div30-seed{SEED}",
)
PROBE_LOG_PATH = os.path.join(OUTPUT_DIR, "probe_trajectory.jsonl")

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Cognitive ASR v2")
print(f"  SEED={SEED}  PROBE_INTERVAL={PROBE_INTERVAL}")
print(f"  TIER1_THRESHOLD={TIER1_THRESHOLD}pp  REHEARSAL_FRACTION={REHEARSAL_FRACTION}")
print(f"  MODEL_PATH={MODEL_PATH}")
print(f"  OUTPUT_DIR={OUTPUT_DIR}", flush=True)

# ---------------------------------------------------------------------------
# Tokenizer + format
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = 512


def format_text(sample: dict) -> str:
    if sample.get("input", "").strip():
        return (
            "Below is an instruction that describes a task, paired with an input. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Input:\n{sample['input']}\n\n"
            f"### Response:\n{sample['output']}{tokenizer.eos_token}"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{sample['instruction']}\n\n"
        f"### Response:\n{sample['output']}{tokenizer.eos_token}"
    )


def tokenize_one(text: str) -> dict:
    enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    enc["labels"] = [
        -100 if tok == tokenizer.pad_token_id else tok
        for tok in enc["input_ids"]
    ]
    return enc


# ---------------------------------------------------------------------------
# Load and tokenize training data
# ---------------------------------------------------------------------------

print(f"\nLoading training data: {INSTRUCTIONS_PATH}", flush=True)
with open(INSTRUCTIONS_PATH) as f:
    alpaca = json.load(f)
print(f"Loaded {len(alpaca)} training examples", flush=True)

# Load tier classifications for upweighting existing Tier 1 data
tier_labels = ["TIER_3"] * len(alpaca)
if os.path.exists(ALPACA_TIERS_PATH):
    with open(ALPACA_TIERS_PATH) as f:
        classifications = json.load(f)
    for c in classifications:
        idx = c.get("instruction_index", -1)
        if 0 <= idx < len(alpaca):
            tier_labels[idx] = c.get("tier", "TIER_3")
    t1_count = sum(1 for t in tier_labels if t == "TIER_1")
    t2_count = sum(1 for t in tier_labels if t == "TIER_2")
    t3_count = sum(1 for t in tier_labels if t == "TIER_3")
    print(f"Tier distribution in training data:")
    print(f"  TIER_1: {t1_count} ({t1_count/len(alpaca)*100:.1f}%)")
    print(f"  TIER_2: {t2_count} ({t2_count/len(alpaca)*100:.1f}%)")
    print(f"  TIER_3: {t3_count} ({t3_count/len(alpaca)*100:.1f}%)", flush=True)
else:
    print(f"WARNING: {ALPACA_TIERS_PATH} not found — tier upweighting disabled", flush=True)

print("Tokenizing training data...", flush=True)
tokenized_train = [tokenize_one(format_text(s)) for s in alpaca]
print(f"Tokenized {len(tokenized_train)} examples", flush=True)

# ---------------------------------------------------------------------------
# Load rehearsal banks
# ---------------------------------------------------------------------------

def load_rehearsal(path: str, tier_name: str) -> list:
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — Tier {tier_name} rehearsal disabled", flush=True)
        return []
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} verified Tier {tier_name} rehearsal examples", flush=True)
    return data


print("\nLoading rehearsal banks...", flush=True)
rehearsal_t1 = load_rehearsal(TIER1_VERIFIED_PATH, "1")
rehearsal_t2 = load_rehearsal(TIER2_VERIFIED_PATH, "2")
rehearsal_t3 = load_rehearsal(TIER3_VERIFIED_PATH, "3")

tokenized_rehearsal = {
    "TIER_1": [tokenize_one(format_text(s)) for s in rehearsal_t1],
    "TIER_2": [tokenize_one(format_text(s)) for s in rehearsal_t2],
    "TIER_3": [tokenize_one(format_text(s)) for s in rehearsal_t3],
}
print(f"Rehearsal tokenized: T1={len(tokenized_rehearsal['TIER_1'])} "
      f"T2={len(tokenized_rehearsal['TIER_2'])} "
      f"T3={len(tokenized_rehearsal['TIER_3'])}", flush=True)

# ---------------------------------------------------------------------------
# Cognitive Adaptive Dataset
# ---------------------------------------------------------------------------

class CognitiveAdaptiveDataset(TorchDataset):
    """Training dataset with dynamic rehearsal injection.

    When intervention is active, a configurable fraction of each call to
    __getitem__ returns a rehearsal example instead of a training example.
    The dataset also upweights existing Tier 1 training examples via weighted
    random sampling.
    """

    def __init__(self, train_examples, tier_labels, rehearsal_bank, rehearsal_fraction, seed):
        self.train_examples = train_examples
        self.tier_labels = tier_labels
        self.rehearsal_bank = rehearsal_bank  # dict: tier -> list of tokenized
        self.rehearsal_fraction = rehearsal_fraction
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Per-example weights (start uniform; Tier 1 examples upweighted at init)
        self.weights = np.ones(len(train_examples), dtype=np.float64)
        t1_indices = [i for i, t in enumerate(tier_labels) if t == "TIER_1"]
        if t1_indices:
            self.weights[t1_indices] *= DEFAULT_REWEIGHT_FACTOR
            self.weights *= len(self.weights) / self.weights.sum()

        # Intervention state
        self.intervention_active = False
        self.intervention_tiers = set()
        self.current_injection_rate = rehearsal_fraction

    def __len__(self):
        return len(self.train_examples)

    def __getitem__(self, _idx):
        if self.intervention_active and self.intervention_tiers:
            if self.rng.random() < self.current_injection_rate:
                # Pick a tier to rehearse from active intervention tiers
                tier = self.rng.choice(list(self.intervention_tiers))
                bank = self.rehearsal_bank.get(tier, [])
                if bank:
                    return self.rng.choice(bank)

        # Weighted sample from training data
        p = self.weights / self.weights.sum()
        idx = int(self.np_rng.choice(len(self.train_examples), p=p))
        return self.train_examples[idx]

    def set_intervention(self, tiers: set, injection_rate: float):
        self.intervention_active = bool(tiers)
        self.intervention_tiers = tiers
        self.current_injection_rate = injection_rate

    def reduce_injection_rate(self, factor: float = 0.5):
        self.current_injection_rate = max(0.05, self.current_injection_rate * factor)
        if self.current_injection_rate <= 0.05:
            self.intervention_active = False
            self.intervention_tiers = set()


dataset = CognitiveAdaptiveDataset(
    tokenized_train,
    tier_labels,
    tokenized_rehearsal,
    REHEARSAL_FRACTION,
    SEED,
)

# ---------------------------------------------------------------------------
# Probe pools
# ---------------------------------------------------------------------------

def load_probe_pool(path: str, tier: int) -> list:
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — Tier {tier} probes disabled", flush=True)
        return []
    with open(path) as f:
        probes = json.load(f)
    print(f"  Loaded {len(probes)} Tier {tier} probe pool entries", flush=True)
    return probes


print("\nLoading probe pools...", flush=True)
probe_pool = {
    1: load_probe_pool(TIER1_POOL_PATH, 1),
    2: load_probe_pool(TIER2_POOL_PATH, 2),
    3: load_probe_pool(TIER3_POOL_PATH, 3),
}

# ---------------------------------------------------------------------------
# Cognitive Probe Evaluator
# ---------------------------------------------------------------------------

class CognitiveProbeEvaluator:
    """Evaluates model on a random subset of each tier's probe pool.

    Probe sampling uses a separate RNG seeded per (training_seed, step) so
    sampling is reproducible across runs but different at each checkpoint.
    """

    def __init__(self, probe_pool: dict, training_seed: int):
        self.probe_pool = probe_pool
        self.training_seed = training_seed
        self.sample_sizes = {
            1: TIER1_SAMPLE_SIZE,
            2: TIER2_SAMPLE_SIZE,
            3: TIER3_SAMPLE_SIZE,
        }

    def _sample_probes(self, step: int) -> dict:
        samples = {}
        for tier, pool in self.probe_pool.items():
            if not pool:
                samples[tier] = []
                continue
            rng = random.Random(self.training_seed * 100000 + step * 7 + tier)
            n = min(self.sample_sizes[tier], len(pool))
            samples[tier] = rng.sample(pool, n)
        return samples

    @torch.no_grad()
    def evaluate(self, model, step: int) -> dict:
        """Returns {tier: accuracy} and the sampled probe ids."""
        was_training = model.training
        model.eval()

        sampled = self._sample_probes(step)
        results = {}
        probe_ids_used = {}

        for tier, probes in sampled.items():
            if not probes:
                results[tier] = None
                probe_ids_used[tier] = []
                continue

            correct = 0
            ids_used = []
            for probe in probes:
                q = probe.get("question", "")
                choices = probe.get("choices", [])
                answer = probe.get("answer", 0)
                probe_id = probe.get("source_benchmark", "?") + ":" + q[:30]
                ids_used.append(probe_id)

                if not choices:
                    continue

                # Log-likelihood scoring over choices
                scores = []
                for choice in choices:
                    text = q + " " + str(choice)
                    enc = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(next(model.parameters()).device)
                    out = model(**enc, labels=enc["input_ids"])
                    seq_len = enc["input_ids"].shape[1]
                    scores.append(-out.loss.item() * seq_len)
                    del out, enc  # free GPU tensors immediately after scoring
                torch.cuda.empty_cache()

                pred = int(np.argmax(scores))
                if isinstance(answer, (int, float)) and pred == int(answer):
                    correct += 1
                elif isinstance(answer, str):
                    try:
                        if pred == int(answer):
                            correct += 1
                    except ValueError:
                        pass

            results[tier] = correct / len(probes) if probes else None
            probe_ids_used[tier] = ids_used

        if was_training:
            model.train()

        return results, probe_ids_used


evaluator = CognitiveProbeEvaluator(probe_pool, SEED)

# ---------------------------------------------------------------------------
# Cognitive Adaptive Callback
# ---------------------------------------------------------------------------

class CognitiveAdaptiveCallback(TrainerCallback):
    """Tier-aware probe monitoring with cascade rule and gradual recovery."""

    def __init__(self, dataset: CognitiveAdaptiveDataset, evaluator: CognitiveProbeEvaluator,
                 probe_interval: int, t1_threshold: float, t2_threshold: float,
                 reweight_factor: float, consecutive_t1: int, consecutive_t2: int,
                 log_path: str):
        self.dataset = dataset
        self.evaluator = evaluator
        self.probe_interval = probe_interval
        self.t1_threshold = t1_threshold
        self.t2_threshold = t2_threshold
        self.reweight_factor = reweight_factor
        self.consecutive_t1 = consecutive_t1
        self.consecutive_t2 = consecutive_t2
        self.log_path = log_path

        self.baseline_accs = None
        self.consecutive_drops = {1: 0, 2: 0, 3: 0}
        self.consecutive_recoveries = {1: 0, 2: 0}
        self.history = []
        self.total_rehearsal_injected = 0

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def _forgetting_score(self, tier: int, current: float) -> Optional[float]:
        if self.baseline_accs is None or self.baseline_accs.get(tier) is None:
            return None
        b = self.baseline_accs[tier]
        if b == 0:
            return 0.0
        return (b - current) / b * 100  # positive = lost

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step == 0 or step % self.probe_interval != 0:
            return

        t_start = time.time()
        accs, probe_ids = self.evaluator.evaluate(model, step)
        elapsed = time.time() - t_start

        # Set baseline on first evaluation
        if self.baseline_accs is None:
            self.baseline_accs = {k: v for k, v in accs.items() if v is not None}
            _fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
            print(
                f"[CogASR] Step {step}: BASELINE set — "
                f"T1={_fmt(accs.get(1))} T2={_fmt(accs.get(2))} T3={_fmt(accs.get(3))} "
                f"({elapsed:.1f}s)",
                flush=True,
            )
            self._log_record(step, accs, probe_ids, {}, elapsed, "baseline")
            return

        # Compute forgetting scores
        forgetting = {
            tier: self._forgetting_score(tier, acc)
            for tier, acc in accs.items()
            if acc is not None
        }

        interventions = {}
        status_parts = []

        # --- Tier 1 monitoring (trigger on 2 consecutive drops) ---
        t1_acc = accs.get(1)
        t1_f = forgetting.get(1)
        if t1_acc is not None and t1_f is not None:
            if t1_f > self.t1_threshold:
                self.consecutive_drops[1] += 1
                self.consecutive_recoveries[1] = 0
            else:
                self.consecutive_drops[1] = 0
                if self.dataset.intervention_active and "TIER_1" in self.dataset.intervention_tiers:
                    self.consecutive_recoveries[1] += 1

            if self.consecutive_drops[1] >= self.consecutive_t1:
                # CASCADE: all Tier 1 subtypes (math + logic + symbolic) live in TIER_1 bank
                self.dataset.set_intervention({"TIER_1"}, REHEARSAL_FRACTION)
                self.consecutive_drops[1] = 0
                interventions[1] = {
                    "action": "CASCADE_INJECT_ALL_T1",
                    "forgetting_pp": round(t1_f, 2),
                    "injection_rate": self.dataset.current_injection_rate,
                    "consecutive_drops": self.consecutive_t1,
                }
                print(
                    f"[CogASR] Step {step}: *** TIER 1 INTERVENTION *** "
                    f"F={t1_f:.1f}pp — CASCADE injecting all T1 subtypes",
                    flush=True,
                )

            elif self.consecutive_recoveries[1] >= 2 and self.dataset.intervention_active:
                # Recovery confirmed — gradually reduce
                old_rate = self.dataset.current_injection_rate
                self.dataset.reduce_injection_rate(0.5)
                self.consecutive_recoveries[1] = 0
                interventions[1] = {
                    "action": "REDUCE_INJECTION",
                    "old_rate": old_rate,
                    "new_rate": self.dataset.current_injection_rate,
                }
                print(
                    f"[CogASR] Step {step}: Tier 1 recovering — "
                    f"injection rate {old_rate:.2f} → {self.dataset.current_injection_rate:.2f}",
                    flush=True,
                )

            status_parts.append(
                f"T1={t1_acc:.3f}(F={t1_f:+.1f}pp,drops={self.consecutive_drops[1]})"
            )

        # --- Tier 2 monitoring (trigger on 3 consecutive drops, log-only) ---
        t2_acc = accs.get(2)
        t2_f = forgetting.get(2)
        if t2_acc is not None and t2_f is not None:
            if t2_f > self.t2_threshold:
                self.consecutive_drops[2] += 1
            else:
                self.consecutive_drops[2] = 0

            if self.consecutive_drops[2] >= self.consecutive_t2:
                self.dataset.set_intervention(
                    self.dataset.intervention_tiers | {"TIER_2"},
                    self.dataset.current_injection_rate if self.dataset.intervention_active
                    else REHEARSAL_FRACTION * 0.5,
                )
                self.consecutive_drops[2] = 0
                interventions[2] = {
                    "action": "INJECT_T2",
                    "forgetting_pp": round(t2_f, 2),
                }
                print(
                    f"[CogASR] Step {step}: Tier 2 drops {self.consecutive_t2} consecutive "
                    f"(F={t2_f:.1f}pp) — adding T2 rehearsal",
                    flush=True,
                )

            status_parts.append(
                f"T2={t2_acc:.3f}(F={t2_f:+.1f}pp)"
            )

        # --- Tier 3 monitoring (log/warn only) ---
        t3_acc = accs.get(3)
        t3_f = forgetting.get(3)
        if t3_acc is not None and t3_f is not None:
            if t3_f > 8.0:
                print(
                    f"[CogASR] Step {step}: WARNING Tier 3 forgetting {t3_f:.1f}pp "
                    "(sanity check — pattern recognition should be stable)",
                    flush=True,
                )
            status_parts.append(f"T3={t3_acc:.3f}(F={t3_f:+.1f}pp)")

        rehear_status = (
            f"rehearsal={'ON' if self.dataset.intervention_active else 'off'}"
            f"(rate={self.dataset.current_injection_rate:.2f})"
        )
        print(
            f"[CogASR] Step {step}: {' | '.join(status_parts)} | {rehear_status} ({elapsed:.1f}s)",
            flush=True,
        )

        self._log_record(step, accs, probe_ids, interventions, elapsed, "checkpoint")

    def _log_record(self, step, accs, probe_ids, interventions, elapsed, event_type):
        record = {
            "step": step,
            "event_type": event_type,
            "accs": {str(k): (float(v) if v is not None else None) for k, v in accs.items()},
            "forgetting_scores": {
                str(tier): round(self._forgetting_score(tier, acc), 3)
                for tier, acc in accs.items()
                if acc is not None and self.baseline_accs and tier in self.baseline_accs
            },
            "baseline_accs": {str(k): v for k, v in (self.baseline_accs or {}).items()},
            "probe_ids_used": {str(k): v for k, v in probe_ids.items()},
            "interventions": {str(k): v for k, v in interventions.items()},
            "intervention_active": self.dataset.intervention_active,
            "intervention_tiers": list(self.dataset.intervention_tiers),
            "injection_rate": self.dataset.current_injection_rate,
            "consecutive_drops": dict(self.consecutive_drops),
            "elapsed_s": round(elapsed, 1),
        }
        self.history.append(record)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

print(f"\nLoading model: {MODEL_PATH}", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False
print("Model loaded.", flush=True)

# ---------------------------------------------------------------------------
# Training args
# ---------------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    seed=SEED,
    num_train_epochs=1,
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

callback = CognitiveAdaptiveCallback(
    dataset=dataset,
    evaluator=evaluator,
    probe_interval=PROBE_INTERVAL,
    t1_threshold=TIER1_THRESHOLD,
    t2_threshold=DEFAULT_TIER2_THRESHOLD,
    reweight_factor=DEFAULT_REWEIGHT_FACTOR,
    consecutive_t1=DEFAULT_CONSECUTIVE_DROPS_T1,
    consecutive_t2=DEFAULT_CONSECUTIVE_DROPS_T2,
    log_path=PROBE_LOG_PATH,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    callbacks=[callback],
)

# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

resume = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        resume = True
        print(f"RESUMING from checkpoint in {OUTPUT_DIR}", flush=True)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

print(f"\nStarting Cognitive ASR v2 training...\n", flush=True)
trainer.train(resume_from_checkpoint=resume)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

n_interventions = sum(len(r["interventions"]) for r in callback.history)
print(f"\n{'='*60}\nTraining complete.")
print(f"  Model saved: {OUTPUT_DIR}")
print(f"  Probe log: {PROBE_LOG_PATH}")
print(f"  Total probe checkpoints: {len(callback.history)}")
print(f"  Total interventions fired: {n_interventions}")
print(f"  Final injection rate: {dataset.current_injection_rate:.2f}")
print(f"  Intervention active at end: {dataset.intervention_active}")
print(f"{'='*60}", flush=True)
