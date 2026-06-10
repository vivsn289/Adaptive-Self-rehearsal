"""Step 6: Post-training comparison — ASR v2 vs vanilla 52K baseline vs base model.

Handles all lm-eval gotchas:
- Dot-prefixed result dirs (model path starting with ./) — uses os.walk
- Timestamped filenames results_2026-*.json — globs for results_*.json
- Non-standard metric keys (GSM8K, TriviaQA) from config.METRIC_KEYS

Key question: Does ASR v2 preserve Tier 1 capability (GSM8K, Abstract Algebra,
College Physics, Formal Logic) while maintaining instruction-following gains?

Also visualizes the probe trajectory log from training to show when interventions
triggered and whether recovery occurred.
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BASE_RESULTS, VANILLA_52K_RESULTS, RESULTS_DIR,
    METRIC_KEYS, METRIC_KEYS_FALLBACK,
)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Score loading — handles dot-prefixed dirs and timestamped filenames
# ---------------------------------------------------------------------------

def find_result_files(root_dir: str) -> list:
    """Walk the directory tree to find all results_*.json files.

    os.walk is used instead of glob to handle dot-prefixed subdirectories
    that lm-eval creates when the model path starts with './'.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.startswith("results_") and fname.endswith(".json"):
                found.append(os.path.join(dirpath, fname))
    return found


def extract_score(metrics: dict, task: str) -> float | None:
    """Extract the primary metric for a task from lm-eval metrics dict."""
    # Check task-specific key first
    if task in METRIC_KEYS:
        key = METRIC_KEYS[task]
        if key in metrics:
            return round(metrics[key] * 100, 2)
    # Fallback order
    for key in METRIC_KEYS_FALLBACK:
        if key in metrics:
            return round(metrics[key] * 100, 2)
    return None


def load_scores_from_dir(results_dir: str) -> dict:
    """Load all task scores from a single experiment directory."""
    scores = {}
    result_files = find_result_files(results_dir)
    if not result_files:
        print(f"  WARNING: no results_*.json found under {results_dir}", flush=True)
    for jf in result_files:
        try:
            with open(jf) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  WARNING: could not parse {jf}: {e}", flush=True)
            continue
        if "results" not in data:
            continue
        for task, metrics in data["results"].items():
            score = extract_score(metrics, task)
            if score is not None:
                scores[task] = score
    return scores


def load_multi_seed(parent_dir: str, pattern: str = "seed*-ep1") -> dict:
    """Returns {seed_name: {task: score}} for all matching seed subdirs."""
    per_seed = {}
    for seed_dir in sorted(glob.glob(os.path.join(parent_dir, pattern))):
        if os.path.isdir(seed_dir):
            name = os.path.basename(seed_dir)
            s = load_scores_from_dir(seed_dir)
            if s:
                per_seed[name] = s
    return per_seed


def mean_se(values: list) -> tuple:
    arr = np.array(values, dtype=float)
    m = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return m, se


# ---------------------------------------------------------------------------
# Load all results
# ---------------------------------------------------------------------------

print("Loading results...", flush=True)

base = load_scores_from_dir(BASE_RESULTS)
print(f"  Base model: {len(base)} tasks")

vanilla_seeds = load_multi_seed(VANILLA_52K_RESULTS, pattern="seed*-ep1")
if not vanilla_seeds:
    # Also try ep4 from old experiments
    vanilla_seeds = load_multi_seed(VANILLA_52K_RESULTS, pattern="seed*-ep4")
print(f"  Vanilla 52K: {len(vanilla_seeds)} seeds — {list(vanilla_seeds.keys())}")

# ASR v2 results — look for seed dirs under cognitive_asr/results/
asr_results_dir = RESULTS_DIR
asr_seeds = load_multi_seed(asr_results_dir, pattern="seed*")
if not asr_seeds:
    # Fallback: look for qwen-cognitive-asr-seed* in parent
    parent = os.path.dirname(os.path.dirname(RESULTS_DIR))
    asr_dirs = glob.glob(os.path.join(parent, "qwen-cognitive-asr-seed*"))
    for d in asr_dirs:
        if os.path.isdir(d):
            name = os.path.basename(d)
            s = load_scores_from_dir(d)
            if s:
                asr_seeds[name] = s
print(f"  ASR v2: {len(asr_seeds)} seeds — {list(asr_seeds.keys())}")

if not base:
    print("ERROR: no base model results found. Check BASE_RESULTS path."); sys.exit(1)
if not vanilla_seeds:
    print("WARNING: no vanilla 52K results found — comparison will be incomplete.")
if not asr_seeds:
    print("WARNING: no ASR v2 results found — run benchmark_instruct.sh after training "
          "and copy results into cognitive_asr/results/seed{SEED}/ or qwen-cognitive-asr-seed*/")


# ---------------------------------------------------------------------------
# Aggregate per-task scores
# ---------------------------------------------------------------------------

def aggregate(seed_dict: dict) -> tuple[dict, dict]:
    """Returns (mean_scores, se_scores) across seeds."""
    all_tasks = set().union(*[s.keys() for s in seed_dict.values()])
    means, ses = {}, {}
    for task in all_tasks:
        vals = [seed_dict[s][task] for s in seed_dict if task in seed_dict[s]]
        if vals:
            m, se = mean_se(vals)
            means[task] = m
            ses[task] = se
    return means, ses


vanilla_mean, vanilla_se = aggregate(vanilla_seeds) if vanilla_seeds else ({}, {})
asr_mean, asr_se = aggregate(asr_seeds) if asr_seeds else ({}, {})


# ---------------------------------------------------------------------------
# Cognitive axis groupings
# ---------------------------------------------------------------------------

COGNITIVE_AXIS = {
    "Formal Derivation (T1)": [
        "gsm8k", "mmlu_abstract_algebra", "mmlu_college_physics",
        "mmlu_formal_logic", "mmlu_college_chemistry",
    ],
    "Commonsense (T2)": ["hellaswag", "winogrande", "piqa", "copa"],
    "Pattern Recognition (T3)": ["arc_easy", "arc_challenge", "boolq", "rte", "openbookqa"],
    "Factual Recall": ["triviaqa", "mmlu"],
    "Language Modeling": ["lambada_openai"],
}


# ---------------------------------------------------------------------------
# Print comparison tables
# ---------------------------------------------------------------------------

def disp(t: str) -> str:
    return t.replace("lambada_openai", "LAMBADA").replace("_", " ").title()


top_tasks = sorted(
    [t for t in base if not t.startswith("mmlu_") and t != "mmlu"]
)
mmlu_subj = sorted(
    [t for t in base if t.startswith("mmlu_") and t != "mmlu"]
)

print(f"\n{'='*110}")
print(f"TOP-LEVEL BENCHMARK COMPARISON")
print(f"{'='*110}")
header = f"{'Task':<26}  {'Base':>8}  {'Vanilla 52K':>13}  {'ΔVan':>8}  {'ASR v2':>13}  {'ΔASR':>8}  {'T1 benefit?':>12}"
print(header)
print("-" * 110)

for task in top_tasks:
    b = base.get(task)
    if b is None:
        continue
    van_m = vanilla_mean.get(task)
    asr_m = asr_mean.get(task)
    van_str = f"{van_m:.2f}±{vanilla_se.get(task, 0):.2f}" if van_m is not None else "  —   "
    asr_str = f"{asr_m:.2f}±{asr_se.get(task, 0):.2f}" if asr_m is not None else "  —   "
    d_van = f"{van_m - b:+.2f}" if van_m is not None else "  —"
    d_asr = f"{asr_m - b:+.2f}" if asr_m is not None else "  —"

    # Flag if ASR improves over vanilla on T1 tasks
    tier1_tasks = COGNITIVE_AXIS["Formal Derivation (T1)"]
    benefit = ""
    if task in tier1_tasks and asr_m is not None and van_m is not None:
        if asr_m > van_m + 0.5:
            benefit = "YES <<"
        elif asr_m < van_m - 0.5:
            benefit = "WORSE"
        else:
            benefit = "~same"

    print(f"{task:<26}  {b:>8.2f}  {van_str:>13}  {d_van:>8}  {asr_str:>13}  {d_asr:>8}  {benefit:>12}")


# Cognitive axis summary
print(f"\n{'='*80}")
print("COGNITIVE AXIS SUMMARY (forgetting scores: positive = lost)")
print(f"{'='*80}")
print(f"{'Category':<28}  {'VanillaF%':>10}  {'ASRF%':>10}  {'ASR better?':>12}")
print("-" * 80)

for cat_name, cat_tasks in COGNITIVE_AXIS.items():
    van_vals = [
        (base[t] - vanilla_mean[t]) / base[t] * 100
        for t in cat_tasks
        if t in base and t in vanilla_mean and base[t] > 0
    ]
    asr_vals = [
        (base[t] - asr_mean[t]) / base[t] * 100
        for t in cat_tasks
        if t in base and t in asr_mean and base[t] > 0
    ]
    van_f = sum(van_vals) / len(van_vals) if van_vals else None
    asr_f = sum(asr_vals) / len(asr_vals) if asr_vals else None

    van_str = f"{van_f:+.2f}%" if van_f is not None else "    —"
    asr_str = f"{asr_f:+.2f}%" if asr_f is not None else "    —"
    better = ""
    if van_f is not None and asr_f is not None:
        diff = van_f - asr_f
        if diff > 2:
            better = f"YES (+{diff:.1f}pp)"
        elif diff < -2:
            better = f"NO (-{abs(diff):.1f}pp)"
        else:
            better = "~same"
    print(f"{cat_name:<28}  {van_str:>10}  {asr_str:>10}  {better:>12}")


# MMLU per-subject
if mmlu_subj and (vanilla_mean or asr_mean):
    rows = []
    for t in mmlu_subj:
        b = base.get(t)
        if not b or b == 0:
            continue
        van_m = vanilla_mean.get(t)
        asr_m = asr_mean.get(t)
        rows.append((t, b, van_m, asr_m))
    rows.sort(key=lambda r: ((r[1] - r[2]) / r[1] * 100 if r[2] is not None else 0), reverse=True)

    print(f"\n{'='*80}")
    print("MMLU TOP 10 MOST FORGOTTEN (vanilla) — Did ASR v2 help?")
    print(f"{'='*80}")
    for t, b, van_m, asr_m in rows[:10]:
        name = t.replace("mmlu_", "").replace("_", " ").title()
        van_f = (b - van_m) / b * 100 if van_m is not None else None
        asr_f = (b - asr_m) / b * 100 if asr_m is not None else None
        print(
            f"  {name:<30}  base={b:.1f}  "
            f"van_F={van_f:+.1f}%  asr_F={asr_f:+.1f}%"
            if (van_f is not None and asr_f is not None)
            else f"  {name:<30}  base={b:.1f}  (incomplete data)"
        )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

fig_dir = os.path.join(RESULTS_DIR, "figures")
os.makedirs(fig_dir, exist_ok=True)


def _disp(t):
    return t.replace("lambada_openai", "LAMBADA").replace("_", " ").title()


# FIG 1: Three-way comparison (base / vanilla / ASR) on top-level tasks
tasks_to_plot = [t for t in top_tasks if t in base and (t in vanilla_mean or t in asr_mean)]
if tasks_to_plot:
    labels = [_disp(t) for t in tasks_to_plot]
    bv = [base[t] for t in tasks_to_plot]
    vv = [vanilla_mean.get(t, 0) for t in tasks_to_plot]
    av = [asr_mean.get(t, 0) for t in tasks_to_plot]

    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(tasks_to_plot))
    w = 0.28
    ax.bar(x - w, bv, w, label="Base Qwen2.5-3B", color="#4878CF")
    ax.bar(x, vv, w, label="Vanilla SFT 52K", color="#E1812C")
    ax.bar(x + w, av, w, label="Cognitive ASR v2", color="#6ACC65")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Base vs Vanilla SFT vs Cognitive ASR v2", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(fig_dir, "three_way_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved: {out}")


# FIG 2: Forgetting delta — vanilla vs ASR on each task
tasks_both = [
    t for t in top_tasks
    if t in base and t in vanilla_mean and t in asr_mean and base[t] > 0
]
if tasks_both:
    van_f = [(base[t] - vanilla_mean[t]) / base[t] * 100 for t in tasks_both]
    asr_f = [(base[t] - asr_mean[t]) / base[t] * 100 for t in tasks_both]
    labels = [_disp(t) for t in tasks_both]
    improvement = [v - a for v, a in zip(van_f, asr_f)]  # positive = ASR reduced forgetting

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: side-by-side forgetting
    ax = axes[0]
    x = np.arange(len(tasks_both))
    w = 0.38
    ax.bar(x - w / 2, van_f, w, label="Vanilla SFT", color="#E1812C", alpha=0.85)
    ax.bar(x + w / 2, asr_f, w, label="ASR v2", color="#6ACC65", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(5, color="#E74C3C", lw=0.8, ls="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Forgetting Score (%) — positive = lost")
    ax.set_title("Forgetting: Vanilla vs ASR v2", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: improvement (van_f - asr_f)
    ax = axes[1]
    colors = ["#27AE60" if v > 0 else "#E74C3C" for v in improvement]
    ax.bar(labels, improvement, color=colors, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Forgetting reduction (pp) — positive = ASR v2 better")
    ax.set_title("ASR v2 benefit over vanilla SFT", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(improvement):
        ax.text(i, v + (0.3 if v >= 0 else -0.3), f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(fig_dir, "forgetting_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# FIG 3: Probe trajectory (intervention timeline)
probe_log_candidates = []
parent = os.path.dirname(os.path.dirname(RESULTS_DIR))
for dirpath, dirnames, filenames in os.walk(parent):
    for fname in filenames:
        if fname == "probe_trajectory.jsonl":
            probe_log_candidates.append(os.path.join(dirpath, fname))

for log_path in probe_log_candidates:
    try:
        records = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            continue

        steps = [r["step"] for r in records]
        t1_accs = [r["accs"].get("1") for r in records]
        t2_accs = [r["accs"].get("2") for r in records]
        t3_accs = [r["accs"].get("3") for r in records]
        intervention_steps = [
            r["step"] for r in records if r.get("interventions")
        ]

        fig, ax = plt.subplots(figsize=(14, 5))
        if any(v is not None for v in t1_accs):
            ax.plot(steps, [v if v is not None else float("nan") for v in t1_accs],
                    "o-", color="#E74C3C", label="Tier 1 (derivation)", markersize=4)
        if any(v is not None for v in t2_accs):
            ax.plot(steps, [v if v is not None else float("nan") for v in t2_accs],
                    "s-", color="#F39C12", label="Tier 2 (reasoning)", markersize=4)
        if any(v is not None for v in t3_accs):
            ax.plot(steps, [v if v is not None else float("nan") for v in t3_accs],
                    "^-", color="#27AE60", label="Tier 3 (pattern)", markersize=4)

        for s in intervention_steps:
            ax.axvline(s, color="purple", lw=1.2, ls="--", alpha=0.5)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Probe Accuracy")
        ax.set_title("Cognitive Tier Probe Accuracy During Training\n"
                     "(dashed purple = intervention triggered)", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        seed_tag = os.path.basename(os.path.dirname(log_path))
        out = os.path.join(fig_dir, f"probe_trajectory_{seed_tag}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")

        n_interventions = len(intervention_steps)
        print(f"  Probe log: {len(records)} checkpoints, {n_interventions} interventions")
    except Exception as e:
        print(f"  Could not parse probe log {log_path}: {e}")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

print(f"\n{'='*60}\nVERDICT\n{'='*60}")

tier1_tasks_in_base = [
    t for t in COGNITIVE_AXIS["Formal Derivation (T1)"]
    if t in base and t in vanilla_mean and t in asr_mean
]
if tier1_tasks_in_base:
    van_t1_f = [
        (base[t] - vanilla_mean[t]) / base[t] * 100
        for t in tier1_tasks_in_base if base[t] > 0
    ]
    asr_t1_f = [
        (base[t] - asr_mean[t]) / base[t] * 100
        for t in tier1_tasks_in_base if base[t] > 0
    ]
    if van_t1_f and asr_t1_f:
        van_avg = sum(van_t1_f) / len(van_t1_f)
        asr_avg = sum(asr_t1_f) / len(asr_t1_f)
        improvement = van_avg - asr_avg
        print(f"Tier 1 (derivation) — avg forgetting:")
        print(f"  Vanilla SFT:   {van_avg:+.2f}%")
        print(f"  Cognitive ASR: {asr_avg:+.2f}%")
        print(f"  Improvement:   {improvement:+.2f}pp")
        if improvement > 3:
            print("=> ASR v2 SUCCESSFULLY reduces Tier 1 forgetting.")
        elif improvement > 0:
            print("=> ASR v2 shows modest Tier 1 improvement.")
        else:
            print("=> ASR v2 did NOT improve Tier 1 forgetting — revisit intervention threshold.")
else:
    print("Insufficient data for Tier 1 verdict — run benchmarks first.")

print(f"{'='*60}")
print(f"\nFigures saved to: {fig_dir}")
