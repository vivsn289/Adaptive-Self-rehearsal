import json, glob, os
import numpy as np

METRIC_KEYS = [
    "acc_norm,none", "acc,none", "exact_match,none",
    "exact_match,strict-match", "exact_match,flexible-extract",
    "exact_match,remove_whitespace",
]

def load_scores_from_dir(results_dir):
    """Load all task scores from results_*.json files under results_dir (recursive)."""
    scores = {}
    for jf in glob.glob(os.path.join(results_dir, "**", "results_*.json"), recursive=True):
        with open(jf) as f:
            data = json.load(f)
        if "results" in data:
            for task, metrics in data["results"].items():
                for key in METRIC_KEYS:
                    if key in metrics:
                        scores[task] = round(metrics[key] * 100, 2)
                        break
    return scores

def load_multi_seed(parent_dir, pattern="seed*-ep4"):
    """Returns {seed_name: {task: score}} for every matching subdir under parent_dir."""
    per_seed = {}
    for seed_dir in sorted(glob.glob(os.path.join(parent_dir, pattern))):
        if os.path.isdir(seed_dir):
            seed_name = os.path.basename(seed_dir)
            s = load_scores_from_dir(seed_dir)
            if s:
                per_seed[seed_name] = s
    return per_seed

def mean_se(values):
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    m = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return m, se


base = load_scores_from_dir("results/base")
per_seed = load_multi_seed("results/instruct", "seed*-ep4")

if not base:
    print("ERROR: no base results found under results/base/"); exit(1)
if not per_seed:
    print("ERROR: no seed dirs found under results/instruct/ matching 'seed*-ep4'"); exit(1)

seeds = list(per_seed.keys())
n_seeds = len(seeds)
print(f"Base tasks: {len(base)}")
print(f"Seeds loaded ({n_seeds}): {seeds}")
for s in seeds:
    print(f"  {s}: {len(per_seed[s])} tasks")

# Aggregate per-task
all_tasks = set().union(*[s.keys() for s in per_seed.values()])
inst_mean, inst_se, inst_n = {}, {}, {}
for task in all_tasks:
    vals = [per_seed[s][task] for s in seeds if task in per_seed[s]]
    if vals:
        m, se = mean_se(vals)
        inst_mean[task] = m
        inst_se[task] = se
        inst_n[task] = len(vals)

# Top-level (non-MMLU-subject) table
print(f"\n{'='*100}")
print(f"TOP-LEVEL BENCHMARKS (mean ± SE across {n_seeds} seeds)")
print(f"{'='*100}")
print(f"{'Task':<22} {'Base':>7}  {'Instruct (mean±SE)':>22}  {'Δ (mean±SE)':>16}  {'Forgetting%':>14}  n")
print("-" * 100)
for task in sorted(base.keys()):
    if task in inst_mean and not task.startswith("mmlu_"):
        b   = base[task]
        im  = inst_mean[task]
        ise = inst_se[task]
        n   = inst_n[task]
        d   = im - b
        fgt    = (b - im) / b * 100 if b > 0 else 0
        fgt_se = (ise / b * 100)    if b > 0 else 0
        flag = " <<<" if fgt > 5 else (" ++" if fgt < -5 else "")
        print(f"{task:<22} {b:>7.2f}  {im:>9.2f} ± {ise:<6.2f}      {d:>+7.2f} ± {ise:<5.2f}    {fgt:>+6.2f} ± {fgt_se:<4.2f}%  {n}{flag}")

# MMLU per-subject — top forgotten + top improved
mmlu_subj = [t for t in base if t in inst_mean and t.startswith("mmlu_") and t != "mmlu"]
if mmlu_subj:
    rows = []
    for t in mmlu_subj:
        b = base[t]
        if b <= 0:  # skip degenerate
            continue
        im, ise = inst_mean[t], inst_se[t]
        fgt    = (b - im) / b * 100
        fgt_se = ise / b * 100
        rows.append((t, b, im, ise, fgt, fgt_se))
    rows.sort(key=lambda r: r[4], reverse=True)

    def print_mmlu(rows, header):
        print(f"\n{'='*100}\n{header}\n{'='*100}")
        print(f"{'Subject':<32} {'Base':>7}  {'Instruct (mean±SE)':>22}  {'Forgetting%':>16}")
        print("-" * 100)
        for t, b, im, ise, fgt, fgt_se in rows:
            name = t.replace("mmlu_", "").replace("_", " ").title()
            print(f"{name:<32} {b:>7.2f}  {im:>9.2f} ± {ise:<6.2f}      {fgt:>+6.2f} ± {fgt_se:<4.2f}%")

    print_mmlu(rows[:15],  "MMLU TOP 15 FORGOTTEN SUBJECTS")
    print_mmlu(rows[-15:][::-1], "MMLU TOP 15 IMPROVED SUBJECTS")

    print(f"\nMMLU subject spread: {rows[0][4] - rows[-1][4]:.1f}pp")
    print(f"Worst forgotten: {rows[0][0]} ({rows[0][4]:+.1f} ± {rows[0][5]:.1f}%)")
    print(f"Best improved:   {rows[-1][0]} ({rows[-1][4]:+.1f} ± {rows[-1][5]:.1f}%)")

# Category-level aggregation (per-seed first, then across seeds — proper SE)
CATEGORIES = {
    "Factual Recall":    ["triviaqa", "mmlu"],
    "Reasoning":         ["arc_challenge", "arc_easy", "openbookqa", "copa"],
    "Commonsense":       ["hellaswag", "winogrande", "piqa"],
    "Comprehension":     ["boolq", "rte"],
    "Language Modeling": ["lambada_openai"],
    "Math":              ["gsm8k"],
}
cat_per_seed = {s: {} for s in seeds}
for s in seeds:
    sc = per_seed[s]
    for cn, ct in CATEGORIES.items():
        vals = [(base[t] - sc[t]) / base[t] * 100
                for t in ct if t in base and t in sc and base[t] > 0]
        if vals:
            cat_per_seed[s][cn] = sum(vals) / len(vals)

print(f"\n{'='*70}\nCATEGORY-LEVEL FORGETTING (per-seed then aggregated)\n{'='*70}")
print(f"{'Category':<22} {'Mean':>8} {'SE':>6}   Per-seed values")
print("-" * 70)
cat_mean = {}
for cn in CATEGORIES:
    vals = [cat_per_seed[s][cn] for s in seeds if cn in cat_per_seed[s]]
    if vals:
        m, se = mean_se(vals)
        cat_mean[cn] = m
        per_str = "  ".join(f"{v:+.1f}" for v in vals)
        print(f"{cn:<22} {m:>+7.2f}% {se:>5.2f}    [{per_str}]")

# Verdict
print(f"\n{'='*70}\nVERDICT\n{'='*70}")
if cat_mean:
    spread = max(cat_mean.values()) - min(cat_mean.values())
    print(f"Category spread: {spread:.1f}pp")
    if spread > 10:
        print("STRONG domain-specific asymmetry. Adaptive self-rehearsal well-motivated.")
    elif spread > 5:
        print("MODERATE domain-specific asymmetry. Reasonable motivation for adaptive intervention.")
    else:
        print("Uniform shift. Simple data mixing may suffice.")
print("=" * 70)