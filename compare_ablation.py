"""Compare vanilla SFT vs adaptive SFT — paired across same seeds.

Loads:
- results/base/                            : base model benchmarks
- results/instruct/seed*-ep4/              : vanilla SFT (existing)
- results/adaptive/seed*-ep4-adaptive/     : adaptive SFT (this update)

Computes paired per-task differences (adaptive_i - vanilla_i) where
seeds match, then aggregates with mean ± SE.
"""

import json, os, glob
import numpy as np

METRIC_KEYS = [
    "acc_norm,none", "acc,none", "exact_match,none",
    "exact_match,strict-match", "exact_match,flexible-extract",
    "exact_match,remove_whitespace",
]

def load_scores_from_dir(results_dir):
    scores = {}
    for root, _, files in os.walk(results_dir):
        for fn in files:
            if fn.startswith("results_") and fn.endswith(".json"):
                with open(os.path.join(root, fn)) as f:
                    data = json.load(f)
                if "results" in data:
                    for task, metrics in data["results"].items():
                        for key in METRIC_KEYS:
                            if key in metrics:
                                scores[task] = round(metrics[key] * 100, 2)
                                break
    return scores

def load_multi(parent, pattern, strip_suffix=""):
    per_seed = {}
    for d in sorted(glob.glob(os.path.join(parent, pattern))):
        if os.path.isdir(d):
            key = os.path.basename(d).replace(strip_suffix, "")
            scores = load_scores_from_dir(d)
            if scores:
                per_seed[key] = scores
    return per_seed

def mean_se(values):
    arr = np.array(values, dtype=float)
    m = float(arr.mean()) if len(arr) else 0.0
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return m, se

base     = load_scores_from_dir("results/base")
vanilla  = load_multi("results/instruct", "seed*-ep4")
adaptive = load_multi("results/adaptive", "seed*-ep4-adaptive", strip_suffix="-adaptive")

print(f"Base: {len(base)} tasks")
print(f"Vanilla seeds:  {list(vanilla.keys())}")
print(f"Adaptive seeds: {list(adaptive.keys())}")

common = sorted(set(vanilla) & set(adaptive))
print(f"Paired seeds: {len(common)} — {common}")
if len(common) < 3:
    print(f"WARNING: fewer than 3 paired seeds; results will be noisier.")

# Top-level table
all_tasks = set().union(*[s.keys() for s in vanilla.values()])
top_tasks = sorted(t for t in all_tasks if not t.startswith("mmlu_") or t == "mmlu")

print(f"\n{'='*120}")
print(f"TOP-LEVEL TASKS: vanilla vs adaptive (paired, n={len(common)})")
print(f"{'='*120}")
print(f"{'Task':<22} {'Base':>7}  {'Vanilla':>18}  {'Adaptive':>18}  {'Δ (paired)':>16}  {'flag':>5}")
print("-" * 120)

rows = []
for task in top_tasks:
    vv = [vanilla[s].get(task)  for s in common if task in vanilla[s]]
    av = [adaptive[s].get(task) for s in common if task in adaptive[s]]
    if not vv or not av: continue

    vm, vse = mean_se(vv)
    am, ase = mean_se(av)
    pairs   = [adaptive[s][task] - vanilla[s][task]
               for s in common if task in vanilla[s] and task in adaptive[s]]
    dm, dse = mean_se(pairs)

    flag = ""
    if dse > 0 and abs(dm) > 2 * dse:
        flag = "★"

    b = base.get(task, 0.0)
    print(f"{task:<22} {b:>7.2f}  {vm:>9.2f} ± {vse:<5.2f}   {am:>9.2f} ± {ase:<5.2f}   {dm:>+7.2f} ± {dse:<5.2f}   {flag:>5}")
    rows.append({"task": task, "base": b,
                 "vanilla_mean": vm, "vanilla_se": vse,
                 "adaptive_mean": am, "adaptive_se": ase,
                 "diff_mean": dm, "diff_se": dse})

# Key MMLU subjects — these are what should improve if the method works
KEY = ["mmlu_abstract_algebra", "mmlu_college_physics", "mmlu_formal_logic",
       "mmlu_college_chemistry", "mmlu_electrical_engineering"]

print(f"\n{'='*120}")
print(f"KEY MMLU SUBJECTS (worst-forgotten in vanilla — adaptive should recover these)")
print(f"{'='*120}")
print(f"{'Subject':<32} {'Base':>7}  {'Vanilla':>18}  {'Adaptive':>18}  {'Δ (paired)':>16}")
print("-" * 120)

for task in KEY:
    vv = [vanilla[s].get(task)  for s in common if task in vanilla[s]]
    av = [adaptive[s].get(task) for s in common if task in adaptive[s]]
    if not vv or not av: continue

    vm, vse = mean_se(vv); am, ase = mean_se(av)
    pairs   = [adaptive[s][task] - vanilla[s][task]
               for s in common if task in vanilla[s] and task in adaptive[s]]
    dm, dse = mean_se(pairs)
    b = base.get(task, 0.0)
    print(f"{task:<32} {b:>7.2f}  {vm:>9.2f} ± {vse:<5.2f}   {am:>9.2f} ± {ase:<5.2f}   {dm:>+7.2f} ± {dse:<5.2f}")

with open("results/ablation_table.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nFull table: results/ablation_table.json")
print(f"\n★ flag = paired Δ exceeds 2 SE (suggestive but not formal significance at n=3)")
