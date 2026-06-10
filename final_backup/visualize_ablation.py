"""Ablation figures: base / vanilla / adaptive three-way comparison."""

import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
common = sorted(set(vanilla) & set(adaptive))
n = len(common)
print(f"Common seeds: {common}")

os.makedirs("results/figures", exist_ok=True)

# -----------------------------------------------------------------------------
# Figure 1: top-level 3-way (base / vanilla / adaptive)
# -----------------------------------------------------------------------------
TOP = ["arc_easy", "arc_challenge", "boolq", "rte", "copa", "gsm8k",
       "lambada_openai", "piqa", "hellaswag", "winogrande", "openbookqa",
       "triviaqa", "mmlu"]
TOP = [t for t in TOP if t in base]

bv = [base[t] for t in TOP]
vm = []; vse = []; am = []; ase = []
for t in TOP:
    m, s = mean_se([vanilla[k][t]  for k in common if t in vanilla[k]]);  vm.append(m); vse.append(s)
    m, s = mean_se([adaptive[k][t] for k in common if t in adaptive[k]]); am.append(m); ase.append(s)

fig, ax = plt.subplots(figsize=(15, 6))
x = np.arange(len(TOP)); w = 0.27
ax.bar(x - w, bv, w, label="Base",            color="#3274A1")
ax.bar(x,     vm, w, yerr=vse, label=f"Vanilla SFT (n={n})",  color="#E1812C",
       error_kw=dict(ecolor='black', capsize=3))
ax.bar(x + w, am, w, yerr=ase, label=f"Adaptive SFT (n={n})", color="#3A923A",
       error_kw=dict(ecolor='black', capsize=3))
ax.set_xticks(x)
ax.set_xticklabels([t.replace("_", " ").title() for t in TOP], rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Ablation: Base vs Vanilla SFT vs Adaptive SFT (mean ± SE, n={n})", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("results/figures/ablation_toplevel.png", dpi=150); plt.close()
print("Saved: ablation_toplevel.png")

# -----------------------------------------------------------------------------
# Figure 2: paired (adaptive - vanilla) per task
# -----------------------------------------------------------------------------
diffs = []; ds = []
for t in TOP:
    pairs = [adaptive[k][t] - vanilla[k][t]
             for k in common if t in vanilla[k] and t in adaptive[k]]
    m, s = mean_se(pairs); diffs.append(m); ds.append(s)

fig, ax = plt.subplots(figsize=(14, 5))
colors = ["#27AE60" if d > 0 else "#E74C3C" for d in diffs]
ax.bar(range(len(TOP)), diffs, yerr=ds, color=colors,
       error_kw=dict(ecolor='black', capsize=3))
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(TOP)))
ax.set_xticklabels([t.replace("_", " ").title() for t in TOP], rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Δ Accuracy (Adaptive − Vanilla, pp)")
ax.set_title(f"Paired Improvement from Adaptive Mechanism (n={n} seeds)", fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("results/figures/ablation_paired_diff.png", dpi=150); plt.close()
print("Saved: ablation_paired_diff.png")

# -----------------------------------------------------------------------------
# Figure 3: key MMLU subjects 3-way
# -----------------------------------------------------------------------------
KEY = ["mmlu_abstract_algebra", "mmlu_college_physics", "mmlu_formal_logic",
       "mmlu_college_chemistry", "mmlu_electrical_engineering"]
KEY = [t for t in KEY if t in base]

bv = [base[t] for t in KEY]
vm = []; vse = []; am = []; ase = []
for t in KEY:
    m, s = mean_se([vanilla[k][t]  for k in common if t in vanilla[k]]);  vm.append(m); vse.append(s)
    m, s = mean_se([adaptive[k][t] for k in common if t in adaptive[k]]); am.append(m); ase.append(s)

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(KEY)); w = 0.27
ax.bar(x - w, bv, w, label="Base",            color="#3274A1")
ax.bar(x,     vm, w, yerr=vse, label="Vanilla SFT", color="#E1812C",
       error_kw=dict(ecolor='black', capsize=3))
ax.bar(x + w, am, w, yerr=ase, label="Adaptive SFT", color="#3A923A",
       error_kw=dict(ecolor='black', capsize=3))
ax.set_xticks(x)
ax.set_xticklabels([t.replace("mmlu_", "").replace("_", " ").title() for t in KEY],
                   rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Recovery of Worst-Forgotten MMLU Subjects (Vanilla → Adaptive)", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("results/figures/ablation_key_subjects.png", dpi=150); plt.close()
print("Saved: ablation_key_subjects.png")

# -----------------------------------------------------------------------------
# Figure 4: probe trajectory during training (uses any one seed)
# -----------------------------------------------------------------------------
traj_files = sorted(glob.glob("results/adaptive/*/probe_trajectory.json"))
if not traj_files:
    traj_files = sorted(glob.glob("qwen-sft-seed*-ep*-adaptive/probe_trajectory.json"))

if traj_files:
    with open(traj_files[0]) as f:
        history = json.load(f)

    steps = [h["step"] for h in history]
    domains = list(history[0]["accs"].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(domains)))

    fig, ax = plt.subplots(figsize=(13, 6))
    for d, c in zip(domains, colors):
        ys = [h["accs"][d] * 100 for h in history]
        ax.plot(steps, ys, marker='o', label=d, color=c, lw=2)

    for h in history:
        for itv in h.get("interventions", []):
            ax.axvline(h["step"], color=colors[domains.index(itv["domain"])],
                       ls="--", alpha=0.4, lw=1)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Probe Accuracy (%)")
    ax.set_title(f"Per-Domain Probe Accuracy During Adaptive Training\n"
                 f"(dashed lines mark interventions; source: {os.path.basename(os.path.dirname(traj_files[0]))})",
                 fontweight="bold")
    ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("results/figures/ablation_probe_trajectory.png", dpi=150); plt.close()
    print("Saved: ablation_probe_trajectory.png")
else:
    print("(no probe trajectory found; skipping trajectory figure)")

print("\nDone.")
