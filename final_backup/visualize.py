import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

METRIC_KEYS = [
    "acc_norm,none", "acc,none", "exact_match,none",
    "exact_match,strict-match", "exact_match,flexible-extract",
    "exact_match,remove_whitespace",
]

def load_scores_from_dir(results_dir):
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
    per_seed = {}
    for seed_dir in sorted(glob.glob(os.path.join(parent_dir, pattern))):
        if os.path.isdir(seed_dir):
            s = load_scores_from_dir(seed_dir)
            if s:
                per_seed[os.path.basename(seed_dir)] = s
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
seeds = list(per_seed.keys())
n_seeds = len(seeds)
assert base and n_seeds > 0, "No base or seed results found"
print(f"Base: {len(base)} tasks | Seeds ({n_seeds}): {seeds}")

# Aggregate per-task
all_tasks = set().union(*[s.keys() for s in per_seed.values()])
inst_mean, inst_se = {}, {}
for task in all_tasks:
    vals = [per_seed[s][task] for s in seeds if task in per_seed[s]]
    if vals:
        m, se = mean_se(vals)
        inst_mean[task] = m
        inst_se[task] = se

CATEGORIES = {
    "Factual Recall":    ["triviaqa", "mmlu"],
    "Reasoning":         ["arc_challenge", "arc_easy", "openbookqa", "copa"],
    "Commonsense":       ["hellaswag", "winogrande", "piqa"],
    "Comprehension":     ["boolq", "rte"],
    "Language Modeling": ["lambada_openai"],
    "Math":              ["gsm8k"],
}

top_tasks  = [t for t in sorted(base) if t in inst_mean and not t.startswith("mmlu_")]
mmlu_tasks = [t for t in sorted(base) if t in inst_mean and t.startswith("mmlu_") and t != "mmlu"]
os.makedirs("results/figures", exist_ok=True)

def disp(t):
    return t.replace("lambada_openai", "LAMBADA").replace("_", " ").title()

# ---------- FIG 1: Side-by-side comparison with error bars ----------
labels = [disp(t) for t in top_tasks]
bv  = [base[t]      for t in top_tasks]
iv  = [inst_mean[t] for t in top_tasks]
ise = [inst_se[t]   for t in top_tasks]

fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(top_tasks)); w = 0.35
b1 = ax.bar(x - w/2, bv, w, label="Base (Qwen2.5-3B)", color="#3274A1")
b2 = ax.bar(x + w/2, iv, w, yerr=ise, label=f"Self-Instruct SFT (n={n_seeds})",
            color="#E1812C", error_kw=dict(ecolor='black', capsize=3, lw=1))
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Base vs Self-Instruct Fine-Tuned: Per-Benchmark (mean ± SE, n={n_seeds} seeds)", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
ax.legend(); ax.grid(axis="y", alpha=0.3)
for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
for bar, se in zip(b2, ise):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+se+0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout(); plt.savefig("results/figures/comparison.png", dpi=150); plt.close()
print("Saved: comparison.png")

# ---------- FIG 2: Forgetting scores with error bars ----------
fgt    = [(base[t] - inst_mean[t]) / base[t] * 100 if base[t] > 0 else 0 for t in top_tasks]
fgt_se = [(inst_se[t] / base[t] * 100)             if base[t] > 0 else 0 for t in top_tasks]

fig, ax = plt.subplots(figsize=(11, 6))
colors = ["#E74C3C" if f > 5 else ("#27AE60" if f < -5 else "#95A5A6") for f in fgt]
bars = ax.barh(labels, fgt, xerr=fgt_se, color=colors, edgecolor="white",
               error_kw=dict(ecolor='black', capsize=3, lw=1))
ax.set_xlabel(f"Forgetting Score (%) — Positive = Lost | Negative = Gained (mean ± SE, n={n_seeds})")
ax.set_title("Per-Domain Forgetting After Self-Instruct Fine-Tuning", fontweight="bold")
ax.axvline(0, color="black", lw=0.8)
ax.axvline(5, color="#E74C3C", lw=0.8, ls="--", alpha=0.4)
ax.axvline(-5, color="#27AE60", lw=0.8, ls="--", alpha=0.4)
for bar, v, se in zip(bars, fgt, fgt_se):
    width = bar.get_width()
    xp = width + se + 0.3 if width >= 0 else width - se - 0.3
    ax.text(xp, bar.get_y()+bar.get_height()/2, f"{v:+.1f}±{se:.1f}%",
            ha="left" if v >= 0 else "right", va="center", fontsize=8)
ax.legend(handles=[
    mpatches.Patch(color="#E74C3C", label="Forgetting (>5%)"),
    mpatches.Patch(color="#95A5A6", label="Stable (±5%)"),
    mpatches.Patch(color="#27AE60", label="Improvement (>5%)"),
], loc="lower right")
plt.tight_layout(); plt.savefig("results/figures/forgetting.png", dpi=150); plt.close()
print("Saved: forgetting.png")

# ---------- FIG 3: Delta waterfall with error bars ----------
deltas   = [inst_mean[t] - base[t] for t in top_tasks]
delta_se = [inst_se[t]             for t in top_tasks]

fig, ax = plt.subplots(figsize=(14, 5))
colors = ["#27AE60" if d >= 0 else "#E74C3C" for d in deltas]
ax.bar(labels, deltas, yerr=delta_se, color=colors, edgecolor="white",
       error_kw=dict(ecolor='black', capsize=3, lw=1))
ax.set_ylabel("Accuracy Change (pp)")
ax.set_title(f"Impact of Self-Instruct Fine-Tuning: Per-Benchmark Δ (mean ± SE, n={n_seeds})", fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
for i, (d, se) in enumerate(zip(deltas, delta_se)):
    off = se + 0.3 if d >= 0 else -(se + 0.3)
    ax.text(i, d + off, f"{d:+.1f}", ha="center",
            va="bottom" if d >= 0 else "top", fontsize=8, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("results/figures/delta_waterfall.png", dpi=150); plt.close()
print("Saved: delta_waterfall.png")

# ---------- FIG 4: MMLU per-subject with error bars ----------
if mmlu_tasks:
    mf = []
    for t in mmlu_tasks:
        if base[t] > 0:
            f    = (base[t] - inst_mean[t]) / base[t] * 100
            f_se = inst_se[t] / base[t] * 100
            mf.append((t, f, f_se))
    mf.sort(key=lambda x: x[1], reverse=True)
    sel = mf[:15] + mf[-15:]
    lm  = [t[0].replace("mmlu_", "").replace("_", " ").title() for t in sel]
    vm  = [t[1] for t in sel]
    sm  = [t[2] for t in sel]
    cm  = ["#E74C3C" if v > 5 else ("#27AE60" if v < -5 else "#95A5A6") for v in vm]
    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(lm, vm, xerr=sm, color=cm, edgecolor="white",
                   error_kw=dict(ecolor='black', capsize=2, lw=1))
    ax.set_xlabel(f"Forgetting Score (%) (mean ± SE, n={n_seeds})")
    ax.set_title("MMLU Per-Subject: Top 15 Forgotten + Top 15 Improved", fontweight="bold")
    ax.axvline(0, color="black", lw=0.8)
    for bar, v, se in zip(bars, vm, sm):
        width = bar.get_width()
        xp = width + se + 0.5 if width >= 0 else width - se - 0.5
        ax.text(xp, bar.get_y()+bar.get_height()/2, f"{v:+.1f}±{se:.1f}%",
                ha="left" if v >= 0 else "right", va="center", fontsize=7)
    plt.tight_layout(); plt.savefig("results/figures/mmlu_subject_forgetting.png", dpi=150); plt.close()
    print("Saved: mmlu_subject_forgetting.png")
    print(f"MMLU spread: {mf[0][1]-mf[-1][1]:.1f}pp | Worst: {mf[0][0]} ({mf[0][1]:+.1f}±{mf[0][2]:.1f}%) | Best: {mf[-1][0]} ({mf[-1][1]:+.1f}±{mf[-1][2]:.1f}%)")

# ---------- FIG 5: Category-level (per-seed first, then across seeds) ----------
cat_per_seed = {s: {} for s in seeds}
for s in seeds:
    sc = per_seed[s]
    for cn, ct in CATEGORIES.items():
        vals = [(base[t] - sc[t]) / base[t] * 100
                for t in ct if t in base and t in sc and base[t] > 0]
        if vals:
            cat_per_seed[s][cn] = sum(vals) / len(vals)

cat_mean, cat_se = {}, {}
for cn in CATEGORIES:
    vals = [cat_per_seed[s][cn] for s in seeds if cn in cat_per_seed[s]]
    if vals:
        m, se = mean_se(vals)
        cat_mean[cn], cat_se[cn] = m, se

if cat_mean:
    cn_list = list(cat_mean.keys())
    cv  = [cat_mean[c] for c in cn_list]
    cse = [cat_se[c]   for c in cn_list]
    cc  = ["#E74C3C" if v > 5 else ("#27AE60" if v < -5 else "#95A5A6") for v in cv]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cn_list, cv, xerr=cse, color=cc, edgecolor="white", height=0.6,
                   error_kw=dict(ecolor='black', capsize=3, lw=1))
    ax.set_xlabel(f"Average Forgetting Score (%) (mean ± SE, n={n_seeds})")
    ax.set_title("Forgetting by Capability Category", fontweight="bold")
    ax.axvline(0, color="black", lw=0.8)
    for bar, v, se in zip(bars, cv, cse):
        width = bar.get_width()
        xp = width + se + 0.3 if width >= 0 else width - se - 0.3
        ax.text(xp, bar.get_y()+bar.get_height()/2, f"{v:+.1f}±{se:.1f}%",
                ha="left" if v >= 0 else "right", va="center", fontsize=10, fontweight="bold")
    plt.tight_layout(); plt.savefig("results/figures/category_forgetting.png", dpi=150); plt.close()
    print("Saved: category_forgetting.png")

# ---------- FIG 6 (NEW): Cross-seed consistency for the worst-forgotten subjects ----------
# This is the chart that answers "does forgetting hold across all 3 seeds?"
key_subjects = []
if mmlu_tasks:
    ranked = sorted(
        [(t, inst_mean[t], base[t]) for t in mmlu_tasks if t in inst_mean and base[t] > 0],
        key=lambda x: (x[2] - x[1]) / x[2], reverse=True
    )
    key_subjects = [t[0] for t in ranked[:6]]
if "gsm8k" in inst_mean:
    key_subjects.append("gsm8k")

if key_subjects:
    fig, ax = plt.subplots(figsize=(13, 6))
    palette = ["#3274A1", "#E1812C", "#27AE60", "#9467BD", "#8C564B"]
    n_show = len(key_subjects)
    bar_w = 0.18
    x_pos = np.arange(n_show)
    sub_labels = [s.replace("mmlu_", "").replace("_", " ").title() for s in key_subjects]

    # Base value as a black horizontal tick
    base_vals = [base[s] for s in key_subjects]
    ax.scatter(x_pos, base_vals, marker="_", s=400, color="black", linewidths=3,
               zorder=5, label="Base")

    # Each seed as a colored dot, offset slightly
    for i, seed in enumerate(seeds):
        seed_vals = [per_seed[seed].get(s, np.nan) for s in key_subjects]
        offsets = (i - (n_seeds - 1) / 2) * bar_w
        ax.scatter(x_pos + offsets, seed_vals, color=palette[i % len(palette)],
                   s=90, label=seed, zorder=4, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(sub_labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Cross-Seed Consistency: Worst-Forgotten Subjects (n={n_seeds} seeds)", fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig("results/figures/seed_consistency.png", dpi=150); plt.close()
    print("Saved: seed_consistency.png")

print("\nDone.")