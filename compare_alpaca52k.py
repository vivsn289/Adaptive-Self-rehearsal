import json, os, glob, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR     = "final_backup/results/base"
INSTRUCT_DIR = "final_backup/results/instruct"
ALPACA_DIR   = "final_backup/results/alpaca52k"
FIGURES_DIR  = "final_backup/figures_52k"
os.makedirs(FIGURES_DIR, exist_ok=True)

METRIC_KEYS = [
    "acc_norm,none", "acc,none", "exact_match,none",
    "exact_match,strict-match", "exact_match,flexible-extract",
    "exact_match,remove_whitespace",
]

def load_scores_from_dir(d):
    scores = {}
    for root, dirs, files in os.walk(d):
        for fn in files:
            if fn.startswith("results_") and fn.endswith(".json"):
                try:
                    data = json.load(open(os.path.join(root, fn)))
                except: continue
                for task, metrics in data.get("results", {}).items():
                    if task in scores: continue
                    for key in METRIC_KEYS:
                        if key in metrics and metrics[key] is not None:
                            scores[task] = round(float(metrics[key]) * 100, 2)
                            break
    return scores

def load_multi_seed(parent, pattern):
    out = {}
    for d in sorted(glob.glob(os.path.join(parent, pattern))):
        if os.path.isdir(d):
            s = load_scores_from_dir(d)
            if s: out[os.path.basename(d)] = s
            else: print(f"  WARN: no scores in {d}")
    return out

def mean_se(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if len(a) == 0: return 0., 0.
    return float(a.mean()), float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else (float(a.mean()), 0.)

def agg(per_seed, tasks):
    ms, ss = [], []
    for t in tasks:
        v = [per_seed[s][t] for s in per_seed if t in per_seed[s]]
        m, se = mean_se(v); ms.append(m); ss.append(se)
    return ms, ss

print("Loading base ..."); base = load_scores_from_dir(BASE_DIR); print(f"  {len(base)} tasks")
print("Loading 2K ..."); instruct = load_multi_seed(INSTRUCT_DIR, "seed*-ep4"); print(f"  Seeds: {list(instruct.keys())}")
print("Loading 52K ..."); alpaca = load_multi_seed(ALPACA_DIR, "seed*-ep1"); print(f"  Seeds: {list(alpaca.keys())}")

if not base: raise RuntimeError("No base results")
if not instruct: raise RuntimeError("No 2K results")
if not alpaca: raise RuntimeError("No 52K results")

n_i, n_a = len(instruct), len(alpaca)

# --- Print table ---
TOP = [t for t in ["arc_easy","arc_challenge","boolq","rte","copa","gsm8k",
       "lambada_openai","piqa","hellaswag","winogrande","openbookqa","triviaqa","mmlu"] if t in base]

print(f"\n{'='*115}")
print(f"THREE-WAY: Base vs 2K SFT (n={n_i}) vs 52K Alpaca (n={n_a})")
print(f"{'='*115}")
print(f"{'Task':<22} {'Base':>7} {'2K (mean+-SE)':>18} {'52K (mean+-SE)':>18} {'d2K':>8} {'d52K':>8}")
print("-"*115)
for t in TOP:
    b = base[t]
    iv = [instruct[s][t] for s in instruct if t in instruct[s]]
    av = [alpaca[s][t] for s in alpaca if t in alpaca[s]]
    im,ise = mean_se(iv); am,ase = mean_se(av)
    f2 = " *" if abs(im-b)>3 else ""; f5 = " *" if abs(am-b)>3 else ""
    print(f"{t:<22} {b:>7.2f} {im:>9.2f} +-{ise:<5.2f}  {am:>9.2f} +-{ase:<5.2f}  {im-b:>+7.2f}{f2:<2} {am-b:>+7.2f}{f5}")

# MMLU subjects
all_mmlu = sorted([t for t in base if "mmlu" in t.lower() and t != "mmlu"
                   and any(t in instruct[s] for s in instruct)
                   and any(t in alpaca[s] for s in alpaca)])
if all_mmlu:
    print(f"\n{'='*115}")
    print("KEY MMLU SUBJECTS")
    print(f"{'='*115}")
    for t in sorted(all_mmlu, key=lambda t: base.get(t,0) - mean_se([instruct[s][t] for s in instruct if t in instruct[s]])[0], reverse=True)[:20]:
        b=base[t]; iv=[instruct[s][t] for s in instruct if t in instruct[s]]; av=[alpaca[s][t] for s in alpaca if t in alpaca[s]]
        im,ise=mean_se(iv); am,ase=mean_se(av)
        print(f"{t:<40} {b:>7.2f} {im:>9.2f} +-{ise:<5.2f}  {am:>9.2f} +-{ase:<5.2f}  {im-b:>+7.2f}  {am-b:>+7.2f}")

# --- Figure 1: Top-level 3-way ---
fig,ax = plt.subplots(figsize=(16,6)); x=np.arange(len(TOP)); w=0.27
bv=[base[t] for t in TOP]; im_,ise_=agg(instruct,TOP); am_,ase_=agg(alpaca,TOP)
ax.bar(x-w,bv,w,label="Base",color="#3274A1")
ax.bar(x,im_,w,yerr=ise_,label=f"2K SFT (n={n_i})",color="#E1812C",error_kw=dict(ecolor="black",capsize=3))
ax.bar(x+w,am_,w,yerr=ase_,label=f"52K SFT (n={n_a})",color="#3A923A",error_kw=dict(ecolor="black",capsize=3))
ax.set_xticks(x); ax.set_xticklabels([t.replace("_"," ").title() for t in TOP],rotation=35,ha="right",fontsize=9)
ax.set_ylabel("Accuracy (%)"); ax.set_title("Top-Level Tasks: Base vs 2K SFT vs 52K Alpaca SFT",fontweight="bold")
ax.legend(); ax.grid(axis="y",alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,"01_toplevel_3way.png"),dpi=150); plt.close(); print(f"Saved 01")

# --- Figure 2: MMLU subjects 3-way ---
if all_mmlu:
    SHOW = sorted(all_mmlu, key=lambda t: base.get(t,0)-mean_se([instruct[s][t] for s in instruct if t in instruct[s]])[0], reverse=True)[:15]
    fig,ax = plt.subplots(figsize=(18,6)); x2=np.arange(len(SHOW)); w2=0.27
    bv2=[base[t] for t in SHOW]; im2,ise2=agg(instruct,SHOW); am2,ase2=agg(alpaca,SHOW)
    ax.bar(x2-w2,bv2,w2,label="Base",color="#3274A1")
    ax.bar(x2,im2,w2,yerr=ise2,label=f"2K SFT",color="#E1812C",error_kw=dict(ecolor="black",capsize=3))
    ax.bar(x2+w2,am2,w2,yerr=ase2,label=f"52K SFT",color="#3A923A",error_kw=dict(ecolor="black",capsize=3))
    ax.set_xticks(x2); ax.set_xticklabels([t.split("mmlu_",1)[-1].replace("_"," ").title() if "mmlu_" in t else t.split("hendrycksTest-",1)[-1].replace("_"," ").title() for t in SHOW],rotation=40,ha="right",fontsize=8)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("MMLU Subjects (top 15 most-forgotten at 2K)",fontweight="bold")
    ax.legend(); ax.grid(axis="y",alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR,"02_mmlu_subjects_3way.png"),dpi=150); plt.close(); print(f"Saved 02")

# --- Figure 3: Reasoning vs Recall ---
REASON = ["gsm8k","copa","arc_challenge"]+[t for t in all_mmlu if any(k in t for k in ["abstract_algebra","college_physics","formal_logic"])]
RECALL = ["boolq","rte","arc_easy","triviaqa"]+[t for t in all_mmlu if any(k in t for k in ["high_school_physics","medical_genetics"])]

def axis_d(tl,ps,bs):
    ds=[]
    for t in tl:
        if t not in bs: continue
        v=[ps[s][t] for s in ps if t in ps[s]]
        if not v: continue
        m,_=mean_se(v); ds.append(m-bs[t])
    return ds

r2k=axis_d(REASON,instruct,base); rc2k=axis_d(RECALL,instruct,base)
r52k=axis_d(REASON,alpaca,base); rc52k=axis_d(RECALL,alpaca,base)
fig,ax=plt.subplots(figsize=(8,6))
dm=[np.mean(r2k) if r2k else 0, np.mean(rc2k) if rc2k else 0, np.mean(r52k) if r52k else 0, np.mean(rc52k) if rc52k else 0]
ds=[np.std(r2k,ddof=1)/np.sqrt(len(r2k)) if len(r2k)>1 else 0, np.std(rc2k,ddof=1)/np.sqrt(len(rc2k)) if len(rc2k)>1 else 0, np.std(r52k,ddof=1)/np.sqrt(len(r52k)) if len(r52k)>1 else 0, np.std(rc52k,ddof=1)/np.sqrt(len(rc52k)) if len(rc52k)>1 else 0]
lb=["Reasoning\n2K SFT","Recall\n2K SFT","Reasoning\n52K SFT","Recall\n52K SFT"]
cl=["#E74C3C","#27AE60","#C0392B","#1E8449"]
bars=ax.bar(lb,dm,yerr=ds,color=cl,error_kw=dict(ecolor="black",capsize=5),width=0.55)
ax.axhline(0,color="black",lw=0.8)
for bar,val in zip(bars,dm):
    yp=val+0.3 if val>=0 else val-0.8
    ax.text(bar.get_x()+bar.get_width()/2,yp,f"{val:+.1f}pp",ha="center",va="bottom",fontsize=10,fontweight="bold")
ax.set_ylabel("Mean Accuracy Delta vs Base (pp)"); ax.set_title("Cognitive Axis: Reasoning Degrades, Recall Improves\nScales with Instruction Volume",fontweight="bold")
ax.grid(axis="y",alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,"03_reasoning_vs_recall_scaling.png"),dpi=150); plt.close(); print("Saved 03")

# --- Figure 4: GSM8K + TriviaQA ---
GT=[t for t in ["gsm8k","triviaqa"] if t in base]
if GT:
    fig,axes=plt.subplots(1,len(GT),figsize=(5*len(GT),5),sharey=False)
    if len(GT)==1: axes=[axes]
    for ax,task in zip(axes,GT):
        b=base[task]; iv=[instruct[s][task] for s in instruct if task in instruct[s]]; av=[alpaca[s][task] for s in alpaca if task in alpaca[s]]
        im,ise=mean_se(iv); am,ase=mean_se(av)
        ms=[b,im,am]; ses=[0,ise,ase]; cols=["#3274A1","#E1812C","#3A923A"]
        bars=ax.bar([0,1,2],ms,yerr=ses,color=cols,width=0.5,error_kw=dict(ecolor="black",capsize=5))
        ax.set_xticks([0,1,2]); ax.set_xticklabels(["Base","2K SFT","52K SFT"],fontsize=11)
        ax.set_ylabel("Score (%)"); note="\n(exact-match; format drift caveat)" if task=="triviaqa" else "\n(strict-match)"
        ax.set_title(f"{task.upper()}{note}",fontweight="bold",fontsize=10); ax.grid(axis="y",alpha=0.3)
        for bar,val,se in zip(bars,ms,ses): ax.text(bar.get_x()+bar.get_width()/2,val+se+0.5,f"{val:.1f}%",ha="center",va="bottom",fontsize=10)
        ax.annotate("",xy=(1,im),xytext=(0,b),arrowprops=dict(arrowstyle="->",color="red",lw=1.5))
        ax.annotate("",xy=(2,am),xytext=(0,b),arrowprops=dict(arrowstyle="->",color="darkred",lw=1.5))
        ax.text(0.5,min(b,im)-2.5,f"d={im-b:+.1f}pp",ha="center",fontsize=9,color="red")
        ax.text(1.5,min(b,am)-2.5,f"d={am-b:+.1f}pp",ha="center",fontsize=9,color="darkred")
    fig.suptitle("Forgetting Amplifies at Scale: Generation-Heavy Tasks",fontweight="bold",fontsize=13)
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR,"04_gsm8k_triviaqa_scaling.png"),dpi=150); plt.close(); print("Saved 04")

print(f"\nDone. All figures in {FIGURES_DIR}/")
