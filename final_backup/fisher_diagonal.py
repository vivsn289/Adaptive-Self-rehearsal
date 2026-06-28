"""Fisher Information Matrix diagonals: which parameters are CRITICAL for each task type?
Compare per-layer Fisher concentration between cognitive (GSM8K) and recall (BoolQ) tasks."""
import torch, json, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BASE = "Qwen/Qwen2.5-3B"
N = 100  # examples per task

def compute_fisher(model, tok, dataset_fn, n):
    """Accumulate squared gradients (Fisher diagonal) over n examples."""
    fisher = {n: torch.zeros_like(p, dtype=torch.float32)
              for n, p in model.named_parameters() if p.requires_grad}

    model.train()  # need gradients, but no optimizer
    count = 0
    for text in dataset_fn(n):
        ids = tok(text, return_tensors='pt', truncation=True, max_length=256).to('cuda')
        if ids.input_ids.shape[1] < 2:
            continue
        labels = ids.input_ids.clone()
        out = model(**ids, labels=labels)
        out.loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.float() ** 2
                p.grad = None  # free memory immediately

        count += 1
        if count % 20 == 0:
            print(f"  {count}/{n} examples processed", flush=True)

    # average
    for name in fisher:
        fisher[name] /= count
    model.eval()
    return fisher, count

def gsm8k_texts(n):
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    for i, ex in enumerate(ds):
        if i >= n: break
        yield "Question: " + ex['question'] + "\nAnswer: " + ex['answer']

def boolq_texts(n):
    ds = load_dataset('google/boolq', split='validation')
    for i, ex in enumerate(ds):
        if i >= n: break
        ans = "True" if ex['answer'] else "False"
        yield ex['passage'][:300] + "\nQuestion: " + ex['question'] + "?\nAnswer: " + ans

def aggregate_per_layer(fisher, num_layers=36):
    """Sum Fisher values per layer."""
    layer_fisher = {}
    other_fisher = 0.0
    other_count = 0

    for l in range(num_layers):
        prefix = f"model.layers.{l}."
        total = 0.0
        count = 0
        for name, val in fisher.items():
            if prefix in name:
                total += val.sum().item()
                count += val.numel()
        layer_fisher[l] = {'total': total, 'mean': total / count if count > 0 else 0, 'num_params': count}

    # non-layer params (embed, lm_head, norm)
    for name, val in fisher.items():
        if 'model.layers.' not in name:
            other_fisher += val.sum().item()
            other_count += val.numel()
    layer_fisher['other'] = {'total': other_fisher, 'mean': other_fisher / other_count if other_count > 0 else 0, 'num_params': other_count}

    return layer_fisher

def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map='cuda')

    print("=== Computing Fisher on GSM8K (cognitive) ===", flush=True)
    fisher_cog, n_cog = compute_fisher(model, tok, gsm8k_texts, N)
    cog_layers = aggregate_per_layer(fisher_cog)

    # clear fisher to free memory
    del fisher_cog
    torch.cuda.empty_cache()

    print("\n=== Computing Fisher on BoolQ (recall) ===", flush=True)
    fisher_rec, n_rec = compute_fisher(model, tok, boolq_texts, N)
    rec_layers = aggregate_per_layer(fisher_rec)

    del fisher_rec
    torch.cuda.empty_cache()

    # print comparison
    print(f"\n{'Layer':>5}  {'GSM8K Fisher':>14}  {'BoolQ Fisher':>14}  {'Ratio (G/B)':>12}")
    print("-" * 50)

    cog_zone = []
    rec_zone = []
    cog_other = []
    rec_other = []

    for l in range(36):
        g = cog_layers[l]['mean']
        b = rec_layers[l]['mean']
        ratio = g / b if b > 0 else float('inf')
        zone = " *" if 10 <= l <= 22 else ""
        print(f"{l:5d}  {g:14.6e}  {b:14.6e}  {ratio:12.2f}{zone}")

        if 10 <= l <= 22:
            cog_zone.append(g)
            rec_zone.append(b)
        elif l >= 2:  # skip layers 0-1
            cog_other.append(g)
            rec_other.append(b)

    import numpy as np
    print(f"\n=== SUMMARY ===")
    print(f"GSM8K Fisher mean in zone 10-22:     {np.mean(cog_zone):.6e}")
    print(f"GSM8K Fisher mean outside zone:       {np.mean(cog_other):.6e}")
    print(f"GSM8K zone/other ratio:               {np.mean(cog_zone)/np.mean(cog_other):.2f}x")
    print(f"")
    print(f"BoolQ Fisher mean in zone 10-22:      {np.mean(rec_zone):.6e}")
    print(f"BoolQ Fisher mean outside zone:        {np.mean(rec_other):.6e}")
    print(f"BoolQ zone/other ratio:                {np.mean(rec_zone)/np.mean(rec_other):.2f}x")
    print(f"")
    print(f"If GSM8K concentrates in zone but BoolQ doesn't,")
    print(f"  that explains why uniform SFT selectively damages chained computation.")

    # save results
    results = {
        'n_cognitive': n_cog,
        'n_recall': n_rec,
        'cognitive_per_layer': {str(k): v for k, v in cog_layers.items()},
        'recall_per_layer': {str(k): v for k, v in rec_layers.items()},
    }
    outpath = 'mechanistic/results/fisher/results.json'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

if __name__ == '__main__':
    main()
