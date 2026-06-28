"""Layer permutation: does the ORDER of layers 10-22 matter?
If yes -> computation is genuinely sequential (pipeline)
If no  -> computation is distributed within the zone (pool)"""
import torch, json, os, random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BASE = "Qwen/Qwen2.5-3B"
N_GSM = 200
N_MMLU = 100
N_BOOLQ = 200
N_PERMUTATIONS = 5
COG_START, COG_END = 10, 23  # layers 10-22 inclusive

def mmlu_fmt(x):
    c = x['choices']
    return (x['question'] + "\nA. " + c[0] + "\nB. " + c[1] +
            "\nC. " + c[2] + "\nD. " + c[3] + "\nAnswer:")

def eval_mmlu(model, tok, subject, label_ids, n):
    ds = load_dataset('cais/mmlu', subject, split='test')
    ds = ds.select(range(min(n, len(ds))))
    correct = 0
    for ex in ds:
        text = mmlu_fmt(ex)
        ids = tok(text, return_tensors='pt', truncation=True, max_length=512).to('cuda')
        with torch.no_grad():
            logits = model(**ids).logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        pred = max(range(4), key=lambda i: probs[label_ids[i]].item())
        if pred == ex['answer']:
            correct += 1
    return correct / len(ds) * 100

def eval_gsm8k(model, tok, n):
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    ds = ds.select(range(min(n, len(ds))))
    total_prob = 0; count = 0
    for ex in ds:
        ans = ex['answer'].split('####')[-1].strip()
        text = "Question: " + ex['question'] + "\nAnswer: The answer is"
        ids = tok(text, return_tensors='pt', truncation=True, max_length=512).to('cuda')
        ans_tokens = tok.encode(" " + ans, add_special_tokens=False)
        if not ans_tokens: continue
        with torch.no_grad():
            logits = model(**ids).logits[0, -1, :]
        prob = torch.softmax(logits.float(), dim=-1)[ans_tokens[0]].item()
        total_prob += prob; count += 1
    return total_prob / count * 100 if count > 0 else 0

def eval_boolq(model, tok, n):
    ds = load_dataset('google/boolq', split='validation')
    ds = ds.select(range(min(n, len(ds))))
    true_id = tok.encode(" True", add_special_tokens=False)[-1]
    false_id = tok.encode(" False", add_special_tokens=False)[-1]
    correct = 0
    for ex in ds:
        text = ex['passage'][:500] + "\nQuestion: " + ex['question'] + "?\nAnswer:"
        ids = tok(text, return_tensors='pt', truncation=True, max_length=512).to('cuda')
        with torch.no_grad():
            logits = model(**ids).logits[0, -1, :]
        pred = ex['answer'] == (logits[true_id].item() > logits[false_id].item())
        if pred:
            correct += 1
    return correct / len(ds) * 100

class LayerPermutationHook:
    """Reroutes forward passes through layers in a permuted order."""
    def __init__(self, model, perm):
        self.model = model
        self.perm = perm  # e.g. [10,15,12,18,11,...] for zone layers
        self.original_forward = None
        self.hooks = []

    def install(self):
        """Replace the model's layer list ordering during forward pass."""
        layers = self.model.model.layers
        # store original layer references
        self.original_layers = [layers[i] for i in range(len(layers))]

        # create permuted list: layers outside zone keep position,
        # layers inside zone get reordered
        zone_layers = [layers[i] for i in range(COG_START, COG_END)]
        permuted_zone = [zone_layers[self.perm[i]] for i in range(len(self.perm))]

        for i, pi in enumerate(range(COG_START, COG_END)):
            layers[pi] = permuted_zone[i]

    def remove(self):
        """Restore original layer ordering."""
        layers = self.model.model.layers
        for i, layer in enumerate(self.original_layers):
            layers[i] = layer

def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    label_ids = [tok.encode(f" {c}", add_special_tokens=False)[-1] for c in "ABCD"]

    zone_size = COG_END - COG_START  # 13 layers

    # baseline
    print("=== BASELINE (original order) ===", flush=True)
    gsm_base = eval_gsm8k(model, tok, N_GSM)
    alg_base = eval_mmlu(model, tok, 'abstract_algebra', label_ids, N_MMLU)
    bool_base = eval_boolq(model, tok, N_BOOLQ)
    print(f"  GSM8K:            {gsm_base:.2f}%")
    print(f"  Abstract Algebra: {alg_base:.2f}%")
    print(f"  BoolQ:            {bool_base:.2f}%")

    results = {
        'baseline': {'gsm8k': gsm_base, 'abstract_algebra': alg_base, 'boolq': bool_base},
        'permutations': []
    }

    # run permutations
    random.seed(42)
    for trial in range(N_PERMUTATIONS):
        perm = list(range(zone_size))
        random.shuffle(perm)
        perm_layers = [COG_START + p for p in perm]

        print(f"\n=== PERMUTATION {trial+1}/{N_PERMUTATIONS} ===", flush=True)
        print(f"  Order: {perm_layers}")

        hook = LayerPermutationHook(model, perm)
        hook.install()

        gsm = eval_gsm8k(model, tok, N_GSM)
        alg = eval_mmlu(model, tok, 'abstract_algebra', label_ids, N_MMLU)
        boolq = eval_boolq(model, tok, N_BOOLQ)

        hook.remove()

        print(f"  GSM8K:            {gsm:.2f}%  (Δ {gsm-gsm_base:+.2f})")
        print(f"  Abstract Algebra: {alg:.2f}%  (Δ {alg-alg_base:+.2f})")
        print(f"  BoolQ:            {boolq:.2f}%  (Δ {boolq-bool_base:+.2f})")

        results['permutations'].append({
            'trial': trial,
            'perm': perm_layers,
            'gsm8k': gsm,
            'abstract_algebra': alg,
            'boolq': boolq,
        })

    # summary
    gsm_deltas = [r['gsm8k'] - gsm_base for r in results['permutations']]
    alg_deltas = [r['abstract_algebra'] - alg_base for r in results['permutations']]
    bool_deltas = [r['boolq'] - bool_base for r in results['permutations']]

    print(f"\n=== SUMMARY ({N_PERMUTATIONS} permutations) ===")
    print(f"GSM8K mean Δ:            {np.mean(gsm_deltas):+.2f}pp  (std {np.std(gsm_deltas):.2f})")
    print(f"Abstract Algebra mean Δ: {np.mean(alg_deltas):+.2f}pp  (std {np.std(alg_deltas):.2f})")
    print(f"BoolQ mean Δ:            {np.mean(bool_deltas):+.2f}pp  (std {np.std(bool_deltas):.2f})")
    print()
    print("If GSM8K drops heavily but BoolQ doesn't:")
    print("  -> Layer ORDER matters for chained computation (it's a pipeline)")
    print("If neither drops:")
    print("  -> Layers are interchangeable (it's a pool)")

    outpath = 'mechanistic/results/layer_permutation/results.json'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

if __name__ == '__main__':
    main()
