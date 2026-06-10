"""Classify each self-instruct instruction into a probe domain via regex.

Output: instruction_domains.json with {"labels": [str * N]} parallel to the
instruction list. Order matches generated_data/self_instruct_generated.json.

Heuristic classification — fast and defensible but imperfect. Acknowledged
as a limitation in the methodology writeup.
"""

import json, re

INSTRUCTIONS_PATH = "generated_data/self_instruct_generated.json"
OUTPUT_PATH       = "instruction_domains.json"

# Patterns ordered by specificity — first match wins.
PATTERNS = [
    ("formal_quant", [
        r"\b(prove|theorem|axiom|lemma|differentiate|integrate|integral|derivative)\b",
        r"\b(algebra|topology|set theory|group theory|polynomial|matrix|determinant)\b",
        r"\b(quantum|relativity|kinetic|potential energy|electric field|magnetic field)\b",
        r"\b(formal logic|propositional|first-order|predicate logic|inference rule|truth table)\b",
        r"\b(force|momentum|wavelength|frequency|voltage|current|circuit)\b",
    ]),
    ("math_words", [
        r"\b(calculate|compute|how many|how much|what is the (total|sum|product|average|difference))\b",
        r"\b(percentage|percent of|fraction of|ratio of)\b",
        r"\b\d+\s*(plus|minus|times|divided by|\+|\-|\*|/)\s*\d+",
        r"\bif (he|she|they|the|a|each) (has|have|had|bought|sold|gives|gave)\b",
    ]),
    ("causal_commonsense", [
        r"\b(because|therefore|hence|thus|as a result|leads to|causes|caused by)\b",
        r"\b(what would happen if|what happens when|what's the (result|effect|consequence))\b",
        r"\b(why does|why is|why would|why did)\b",
    ]),
    ("comprehension", [
        r"\b(read the following|given the passage|according to the (text|passage|article))\b",
        r"\b(summarize|paraphrase|explain in your own words)\b",
        r"\b(is the following statement (true|false))\b",
        r"\b(what does (the author|this) (mean|imply|suggest))\b",
    ]),
    ("factual", [
        r"\b(who (is|was|invented|wrote|discovered|founded))\b",
        r"\b(what (is|was) the (capital|population|currency|language|name) of)\b",
        r"\b(when (did|was|is)|in what year|in which year)\b",
        r"\b(where (is|was|did))\b",
        r"\b(list (the|some|all))\b",
        r"\b(name (the|a|some|several))\b",
    ]),
]

def classify(text):
    text_lower = text.lower()
    for domain, patterns in PATTERNS:
        for p in patterns:
            if re.search(p, text_lower):
                return domain
    return "other"

with open(INSTRUCTIONS_PATH) as f:
    instructions = json.load(f)

labels = []
counts = {"formal_quant": 0, "math_words": 0, "causal_commonsense": 0,
          "comprehension": 0, "factual": 0, "other": 0}

for instr in instructions:
    # Data has flat instruction/input fields (not nested instances)
    text = instr.get("instruction", "")
    if instr.get("input", "").strip():
        text += " " + instr.get("input", "")
    label = classify(text)
    labels.append(label)
    counts[label] += 1

with open(OUTPUT_PATH, "w") as f:
    json.dump({"labels": labels}, f)

print(f"Classified {len(labels)} instructions into domains:")
for d, c in counts.items():
    print(f"  {d:<22}: {c:>4}  ({100*c/len(labels):.1f}%)")
print(f"\nSaved to {OUTPUT_PATH}")
