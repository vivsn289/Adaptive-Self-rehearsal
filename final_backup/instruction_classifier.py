"""Classify each self-instruct instruction into a probe domain via regex.

Output: instruction_domains.json with {"labels": [str * N]}.

Broader patterns calibrated to catch a meaningful share of conversational
Self-Instruct instructions. Order = specificity (first match wins).
"""

import json, re

INSTRUCTIONS_PATH = "generated_data/self_instruct_generated.json"
OUTPUT_PATH       = "instruction_domains.json"

PATTERNS = [
    ("formal_quant", [
        r"\b(equation|formula|theorem|axiom|lemma|proof|prove|derive|derivation)\b",
        r"\b(algebra|geometry|trigonometry|calculus|topology|matrix|polynomial|determinant)\b",
        r"\b(differential|integral|integrate|differentiate|derivative|antiderivative)\b",
        r"\b(physics|chemistry|quantum|relativity|kinetic|momentum|wavelength|frequency)\b",
        r"\b(voltage|current|circuit|resistance|capacitance|inductance|ohm)\b",
        r"\b(atom|molecule|electron|proton|neutron|photon|particle|nucleus|isotope)\b",
        r"\b(force|mass|acceleration|velocity|gravity|friction|torque|inertia)\b",
        r"\b(logic|logical|propositional|predicate|inference|deduction|syllogism)\b",
        r"\b(solve|prove|show that|demonstrate that)\b",
        r"\b(function|equation|graph|plot|curve|slope|intercept)\b",
    ]),
    ("math_words", [
        r"\b(how many|how much|how often|how long|how far|how old)\b",
        r"\b(add|subtract|multiply|divide|sum|product|difference|quotient)\b",
        r"\bif (he|she|they|the|a|an|each|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        r"\b\d+\s*(apples|oranges|books|cars|dollars|cents|miles|kilometers|hours|minutes|seconds|days|weeks|months|years|items|people|pieces|times)",
        r"\b(percent|percentage|ratio|fraction|proportion|average|mean|median)\b",
        r"\bthere (are|were|is|was) \d+",
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\b(total|altogether|combined|together)\b",
    ]),
    ("causal_commonsense", [
        r"\b(because|since|therefore|hence|thus|consequently|as a result|due to)\b",
        r"\b(cause|caused|causes|causing|effect|effects|affect|affects|affecting)\b",
        r"\b(impact|impacts|influence|influences|consequence|consequences)\b",
        r"\b(what (would|could|might) happen)\b",
        r"\b(would happen if|happens? (if|when))\b",
        r"\bwhy\b",
        r"\b(predict|prediction|forecast|expect|expected|outcome)\b",
        r"\b(infer|conclude|implies?|imply|implication)\b",
        r"\b(reason|reasons|reasoning|rationale)\b",
    ]),
    ("comprehension", [
        r"\b(passage|paragraph|article|text|essay|story|document|excerpt)\b",
        r"\bgiven (the|this|a|an|following)\b",
        r"\b(summary|summarize|summarise|paraphrase|rephrase|restate)\b",
        r"\b(read|reading)\b",
        r"\bthe (author|writer|speaker|narrator)\b",
        r"\b(explain|describe|elaborate|clarify|illustrate)\b",
        r"\b(meaning|definition|interpret|interpretation) of\b",
        r"\b(main idea|central theme|key point|thesis|argument)\b",
        r"\bin your own words\b",
    ]),
    ("factual", [
        r"\b(who (is|was|are|were|invented|wrote|discovered|founded|created|developed|built|painted|composed))\b",
        r"\b(what (is|was|are|were) (the|a|an))\b",
        r"\b(when (did|was|is|were|are|do|does))\b",
        r"\b(where (is|was|are|were|did|does|do))\b",
        r"\b(which (country|city|state|year|decade|century|continent|ocean|river|mountain))\b",
        r"\b(list|name|enumerate|mention|cite|identify)\b",
        r"\b(capital of|population of|currency of|language of|history of)\b",
        r"\b(famous|known for|notable for|recognized for)\b",
        r"\b(define|definition)\b",
        r"\bbiography\b",
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
    text = instr.get("instruction", "")
    if "input" in instr:
        text += " " + str(instr.get("input", ""))
    label = classify(text)
    labels.append(label)
    counts[label] += 1

with open(OUTPUT_PATH, "w") as f:
    json.dump({"labels": labels}, f)

print(f"Classified {len(labels)} instructions into domains:")
for d, c in counts.items():
    print(f"  {d:<22}: {c:>4}  ({100*c/len(labels):.1f}%)")
print(f"\nSaved to {OUTPUT_PATH}")
