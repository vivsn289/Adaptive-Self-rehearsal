"""Generate formal logic rehearsal problems programmatically.
Covers: propositional eval, modus ponens chains, syllogisms, contrapositive,
        equivalence, argument validity, De Morgan, quantifier negation.
Output format matches rehearsal bank JSONL.
"""
import json, random, argparse, itertools

# ── Propositional logic engine ───────────────────────────────────────

OPS = {
    '∧': lambda a, b: a and b,
    '∨': lambda a, b: a or b,
    '→': lambda a, b: (not a) or b,
    '↔': lambda a, b: a == b,
}

def eval_expr(expr, vals):
    """Evaluate a propositional expression tree."""
    if isinstance(expr, str):
        return vals[expr]
    op = expr[0]
    if op == '¬':
        return not eval_expr(expr[1], vals)
    return OPS[op](eval_expr(expr[1], vals), eval_expr(expr[2], vals))

def expr_to_str(expr):
    if isinstance(expr, str):
        return expr
    op = expr[0]
    if op == '¬':
        inner = expr_to_str(expr[1])
        return f"¬{inner}" if isinstance(expr[1], str) else f"¬({inner})"
    l, r = expr_to_str(expr[1]), expr_to_str(expr[2])
    if not isinstance(expr[1], str) and len(expr[1]) > 1:
        l = f"({l})"
    if not isinstance(expr[2], str) and len(expr[2]) > 1:
        r = f"({r})"
    return f"{l} {op} {r}"

def random_expr(rng, vars, depth=0, max_depth=2):
    if depth >= max_depth or (depth > 0 and rng.random() < 0.4):
        return rng.choice(vars)
    if rng.random() < 0.25:
        return ('¬', random_expr(rng, vars, depth + 1, max_depth))
    op = rng.choice(['∧', '∨', '→', '↔'])
    return (op, random_expr(rng, vars, depth + 1, max_depth),
                random_expr(rng, vars, depth + 1, max_depth))

# ── Problem generators ──────────────────────────────────────────────

def gen_prop_eval(rng, n=55):
    """Evaluate a propositional expression given truth values."""
    problems = []
    seen = set()
    attempts = 0
    while len(problems) < n and attempts < n * 10:
        attempts += 1
        nvars = rng.choice([2, 3])
        vars = ['P', 'Q', 'R'][:nvars]
        expr = random_expr(rng, vars, max_depth=2)
        s = expr_to_str(expr)
        if s in seen or len(s) < 5: continue
        seen.add(s)
        vals = {v: rng.choice([True, False]) for v in vars}
        result = eval_expr(expr, vals)
        val_str = ", ".join(f"{v} = {'T' if vals[v] else 'F'}" for v in vars)
        instr = f"Given {val_str}, evaluate the expression: {s}"

        # step by step
        steps = []
        for v in vars:
            steps.append(f"{v} = {'True' if vals[v] else 'False'}")
        steps.append(f"Evaluating {s}:")
        steps.append(f"Result = {'True' if result else 'False'}")
        out = "\n".join(steps) + f"\n\nThe answer is {'True' if result else 'False'}."

        problems.append(make_entry(instr, out, "formal_logic"))
    return problems

def gen_modus_ponens_chain(rng, n=55):
    """Multi-step implication chains."""
    problems = []
    props = ['P', 'Q', 'R', 'S', 'T', 'U', 'W']
    seen = set()
    attempts = 0
    while len(problems) < n and attempts < n * 10:
        attempts += 1
        length = rng.randint(3, 5)
        chain = rng.sample(props, length)
        key = tuple(chain)
        if key in seen: continue
        seen.add(key)
        premises = [f"{chain[i]} → {chain[i+1]}" for i in range(len(chain)-1)]
        premises_str = ", ".join(premises)
        instr = f"Given the premises: {premises_str}, and that {chain[0]} is true. What can we conclude about {chain[-1]}?"
        steps = [f"1. {chain[0]} is true (given)"]
        for i in range(len(chain)-1):
            steps.append(f"{i+2}. {chain[i]} → {chain[i+1]} (given), and {chain[i]} is true, so {chain[i+1]} is true (modus ponens)")
        out = "\n".join(steps) + f"\n\nThe answer is {chain[-1]} is true."
        problems.append(make_entry(instr, out, "formal_logic"))
    return problems

def gen_syllogism_validity(rng, n=60):
    """Categorical syllogism validity checking."""
    categories = [
        'mammals', 'birds', 'reptiles', 'dogs', 'cats', 'fish',
        'students', 'athletes', 'musicians', 'scientists', 'teachers',
        'vehicles', 'flowers', 'trees', 'metals', 'planets',
    ]
    # Valid forms
    valid_forms = [
        # Barbara: All M are P, All S are M → All S are P
        ("All {M} are {P}", "All {S} are {M}", "All {S} are {P}", True),
        # Celarent: No M are P, All S are M → No S are P
        ("No {M} are {P}", "All {S} are {M}", "No {S} are {P}", True),
        # Darii: All M are P, Some S are M → Some S are P
        ("All {M} are {P}", "Some {S} are {M}", "Some {S} are {P}", True),
        # Modus Tollens style: All P are M, No S are M → No S are P
        ("All {P} are {M}", "No {S} are {M}", "No {S} are {P}", True),
    ]
    # Invalid forms
    invalid_forms = [
        # Undistributed middle
        ("All {P} are {M}", "All {S} are {M}", "All {S} are {P}", False),
        # Illicit major
        ("All {M} are {P}", "No {S} are {M}", "No {S} are {P}", False),
        # Affirming consequent (adapted)
        ("All {M} are {P}", "Some {S} are {P}", "Some {S} are {M}", False),
        # Existential from universals
        ("All {M} are {P}", "All {S} are {M}", "Some {S} are {P}", False),
    ]
    problems = []
    seen = set()
    all_forms = valid_forms + invalid_forms
    attempts = 0
    while len(problems) < n and attempts < n * 10:
        attempts += 1
        form = rng.choice(all_forms)
        cats = rng.sample(categories, 3)
        S, M, P = cats
        p1 = form[0].format(S=S, M=M, P=P)
        p2 = form[1].format(S=S, M=M, P=P)
        conc = form[2].format(S=S, M=M, P=P)
        key = (p1, p2, conc)
        if key in seen: continue
        seen.add(key)
        valid = form[3]
        ans = "Valid" if valid else "Invalid"
        if valid:
            reason = "The conclusion follows necessarily from the premises by the rules of categorical syllogism."
        else:
            reason = "The conclusion does not follow necessarily from the premises. The syllogistic form commits a logical fallacy."
        instr = f"Determine whether the following argument is valid or invalid.\nPremise 1: {p1}\nPremise 2: {p2}\nConclusion: {conc}"
        out = (
            f"Analyzing the syllogistic form:\n"
            f"Premise 1: {p1}\n"
            f"Premise 2: {p2}\n"
            f"Conclusion: {conc}\n\n"
            f"{reason}\n\n"
            f"The answer is {ans}."
        )
        problems.append(make_entry(instr, out, "formal_logic"))
    return problems

def gen_contrapositive(rng, n=45):
    """Identify the contrapositive of a conditional statement."""
    statements = [
        ("it rains", "the ground is wet"),
        ("x is even", "x is divisible by 2"),
        ("a number is prime", "it has exactly two factors"),
        ("an animal is a dog", "it is a mammal"),
        ("a shape is a square", "it has four equal sides"),
        ("f is differentiable", "f is continuous"),
        ("a matrix is invertible", "its determinant is nonzero"),
        ("n > 5", "n > 3"),
        ("x^2 = 0", "x = 0"),
        ("a group is cyclic", "it is abelian"),
        ("the function is bounded", "it has a finite supremum"),
        ("x is rational", "x can be written as p/q"),
        ("a triangle is equilateral", "all its angles are 60 degrees"),
        ("a number ends in 0", "it is divisible by 10"),
        ("a set is finite", "it has a well-defined cardinality"),
        ("f is integrable", "f is measurable"),
        ("the series converges", "the terms approach zero"),
        ("a graph is connected", "there is a path between any two vertices"),
        ("a language is regular", "it is recognized by a finite automaton"),
        ("x is in the kernel", "f(x) = 0"),
    ]
    problems = []
    seen = set()
    attempts = 0
    while len(problems) < n and attempts < n * 15:
        attempts += 1
        idx = rng.randint(0, len(statements) - 1)
        if idx in seen: continue
        seen.add(idx)
        p, q = statements[idx]
        instr = f"What is the contrapositive of the statement: \"If {p}, then {q}\"?"
        out = (
            f"The contrapositive of \"If P, then Q\" is \"If not Q, then not P.\"\n\n"
            f"Original: If {p}, then {q}.\n"
            f"Contrapositive: If it is not the case that {q}, then it is not the case that {p}.\n\n"
            f"The answer is: \"If not ({q}), then not ({p}).\""
        )
        problems.append(make_entry(instr, out, "formal_logic"))
        if len(seen) >= len(statements): break
    return problems

def gen_argument_validity(rng, n=55):
    """Propositional argument validity (named forms)."""
    forms = [
        # Valid forms
        {
            "name": "Modus Ponens",
            "premises": ["P → Q", "P"],
            "conclusion": "Q",
            "valid": True,
            "reason": "Modus ponens: from P → Q and P, we derive Q.",
        },
        {
            "name": "Modus Tollens",
            "premises": ["P → Q", "¬Q"],
            "conclusion": "¬P",
            "valid": True,
            "reason": "Modus tollens: from P → Q and ¬Q, we derive ¬P.",
        },
        {
            "name": "Hypothetical Syllogism",
            "premises": ["P → Q", "Q → R"],
            "conclusion": "P → R",
            "valid": True,
            "reason": "Hypothetical syllogism: from P → Q and Q → R, we derive P → R.",
        },
        {
            "name": "Disjunctive Syllogism",
            "premises": ["P ∨ Q", "¬P"],
            "conclusion": "Q",
            "valid": True,
            "reason": "Disjunctive syllogism: from P ∨ Q and ¬P, we derive Q.",
        },
        {
            "name": "Constructive Dilemma",
            "premises": ["P → Q", "R → S", "P ∨ R"],
            "conclusion": "Q ∨ S",
            "valid": True,
            "reason": "Constructive dilemma: valid argument form.",
        },
        # Invalid forms
        {
            "name": "Affirming the Consequent",
            "premises": ["P → Q", "Q"],
            "conclusion": "P",
            "valid": False,
            "reason": "Affirming the consequent: from P → Q and Q, we cannot conclude P. Q could be true for other reasons.",
        },
        {
            "name": "Denying the Antecedent",
            "premises": ["P → Q", "¬P"],
            "conclusion": "¬Q",
            "valid": False,
            "reason": "Denying the antecedent: from P → Q and ¬P, we cannot conclude ¬Q. Q might still be true.",
        },
        {
            "name": "Undistributed Middle (prop)",
            "premises": ["P → R", "Q → R"],
            "conclusion": "P → Q",
            "valid": False,
            "reason": "Invalid: sharing a consequent does not establish a connection between the antecedents.",
        },
    ]
    props_pool = ['P', 'Q', 'R', 'S', 'T', 'U']
    problems = []
    seen = set()
    attempts = 0
    while len(problems) < n and attempts < n * 15:
        attempts += 1
        form = rng.choice(forms)
        # Substitute variable names for variety
        used = set()
        mapping = {}
        for v in ['P', 'Q', 'R', 'S']:
            choices = [p for p in props_pool if p not in used]
            if not choices: break
            new = rng.choice(choices)
            used.add(new)
            mapping[v] = new

        def sub(s):
            for old, new in mapping.items():
                s = s.replace(old, new)
            return s

        premises = [sub(p) for p in form["premises"]]
        conclusion = sub(form["conclusion"])
        key = (tuple(premises), conclusion)
        if key in seen: continue
        seen.add(key)

        premises_str = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(premises))
        ans = "Valid" if form["valid"] else "Invalid"
        instr = f"Is the following argument valid or invalid?\nPremises:\n{premises_str}\nConclusion: {conclusion}"
        out = (
            f"Analyzing the argument form:\n"
            f"This is an instance of {form['name']}.\n"
            f"{sub(form['reason'])}\n\n"
            f"The answer is {ans}."
        )
        problems.append(make_entry(instr, out, "formal_logic"))
    return problems

def gen_de_morgan(rng, n=40):
    """Apply De Morgan's laws."""
    problems = []
    vars_pool = ['P', 'Q', 'R', 'S']
    seen = set()
    attempts = 0
    while len(problems) < n and attempts < n * 15:
        attempts += 1
        nvars = rng.choice([2, 3])
        vars = rng.sample(vars_pool, nvars)
        # ¬(A ∧ B ∧ ...) = ¬A ∨ ¬B ∨ ...
        # ¬(A ∨ B ∨ ...) = ¬A ∧ ¬B ∧ ...
        op = rng.choice(['∧', '∨'])
        negations = [rng.choice([True, False]) for _ in vars]
        terms = [f"¬{v}" if neg else v for v, neg in zip(vars, negations)]
        inner = f" {op} ".join(terms)
        expr = f"¬({inner})"
        if expr in seen: continue
        seen.add(expr)
        # Apply De Morgan
        new_op = '∨' if op == '∧' else '∧'
        new_terms = []
        for v, neg in zip(vars, negations):
            if neg:
                new_terms.append(v)  # ¬(¬v) = v
            else:
                new_terms.append(f"¬{v}")
        result = f" {new_op} ".join(new_terms)
        law = "¬(A ∧ B) ≡ ¬A ∨ ¬B" if op == '∧' else "¬(A ∨ B) ≡ ¬A ∧ ¬B"
        instr = f"Apply De Morgan's law to simplify: {expr}"
        out = (
            f"De Morgan's law states: {law}\n"
            f"Applying to {expr}:\n"
            f"Negate each component and flip {op} to {new_op}:\n"
            f"= {result}\n\n"
            f"The answer is {result}."
        )
        problems.append(make_entry(instr, out, "formal_logic"))
    return problems

def gen_logical_equivalence(rng, n=45):
    """Check if two expressions are logically equivalent."""
    # Known equivalences and non-equivalences
    equivs = [
        ("P → Q", "¬P ∨ Q", True, "A conditional P → Q is equivalent to ¬P ∨ Q by material implication."),
        ("P → Q", "¬Q → ¬P", True, "A conditional is equivalent to its contrapositive."),
        ("¬(P ∧ Q)", "¬P ∨ ¬Q", True, "By De Morgan's law."),
        ("¬(P ∨ Q)", "¬P ∧ ¬Q", True, "By De Morgan's law."),
        ("P ↔ Q", "(P → Q) ∧ (Q → P)", True, "A biconditional is equivalent to conjunction of both conditionals."),
        ("P ∧ (Q ∨ R)", "(P ∧ Q) ∨ (P ∧ R)", True, "By the distributive law."),
        ("P ∨ (Q ∧ R)", "(P ∨ Q) ∧ (P ∨ R)", True, "By the distributive law."),
        ("P → Q", "Q → P", False, "A conditional is not equivalent to its converse. P → Q and Q → P have different truth tables."),
        ("P → Q", "P → ¬Q", False, "These are not equivalent. If P is true and Q is true, the first is true but the second is false."),
        ("P ∨ Q", "P ∧ Q", False, "Disjunction and conjunction are not equivalent. When P is true and Q is false, P ∨ Q is true but P ∧ Q is false."),
        ("P → (Q → R)", "(P → Q) → R", False, "These have different logical structures and different truth tables."),
        ("P ∧ (P ∨ Q)", "P", True, "By the absorption law, P ∧ (P ∨ Q) ≡ P."),
        ("P ∨ (P ∧ Q)", "P", True, "By the absorption law, P ∨ (P ∧ Q) ≡ P."),
    ]
    problems = []
    seen = set()
    vars_sub = ['P', 'Q', 'R', 'S', 'T', 'U']
    attempts = 0
    while len(problems) < n and attempts < n * 15:
        attempts += 1
        eq = rng.choice(equivs)
        e1, e2, is_eq, reason = eq
        # variable substitution for variety
        orig_vars = sorted(set(c for c in e1 + e2 if c.isupper() and c not in '∧∨→↔¬'))
        if len(orig_vars) <= len(vars_sub):
            new_vars = rng.sample(vars_sub, len(orig_vars))
            mapping = dict(zip(orig_vars, new_vars))
            def sub(s):
                for old, new in mapping.items():
                    s = s.replace(old, new)
                return s
            e1s, e2s = sub(e1), sub(e2)
            reasons = sub(reason)
        else:
            e1s, e2s, reasons = e1, e2, reason
        key = (e1s, e2s)
        if key in seen: continue
        seen.add(key)
        ans = "Yes" if is_eq else "No"
        instr = f"Are the following two propositions logically equivalent?\n  (1) {e1s}\n  (2) {e2s}"
        out = (
            f"{reasons}\n\n"
            f"The answer is {ans}, they are {'logically equivalent' if is_eq else 'not logically equivalent'}."
        )
        problems.append(make_entry(instr, out, "formal_logic"))
    return problems

def gen_quantifier_negation(rng, n=40):
    """Negate quantified statements."""
    templates = [
        {
            "stmt": "For all x, {pred}",
            "negation": "There exists an x such that not ({pred})",
            "rule": "The negation of ∀x P(x) is ∃x ¬P(x).",
        },
        {
            "stmt": "There exists an x such that {pred}",
            "negation": "For all x, not ({pred})",
            "rule": "The negation of ∃x P(x) is ∀x ¬P(x).",
        },
    ]
    predicates = [
        "x is positive", "x > 0", "x^2 ≥ 0", "f(x) is continuous",
        "x is rational", "x has a multiplicative inverse",
        "the sequence converges", "x is divisible by 3",
        "g(x) = 0", "x belongs to the set S",
        "x is a prime number", "x + y = y + x",
        "the function f is bounded at x", "x is an eigenvalue",
        "the graph contains vertex x", "x is a solution",
    ]
    problems = []
    seen = set()
    attempts = 0
    while len(problems) < n and attempts < n * 15:
        attempts += 1
        tmpl = rng.choice(templates)
        pred = rng.choice(predicates)
        stmt = tmpl["stmt"].format(pred=pred)
        if stmt in seen: continue
        seen.add(stmt)
        neg = tmpl["negation"].format(pred=pred)
        instr = f"What is the negation of the statement: \"{stmt}\"?"
        out = (
            f"{tmpl['rule']}\n\n"
            f"Original: {stmt}\n"
            f"Negation: {neg}\n\n"
            f"The answer is: \"{neg}\"."
        )
        problems.append(make_entry(instr, out, "formal_logic"))
    return problems

# ── Helpers ──────────────────────────────────────────────────────────

def make_entry(instruction, output, subtype):
    return {
        "instruction": instruction,
        "output": output,
        "tier": "TIER_1",
        "subtype": subtype,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="rehearsal_bank/logic_programmatic.jsonl")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    rng = random.Random(args.seed)
    all_problems = []

    generators = [
        ("prop_eval", gen_prop_eval),
        ("modus_ponens_chain", gen_modus_ponens_chain),
        ("syllogism_validity", gen_syllogism_validity),
        ("contrapositive", gen_contrapositive),
        ("argument_validity", gen_argument_validity),
        ("de_morgan", gen_de_morgan),
        ("logical_equivalence", gen_logical_equivalence),
        ("quantifier_negation", gen_quantifier_negation),
    ]

    for name, gen in generators:
        probs = gen(rng)
        print(f"  {name}: {len(probs)} problems")
        all_problems.extend(probs)

    rng.shuffle(all_problems)

    with open(args.out, "w") as f:
        for p in all_problems:
            f.write(json.dumps(p) + "\n")

    print(f"\nTotal: {len(all_problems)} logic problems → {args.out}")

if __name__ == "__main__":
    main()
