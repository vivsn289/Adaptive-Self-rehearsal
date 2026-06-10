"""Generate abstract algebra rehearsal problems programmatically.
Covers: element order, generators, Lagrange, fields, direct products, isomorphisms.
Output format matches rehearsal bank JSONL.
"""
import json, random, math, argparse

def gcd(a, b):
    while b: a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def euler_totient(n):
    r = n
    p = 2
    t = n
    while p * p <= t:
        if t % p == 0:
            while t % p == 0: t //= p
            r -= r // p
        p += 1
    if t > 1: r -= r // t
    return r

def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def factorize(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1: factors[n] = factors.get(n, 0) + 1
    return factors

def divisors(n):
    divs = []
    for i in range(1, n + 1):
        if n % i == 0: divs.append(i)
    return divs

# ── Problem generators ──────────────────────────────────────────────

def gen_element_order(rng, n=50):
    """Order of element x in (Z_n, +)."""
    problems = []
    seen = set()
    while len(problems) < n:
        N = rng.randint(6, 60)
        x = rng.randint(1, N - 1)
        if (x, N) in seen: continue
        seen.add((x, N))
        g = gcd(x, N)
        order = N // g
        instr = f"Find the order of the element {x} in the group (Z_{N}, +)."
        out = (
            f"The order of an element x in (Z_n, +) is n/gcd(x, n).\n"
            f"gcd({x}, {N}) = {g}\n"
            f"Order = {N}/{g} = {order}\n\n"
            f"The answer is {order}."
        )
        problems.append(make_entry(instr, out, "abstract_algebra"))
    return problems

def gen_num_generators(rng, n=45):
    """How many generators does Z_n have?"""
    problems = []
    seen = set()
    while len(problems) < n:
        N = rng.randint(4, 80)
        if N in seen: continue
        seen.add(N)
        phi = euler_totient(N)
        fac = factorize(N)
        fac_str = " × ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(fac.items()))
        instr = f"How many generators does the cyclic group Z_{N} have?"
        out = (
            f"The number of generators of Z_n equals Euler's totient φ(n).\n"
            f"{N} = {fac_str}\n"
            f"φ({N}) = {phi}\n\n"
            f"The answer is {phi}."
        )
        problems.append(make_entry(instr, out, "abstract_algebra"))
    return problems

def gen_lagrange(rng, n=50):
    """Can Z_N have a subgroup of order K?"""
    problems = []
    seen = set()
    while len(problems) < n:
        N = rng.randint(6, 60)
        if rng.random() < 0.5:
            # yes case: k divides N
            divs = [d for d in divisors(N) if 1 < d < N]
            if not divs: continue
            k = rng.choice(divs)
            ans = "Yes"
            reason = f"{k} divides {N} (since {N}/{k} = {N // k})"
        else:
            # no case: k does not divide N
            non_divs = [d for d in range(2, N) if N % d != 0]
            if not non_divs: continue
            k = rng.choice(non_divs)
            ans = "No"
            reason = f"{k} does not divide {N}"
        if (N, k) in seen: continue
        seen.add((N, k))
        instr = f"Can the group Z_{N} have a subgroup of order {k}?"
        out = (
            f"By Lagrange's theorem, the order of any subgroup must divide the order of the group.\n"
            f"|Z_{N}| = {N}\n"
            f"Check: does {k} divide {N}? {reason}.\n\n"
            f"The answer is {ans}."
        )
        problems.append(make_entry(instr, out, "abstract_algebra"))
    return problems

def gen_is_field(rng, n=40):
    """Is Z_n a field?"""
    problems = []
    seen = set()
    while len(problems) < n:
        if rng.random() < 0.5:
            # prime
            primes = [p for p in range(2, 100) if is_prime(p)]
            N = rng.choice(primes)
        else:
            # composite
            composites = [c for c in range(4, 100) if not is_prime(c)]
            N = rng.choice(composites)
        if N in seen: continue
        seen.add(N)
        prime = is_prime(N)
        ans = "Yes" if prime else "No"
        if prime:
            reason = f"{N} is prime, so every nonzero element has a multiplicative inverse"
        else:
            fac = factorize(N)
            fac_str = " × ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(fac.items()))
            reason = f"{N} = {fac_str} is composite, so it has zero divisors"
        instr = f"Is Z_{N} a field?"
        out = (
            f"Z_n is a field if and only if n is prime.\n"
            f"{reason}.\n\n"
            f"The answer is {ans}."
        )
        problems.append(make_entry(instr, out, "abstract_algebra"))
    return problems

def gen_direct_product_order(rng, n=50):
    """Order of (a,b) in Z_m × Z_n."""
    problems = []
    seen = set()
    while len(problems) < n:
        m = rng.randint(4, 30)
        nn = rng.randint(4, 30)
        a = rng.randint(1, m - 1)
        b = rng.randint(1, nn - 1)
        if (m, nn, a, b) in seen: continue
        seen.add((m, nn, a, b))
        ord_a = m // gcd(a, m)
        ord_b = nn // gcd(b, nn)
        order = lcm(ord_a, ord_b)
        instr = f"What is the order of the element ({a}, {b}) in the group Z_{m} × Z_{nn}?"
        out = (
            f"The order of (a, b) in Z_m × Z_n is lcm(ord(a), ord(b)).\n"
            f"ord({a}) in Z_{m} = {m}/gcd({a},{m}) = {m}/{gcd(a,m)} = {ord_a}\n"
            f"ord({b}) in Z_{nn} = {nn}/gcd({b},{nn}) = {nn}/{gcd(b,nn)} = {ord_b}\n"
            f"lcm({ord_a}, {ord_b}) = {order}\n\n"
            f"The answer is {order}."
        )
        problems.append(make_entry(instr, out, "abstract_algebra"))
    return problems

def gen_isomorphism(rng, n=45):
    """Is Z_m × Z_n isomorphic to Z_{mn}?"""
    problems = []
    seen = set()
    while len(problems) < n:
        m = rng.randint(2, 20)
        nn = rng.randint(2, 20)
        if m == nn: continue
        key = (min(m, nn), max(m, nn))
        if key in seen: continue
        seen.add(key)
        g = gcd(m, nn)
        iso = g == 1
        ans = "Yes" if iso else "No"
        prod = m * nn
        if iso:
            reason = f"gcd({m}, {nn}) = 1, so Z_{m} × Z_{nn} ≅ Z_{prod}"
        else:
            reason = f"gcd({m}, {nn}) = {g} ≠ 1, so Z_{m} × Z_{nn} is not isomorphic to Z_{prod}"
        instr = f"Is Z_{m} × Z_{nn} isomorphic to Z_{prod}?"
        out = (
            f"Z_m × Z_n ≅ Z_{{mn}} if and only if gcd(m, n) = 1.\n"
            f"{reason}.\n\n"
            f"The answer is {ans}."
        )
        problems.append(make_entry(instr, out, "abstract_algebra"))
    return problems

def gen_elements_of_order(rng, n=50):
    """How many elements of order d does Z_n have?"""
    problems = []
    seen = set()
    while len(problems) < n:
        N = rng.randint(6, 60)
        d = rng.randint(2, N)
        if (N, d) in seen: continue
        seen.add((N, d))
        if N % d == 0:
            count = euler_totient(d)
            reason = f"{d} divides {N}, so elements of order {d} exist. Count = φ({d}) = {count}"
        else:
            count = 0
            reason = f"{d} does not divide {N}, so no elements of order {d} exist"
        instr = f"How many elements of order {d} does the group Z_{N} have?"
        out = (
            f"In Z_n, elements of order d exist iff d divides n. If so, there are φ(d) such elements.\n"
            f"n = {N}, d = {d}\n"
            f"{reason}.\n\n"
            f"The answer is {count}."
        )
        problems.append(make_entry(instr, out, "abstract_algebra"))
    return problems

def gen_quotient_order(rng, n=45):
    """Order of quotient group Z_n / <k>."""
    problems = []
    seen = set()
    while len(problems) < n:
        N = rng.randint(6, 48)
        k = rng.randint(2, N - 1)
        if (N, k) in seen: continue
        seen.add((N, k))
        g = gcd(k, N)
        subgroup_order = N // g
        quotient_order = g
        instr = f"What is the order of the quotient group Z_{N} / ⟨{k}⟩?"
        out = (
            f"The subgroup ⟨{k}⟩ in Z_{N} has order {N}/gcd({k},{N}) = {N}/{g} = {subgroup_order}.\n"
            f"By Lagrange, |Z_{N} / ⟨{k}⟩| = |Z_{N}| / |⟨{k}⟩| = {N}/{subgroup_order} = {quotient_order}.\n\n"
            f"The answer is {quotient_order}."
        )
        problems.append(make_entry(instr, out, "abstract_algebra"))
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
    p.add_argument("--out", default="rehearsal_bank/algebra_programmatic.jsonl")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    rng = random.Random(args.seed)
    all_problems = []

    generators = [
        ("element_order", gen_element_order),
        ("num_generators", gen_num_generators),
        ("lagrange", gen_lagrange),
        ("is_field", gen_is_field),
        ("direct_product_order", gen_direct_product_order),
        ("isomorphism", gen_isomorphism),
        ("elements_of_order", gen_elements_of_order),
        ("quotient_order", gen_quotient_order),
    ]

    for name, gen in generators:
        probs = gen(rng)
        print(f"  {name}: {len(probs)} problems")
        all_problems.extend(probs)

    rng.shuffle(all_problems)

    with open(args.out, "w") as f:
        for p in all_problems:
            f.write(json.dumps(p) + "\n")

    print(f"\nTotal: {len(all_problems)} algebra problems → {args.out}")

if __name__ == "__main__":
    main()
