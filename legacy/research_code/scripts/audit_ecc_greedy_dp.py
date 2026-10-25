"""Exact cost-state DP on the binary singular values of frozen ECC layers.

Run after MATLAB export_ecc_rank_allocation_inputs. This uses Python integers
for all energy feasibility, density, and dominance comparisons. Exactness is
relative to the saved floating-point singular values, not an exact SVD of E.
"""
from bisect import bisect_left
from fractions import Fraction
from functools import reduce
from pathlib import Path
import csv
import hashlib
import json
import math
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/greedy_dp_audit_20260920"


def binary_float(h):
    return struct.unpack(">d", bytes.fromhex(h))[0]


def integer_energies(record):
    spectra = [[Fraction.from_float(binary_float(h)) ** 2 for h in v]
               for v in record["svals_hex"]]
    for a in spectra:
        assert all(x >= y >= 0 for x, y in zip(a, a[1:]))
    total = sum(map(sum, spectra))
    tau = Fraction.from_float(binary_float(record["eta_hex"])) ** 2 * total
    # IEEE binary values and their products have power-of-two denominators.
    scale = max([tau.denominator] + [x.denominator for a in spectra for x in a])
    energies = [[int(x * scale) for x in a] for a in spectra]
    assert (tau * scale).denominator == 1
    return energies, int(tau * scale)


def tails_of(energies):
    tails = []
    for a in energies:
        t = [0] * (len(a) + 1)
        for r in range(len(a) - 1, -1, -1):
            t[r] = t[r + 1] + a[r]
        tails.append(t)
    return tails


def greedy(energies, n, tau):
    tails = tails_of(energies)
    r = [0] * len(n)
    last = None
    while sum(t[r[b]] for b, t in enumerate(tails)) > tau:
        active = [b for b in range(len(n)) if r[b] < len(energies[b])]
        assert active, "Infeasible energy budget"
        best = active[0]
        for b in active[1:]:
            if energies[b][r[b]] * n[best] > energies[best][r[best]] * n[b]:
                best = b
        last = (best, energies[best][r[best]])
        r[best] += 1
    return r, last


def cost_dp(tails, n, tau, upper):
    """Sparse exact-cost DP storing the minimum sum of discarded tails.

    Retain the best state for each integer cost, then discard dominated states.
    Partial tails > tau and costs > the greedy upper bound cannot be optimal.
    """
    g = reduce(math.gcd, n)
    costs = [nb // g for nb in n]
    cap = upper // g
    states = {0: (0, ())}
    counts = []
    for b, t in enumerate(tails):
        candidates = [(r, tail) for r, tail in enumerate(t) if tail <= tau]
        new = {}
        for c, (value, ranks) in states.items():
            for r, tail in candidates:
                nc = c + costs[b] * r
                if nc > cap:
                    break
                nt = value + tail
                if nt > tau:
                    continue
                if nc not in new or nt < new[nc][0]:
                    new[nc] = (nt, ranks + (r,))
        states = {}
        best_tail = None
        for c in sorted(new):
            value = new[c]
            if best_tail is None or value[0] < best_tail:
                states[c] = value
                best_tail = value[0]
        assert states
        counts.append(len(states))
    c = min(states)
    return c * g, list(states[c][1]), counts


def enumeration_three(tails, n, tau):
    """Independent check: enumerate first two ranks; binary search the third."""
    best = None
    third = [-t for t in tails[2]]
    for r0, t0 in enumerate(tails[0]):
        if t0 > tau:
            continue
        for r1, t1 in enumerate(tails[1]):
            remaining = tau - t0 - t1
            if remaining < 0:
                continue
            r2 = bisect_left(third, -remaining)
            if r2 == len(third):
                continue
            cost = n[0] * r0 + n[1] * r1 + n[2] * r2
            if best is None or cost < best:
                best = cost
    return best


def self_check():
    # Known greedy failure; a tiny-tail case; and deterministic random cases.
    import random
    cases = [([[5], [3]], [3, 2], 5),
             ([[10**40, 1], [2]], [7, 2], 0)]
    rng = random.Random(20260920)
    for _ in range(100):
        a = [sorted([rng.randrange(1, 40) for _ in range(4)], reverse=True)
             for _ in range(3)]
        cases.append((a, [rng.randrange(1, 10) for _ in range(3)],
                      rng.randrange(sum(map(sum, a)) + 1)))
    for a, n, tau in cases:
        r, _ = greedy(a, n, tau)
        upper = sum(x*y for x, y in zip(n, r))
        cost, _, _ = cost_dp(tails_of(a), n, tau, upper)
        if len(n) == 3:
            assert cost == enumeration_three(tails_of(a), n, tau)
        if a == [[5], [3]]:
            assert upper == 3 and cost == 2
    print("DP_SELF_CHECK_PASS cases=102")


def main():
    self_check()
    src = OUT / "inputs.json"
    payload = json.loads(src.read_text())
    results = []
    for record in payload["records"]:
        a, tau = integer_energies(record)
        n = list(map(int, record["n"]))
        tails = tails_of(a)
        total = sum(t[0] for t in tails)
        rg, last = greedy(a, n, tau)
        assert rg == record["saved_ranks"], (record["tag"], record["layer"], rg)
        sg = sum(x*y for x, y in zip(n, rg))
        started = time.perf_counter()
        sd, rd, counts = cost_dp(tails, n, tau, sg)
        dp_seconds = time.perf_counter() - started
        assert sd == enumeration_three(tails, n, tau)
        dg = sum(t[rg[b]] for b, t in enumerate(tails))
        dd = sum(t[rd[b]] for b, t in enumerate(tails))
        assert dg <= tau and dd <= tau
        if last is None:
            bound, slp, last_n = Fraction(0), Fraction(0), 0
        else:
            b, energy = last
            last_n = n[b]
            bound = Fraction(n[b] * (tau-dg), energy)
            slp = sg - bound
            assert 0 <= sg-sd <= bound < n[b]
        g = reduce(math.gcd, n)
        lb = g * math.ceil(slp / g)
        assert lb <= sd <= sg
        result = dict(tag=record["tag"], layer=record["layer"], eta=record["eta"],
            n=n, greedy_ranks=rg, dp_ranks=rd, greedy_entries=sg, dp_entries=sd,
            saved_entries=sg-sd, savings_percent=100*(sg-sd)/sg if sg else 0,
            greedy_tail=math.sqrt(Fraction(dg, total)),
            dp_tail=math.sqrt(Fraction(dd, total)),
            lp_lower_bound=float(slp), gcd_lower_bound=lb,
            gap_bound_entries=float(bound), last_block_size=last_n,
            dp_frontier_counts=counts, dp_seconds=dp_seconds,
            dp_budget_fraction=float(Fraction(dd, tau)) if tau else 0,
            source=record["source"])
        results.append(result)
        print(f"{result['tag']} L{result['layer']} {rg} -> {rd}: "
              f"{sg} -> {sd} ({result['savings_percent']:.4f}%), "
              f"bound {float(bound):.2f}, DP {dp_seconds:.4f}s", flush=True)
    report = dict(scope=payload["scope"],
        arithmetic="Exact integer comparisons after squaring saved binary singular values exactly",
        input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        validation="102 self-checks; all saved ranks reproduced; all 20 layer optima matched independent enumeration",
        results=results)
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print("ECC_GREEDY_DP_AUDIT_PASS")


if __name__ == "__main__":
    main()
