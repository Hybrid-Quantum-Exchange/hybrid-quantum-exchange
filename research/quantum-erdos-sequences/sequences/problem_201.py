"""
Erdos problem #201 (erdosproblems.com) -- quantum-testable instance.

Metadata (from data/problems.yaml): open, no prize, tags
["additive combinatorics", "arithmetic progressions"], OEIS ids
["A003002", "A003003", "A003004", "A003005", "possible"].

A003002 is the sequence of van der Waerden-type numbers for 2 colors and
3-term arithmetic progressions: the least N such that EVERY 2-coloring of
{1, ..., N} contains a monochromatic 3-term arithmetic progression (AP-3).
Its first term is 9 (W(2,3) = 9), which is equivalent to the classical fact
that {1, ..., 8} CAN be 2-colored with no monochromatic 3-term AP, while
{1, ..., 9} cannot.

Classical property tested here (computed from first principles, not copied
from OEIS): for N = 8 (positions 0..7, colors in {0,1}), does there exist a
2-coloring with no monochromatic 3-term arithmetic progression? We first
brute-force enumerate all 2^8 = 256 colorings classically to find the exact
set of "good" (AP-3-free) colorings. Consistency with A003002(1) = 9 requires
this good set to be non-empty (since W(2,3) = 9 > 8).

Quantum task: build a genuine Grover search circuit over the 8-qubit space
of colorings whose oracle marks exactly the good (AP-3-free) colorings (the
same set computed classically), run it on the ideal AerSimulator with the
Grover-optimal number of iterations, and check that the circuit's most
probable measured outcomes are indeed members of the classically-computed
good set. This directly exercises the finite combinatorial search problem
underlying A003002 for the smallest interesting case.

PASS/FAIL: PASS if (a) the classical good set is non-empty (confirms
W(2,3) > 8, consistent with A003002 = [9, ...]), and (b) the top measured
bitstrings from the Grover circuit are all members of that classical good
set with combined probability mass above a threshold.
"""

import itertools
from math import floor, pi, sqrt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N = 8  # {0, ..., 7} representing {1, ..., 8}


def three_term_aps(n):
    """All (i, j, k) with 0 <= i < j < k < n forming an arithmetic progression."""
    aps = []
    for i in range(n):
        for k in range(i + 2, n):
            if (i + k) % 2 == 0:
                j = (i + k) // 2
                if i < j < k:
                    aps.append((i, j, k))
    return aps


APS = three_term_aps(N)


def is_ap3_free(bits):
    """bits: tuple of 0/1 length N. True if no AP-3 is monochromatic."""
    for (i, j, k) in APS:
        if bits[i] == bits[j] == bits[k]:
            return False
    return True


def classical_good_set(n):
    good = []
    for bits in itertools.product([0, 1], repeat=n):
        if is_ap3_free(bits):
            good.append(bits)
    return good


def bits_to_int(bits):
    # bits[0] is qubit 0 (least significant in our convention)
    val = 0
    for idx, b in enumerate(bits):
        val |= (b << idx)
    return val


def build_oracle(n, good_ints):
    """Phase-flip oracle marking exactly the states in good_ints (list of ints)."""
    qc = QuantumCircuit(n, name="oracle")
    for val in good_ints:
        bits = [(val >> i) & 1 for i in range(n)]
        # flip qubits that are 0 in this target so an all-ones pattern lines up
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
        # multi-controlled Z on all n qubits (phase flip on |11...1>)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def main():
    good = classical_good_set(N)
    good_ints = sorted(bits_to_int(b) for b in good)
    M = len(good_ints)
    total = 2 ** N

    print(f"N = {N}, total colorings = {total}, AP-3-free colorings M = {M}")
    print(f"Confirms W(2,3) > {N} (A003002(1) = 9): {M > 0}")

    assert M > 0, "classical search space collapsed; A003002 property violated"

    # Optimal number of Grover iterations for M solutions out of 2^N states.
    theta = 2 * (sqrt(M / total)).real
    import math
    angle = math.asin(sqrt(M / total))
    iterations = max(1, floor((pi / 4) * sqrt(total / M)))

    oracle = build_oracle(N, good_ints)
    diffuser = build_diffuser(N)

    qc = QuantumCircuit(N, N)
    qc.h(range(N))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(N), range(N))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # qiskit bit-strings are big-endian in classical register order c[N-1..0]
    def outcome_to_int(bitstr):
        # bitstr like 'q7 q6 ... q0' -> reverse to match our bits_to_int convention
        return int(bitstr[::-1], 2)

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_k = min(M, 8)
    top_outcomes = sorted_counts[:top_k]

    hits = 0
    mass = 0
    for bitstr, c in top_outcomes:
        val = outcome_to_int(bitstr)
        mass += c
        if val in good_ints:
            hits += 1

    fraction_good = mass / shots
    all_top_valid = hits == len(top_outcomes)

    print(f"Grover iterations used: {iterations}")
    print(f"Top {top_k} measured outcomes all classically valid AP-3-free colorings: {all_top_valid}")
    print(f"Probability mass on those top outcomes: {fraction_good:.3f}")

    success = all_top_valid and fraction_good > 0.15

    print("PASS" if success else "FAIL")
    return success


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
