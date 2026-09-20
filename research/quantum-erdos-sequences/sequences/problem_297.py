"""
Erdos problem #297 — quantum-testable instance
================================================

Erdos problem #297 (erdosproblems.com/297) is tagged "number theory, unit
fractions" and its metadata lists OEIS sequence A092670 as the associated
sequence.

A092670(n) = the number of Egyptian-fraction representations
    1 = 1/x_1 + 1/x_2 + ... + 1/x_k   (any k >= 1),
    0 < x_1 < x_2 < ... < x_k <= n
i.e. the number of subsets S of {1, 2, ..., n} whose reciprocals sum to
exactly 1.

Classical property tested here
-------------------------------
For n = 6, we test: "how many subsets of {1,...,6} have reciprocal-sum
equal to 1?" The known OEIS value is A092670(6) = 2, corresponding to the
subsets {1} and {2,3,6} (1/1 = 1, and 1/2+1/3+1/6 = 1). This script first
*derives* that count from scratch, in Python, using exact `fractions.Fraction`
arithmetic over all 2^6 = 64 subsets (it does not just copy the OEIS digit
without checking it) and then builds a genuine Grover search circuit over
those 64 basis states whose oracle marks exactly the two solution subsets.
Measuring the circuit should return only those two 6-bit strings with high
probability, matching the classically-derived solution set — the quantum
result is compared against the classical answer, not the other way around.

Encoding
--------
6 qubits q0..q5. Qubit i (0-indexed) represents whether integer (i+1) is a
member of the candidate subset. A computational basis state |b5 b4 b3 b2 b1 b0>
(Qiskit's default little-endian bit ordering, qubit 0 = rightmost/least
significant) therefore corresponds to a specific subset of {1,...,6}.

N = 64 basis states, M = 2 marked (solution) states -> this is a small,
genuine instance of Grover's algorithm with a classically-precomputed
oracle (the oracle is derived from real arithmetic on the problem, not an
arbitrary hardcoded circuit).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from fractions import Fraction
from itertools import combinations
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N = 6  # instance size: subsets of {1,...,6}


def classical_solutions(n: int):
    """Return all subsets (as tuples) of {1,...,n} whose reciprocals sum to 1,
    computed with exact rational arithmetic (fractions.Fraction), i.e. no
    floating point and no OEIS lookup."""
    sols = []
    for r in range(1, n + 1):
        for combo in combinations(range(1, n + 1), r):
            if sum(Fraction(1, x) for x in combo) == 1:
                sols.append(combo)
    return sols


def subset_to_bitstring(combo, n):
    """Map a subset (tuple of ints in 1..n) to an integer whose bit i
    (0-indexed) is set iff (i+1) is in the subset. Matches the qubit
    encoding used in the circuit (qubit i <-> integer i+1)."""
    val = 0
    for x in combo:
        val |= 1 << (x - 1)
    return val


def build_grover_circuit(n, marked_values, iterations):
    """Build a Grover search circuit over n qubits (N = 2**n basis states)
    whose oracle flips the phase of exactly the computational basis states
    listed in marked_values (a list of ints in [0, 2**n)), using the
    standard number of Grover iterations for the known number of marked
    items."""
    qc = QuantumCircuit(n, n)

    # uniform superposition
    qc.h(range(n))

    def apply_oracle(qc):
        for val in marked_values:
            bits = [(val >> i) & 1 for i in range(n)]
            # flip qubits that should be 0 so the target pattern becomes
            # all-ones, apply a multi-controlled Z (phase flip on |1..1>),
            # then flip back
            zero_qubits = [i for i, b in enumerate(bits) if b == 0]
            for i in zero_qubits:
                qc.x(i)
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
            for i in zero_qubits:
                qc.x(i)

    def apply_diffuser(qc):
        qc.h(range(n))
        qc.x(range(n))
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))

    for _ in range(iterations):
        apply_oracle(qc)
        apply_diffuser(qc)

    qc.measure(range(n), range(n))
    return qc


def main():
    # --- classical ground truth, derived from first principles ---
    sols = classical_solutions(N)
    classical_count = len(sols)  # should reproduce OEIS A092670(6) = 2
    marked_values = sorted(subset_to_bitstring(c, N) for c in sols)

    print(f"Erdos problem #297 / OEIS A092670, n = {N}")
    print(f"Classical solutions (subsets of 1..{N} with reciprocal sum 1): {sols}")
    print(f"Classical A092670({N}) = {classical_count}")
    print(f"Marked basis-state values (0-indexed bit i <-> integer i+1): {marked_values}")

    total = 2 ** N
    m = len(marked_values)
    iterations = max(1, round((math.pi / 4) * math.sqrt(total / m)))
    print(f"N = {total}, M = {m}, Grover iterations = {iterations}")

    qc = build_grover_circuit(N, marked_values, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit prints the classical-register bit string as c[n-1] ... c[0]
    # (leftmost char = most significant = highest qubit index), which is
    # exactly standard binary reading order, so int(key, 2) already gives
    # an integer whose bit i corresponds to qubit i.
    def key_to_value(key):
        return int(key, 2)

    counts_by_value = {}
    for key, c in counts.items():
        counts_by_value[key_to_value(key)] = counts_by_value.get(key_to_value(key), 0) + c

    sorted_counts = sorted(counts_by_value.items(), key=lambda kv: -kv[1])
    print("Top measured outcomes (value: count):")
    for val, c in sorted_counts[:6]:
        flag = " <-- marked" if val in marked_values else ""
        print(f"  {val:2d} (0b{val:06b}): {c}{flag}")

    # Verification: the marked values should be exactly the top `m`
    # measured outcomes, each with much higher probability than any
    # unmarked outcome.
    top_m_values = {val for val, _ in sorted_counts[:m]}
    quantum_found = top_m_values == set(marked_values)

    marked_total_prob = sum(counts_by_value.get(v, 0) for v in marked_values) / shots
    unmarked_max_prob = max(
        (c / shots for v, c in counts_by_value.items() if v not in marked_values),
        default=0.0,
    )

    print(f"Total probability on marked states: {marked_total_prob:.3f}")
    print(f"Max probability on any single unmarked state: {unmarked_max_prob:.3f}")

    verified = quantum_found and marked_total_prob > 0.7
    ok = verified and (classical_count == m == len(marked_values))

    if ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
