"""
Erdos problem #747 -- quantum-testable lane.

Source: data/problems.yaml entry for number "747" (Erdos Problems database,
github.com/manman4/erdosproblems, checked 2026-09-19):

    number: "747"
    prize: "no"
    status: solved (2025-08-31)
    oeis: ["N/A"]
    tags: ["combinatorics", "hypergraphs"]

LIMITATION (read before trusting the "sequence" framing): problem #747 has
NO associated OEIS sequence id in the source data (oeis: ["N/A"]). There is
therefore no genuine "sequence membership" property of #747 itself to encode
in a circuit -- building one would mean fabricating content the problem does
not have, which the task explicitly forbids. This script is an honest
best-attempt substitute: it stays faithful to problem #747's *tags*
("combinatorics", "hypergraphs") by testing a real, small, finite,
classically-checkable hypergraph combinatorics property -- a minimum hitting
set (transversal) of a fixed hypergraph -- with a genuine Grover search
circuit. This is NOT a claim that #747 reduces to this instance; it is a
disclosed substitution because #747 itself offers no computable sequence
property to test.

Classical property being tested
--------------------------------
Fix a 5-vertex hypergraph H on V = {0,1,2,3,4} with hyperedges:

    E1 = {0,1,2}
    E2 = {2,3}
    E3 = {3,4}
    E4 = {0,4}

A "hitting set" (transversal) of H is a subset S of V that intersects every
edge in E1..E4. We ask: which subsets S with |S| = 2 (5 qubits -> search
space of 2^5 = 32 bitstrings, restricted by an |S|=2 penalty-free oracle
that simply marks the *hitting* size-2 sets) are hitting sets?

The classical answer is computed in this script from first principles by
brute-force enumeration over all 32 bitstrings (no OEIS lookup, no literature
value copied) -- see `classical_min_hitting_sets()`.

Quantum approach
-----------------
A Grover search over 5 qubits (32-item search space). The oracle is built
directly from the classically-enumerated set of marked bitstrings (the
size-2 hitting sets): each marked bitstring gets a multi-controlled-Z phase
flip (X-sandwiched around the 0-bits of that string), which is a standard,
literal way to realize "the classically known Boolean predicate" as a phase
oracle -- not a shortcut that bypasses computation, since the predicate
itself (which sets are hitting sets) is computed classically here and then
compiled into gates. Grover's diffusion operator is applied for
floor(pi/4 * sqrt(N/M)) iterations, and the ideal AerSimulator is sampled;
the circuit passes if the measured distribution is concentrated (top-M
outcomes carry the large majority of probability) on exactly the classically
computed set of marked bitstrings.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no external values copied)
# ---------------------------------------------------------------------------

N_VERTICES = 5
EDGES = [frozenset({0, 1, 2}), frozenset({2, 3}), frozenset({3, 4}), frozenset({0, 4})]


def is_hitting_set(bits, edges=EDGES):
    """bits: tuple of 0/1 of length N_VERTICES, index i -> vertex i in S."""
    s = {i for i, b in enumerate(bits) if b == 1}
    return all(len(s & e) > 0 for e in edges)


def classical_min_hitting_sets():
    """Brute force over all 2**N_VERTICES bitstrings; return the size-2
    hitting sets (the minimum hitting-set size for this H, verified below)."""
    all_bits = [tuple((n >> i) & 1 for i in range(N_VERTICES)) for n in range(2 ** N_VERTICES)]

    # Confirm no size-1 hitting set exists, so 2 is genuinely the minimum.
    size1 = [b for b in all_bits if sum(b) == 1 and is_hitting_set(b)]
    assert size1 == [], f"expected no size-1 hitting set, found {size1}"

    size2 = sorted(b for b in all_bits if sum(b) == 2 and is_hitting_set(b))
    assert size2, "expected at least one size-2 hitting set"
    return size2


CLASSICAL_HITTING_SETS = classical_min_hitting_sets()  # list of 5-bit tuples (LSB = vertex 0)
MARKED_INTS = sorted(int("".join(str(b) for b in reversed(bits)), 2) for bits in CLASSICAL_HITTING_SETS)
N_QUBITS = N_VERTICES
N_STATES = 2 ** N_QUBITS
M = len(MARKED_INTS)

print(f"Classical size-2 hitting sets of H (vertex subsets): "
      f"{[tuple(i for i, b in enumerate(bits) if b == 1) for bits in CLASSICAL_HITTING_SETS]}")
print(f"Marked integers (bitstring encodings): {MARKED_INTS}  (M={M} out of N={N_STATES})")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for the marked (hitting-set) bitstrings
# ---------------------------------------------------------------------------

def apply_oracle(qc, qubits, marked_int, n):
    """Phase-flip the basis state equal to marked_int (n-bit, little-endian
    on `qubits`), using an X-sandwiched multi-controlled Z."""
    bits = [(marked_int >> i) & 1 for i in range(n)]
    flip_qubits = [qubits[i] for i, b in enumerate(bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in flip_qubits:
        qc.x(q)


def diffusion(qc, qubits, n):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(n, marked_ints):
    qc = QuantumCircuit(n, n)
    qubits = list(range(n))
    qc.h(qubits)

    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N_STATES / len(marked_ints)))))
    for _ in range(iterations):
        for mi in marked_ints:
            apply_oracle(qc, qubits, mi, n)
        diffusion(qc, qubits, n)

    qc.measure(qubits, qubits)
    return qc, iterations


circuit, num_iterations = build_grover_circuit(N_QUBITS, MARKED_INTS)
print(f"Grover iterations used: {num_iterations}")

sim = AerSimulator()
shots = 4096
result = sim.run(circuit, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-first over the classical register (which was
# filled little-endian from qubits[0..n-1]); convert each key back to an int
# matching our little-endian `marked_int` convention.
def bitstring_to_int(key):
    # Qiskit's classical-register string is MSB-first over bit index, i.e.
    # key[0] is classical bit n-1 and key[-1] is classical bit 0. Since
    # measure(qubits, qubits) mapped qubit i -> classical bit i, the value
    # int(key, 2) already equals sum_i qubit_i * 2**i, matching the
    # little-endian `marked_int` convention used to build the oracle.
    return int(key, 2)


int_counts = {}
for key, c in counts.items():
    int_counts[bitstring_to_int(key)] = int_counts.get(bitstring_to_int(key), 0) + c

sorted_outcomes = sorted(int_counts.items(), key=lambda kv: -kv[1])
top_m = set(k for k, _ in sorted_outcomes[:M])
top_m_probability = sum(c for k, c in int_counts.items() if k in set(MARKED_INTS)) / shots

print(f"Top-{M} most frequent measured outcomes: {sorted(top_m)}")
print(f"Expected marked outcomes:                {sorted(MARKED_INTS)}")
print(f"Fraction of shots landing on a classically-marked (hitting-set) outcome: "
      f"{top_m_probability:.3f}")

# ---------------------------------------------------------------------------
# 3. Verify quantum result against classical answer
# ---------------------------------------------------------------------------

verified = (top_m == set(MARKED_INTS)) and (top_m_probability > 0.8)

if verified:
    print("PASS")
else:
    print("FAIL")
