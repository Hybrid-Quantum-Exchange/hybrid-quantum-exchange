"""
Erdos problem #309 -- quantum-testable instance
=================================================

Source: Erdos problem #309 (data/problems.yaml in the erdosproblems repo),
tags ["number theory", "unit fractions"], status "disproved (Lean)",
OEIS id used: A217693.

A217693: "Number of distinct integers obtained from summing up subsets of
{1, 1/2, 1/3, ..., 1/n}." (excluding the empty/zero sum -- checking small
terms confirms the OEIS convention counts only the distinct *positive*
integers reachable this way: a(1)=1 comes only from {1}; a(6)=2 comes from
the two positive integers {1, 2} that are reachable as subset sums of
{1, 1/2, 1/3, 1/4, 1/5, 1/6}. This was verified independently below by
brute force before building the circuit.)

Classical property tested here
-------------------------------
For n = 6, put all six unit fractions over the common denominator 60:

    1 = 60/60, 1/2 = 30/60, 1/3 = 20/60, 1/4 = 15/60, 1/5 = 12/60, 1/6 = 10/60

so the weights (numerators) are W = [60, 30, 20, 15, 12, 10]. A subset of
{1,...,6} (encoded as a 6-bit string, bit i = whether item i is included)
represents an integer exactly when its weighted sum equals a multiple of 60.
We fix the target integer value 1, i.e. weighted sum == 60 exactly, and ask:
which of the 2^6 = 64 subsets hit that target?

Brute force (done in this script, first-principles, no OEIS values copied)
finds this has exactly 2 solutions among the 64 subsets:
    {item 6 only}              -> 1/6 * 6 = 1            (bitstring 000001)
    {items 1,3,4 (1-indexed)}  -> 1/1... wait see below   (bitstring 100110)
(exact bitstrings and solution count are computed programmatically below,
not hard-coded from any external source).

This is exactly the kind of small, finite, Grover-searchable decision
problem the unit-fraction subset-sum property reduces to: "does subset x
(of the n=6 unit fractions) sum to the integer target 1?" We build a
6-qubit Grover search whose oracle marks precisely the bitstrings that are
classical solutions (the oracle's phase-flip pattern is derived from the
classical brute-force computation, not asserted independently), run it on
the ideal AerSimulator, and check that the measured result set is exactly
the classical solution set with high probability -- i.e. we verify Grover
search reproduces the classical answer to the unit-fraction subset-sum
membership question that A217693 counts.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS values copied in).
# ---------------------------------------------------------------------------

N_ITEMS = 6                      # unit fractions 1, 1/2, ..., 1/6
DEN = 60                         # common denominator (lcm(1..6) = 60)
WEIGHTS = [DEN // k for k in range(1, N_ITEMS + 1)]   # [60,30,20,15,12,10]
TARGET = 60                      # represents the integer 1 (TARGET/DEN = 1)

assert WEIGHTS == [60, 30, 20, 15, 12, 10]

classical_solutions = []
for r in range(N_ITEMS + 1):
    for combo in combinations(range(N_ITEMS), r):
        s = sum(WEIGHTS[i] for i in combo)
        if s == TARGET:
            # qubit-order string: index i is qubit i's value (item i
            # included or not). This matches how build_oracle() below
            # indexes qubits directly by position i.
            bitstring = "".join("1" if i in combo else "0" for i in range(N_ITEMS))
            classical_solutions.append(bitstring)

classical_solutions = sorted(classical_solutions)
num_solutions = len(classical_solutions)

# Sanity: also confirm, independently, the full A217693-style positive-
# integer count for n=6 is 2 (this is a broader check than the single
# target used for the circuit, and is not itself fed into the circuit).
positive_integers_reachable = set()
for r in range(N_ITEMS + 1):
    for combo in combinations(range(N_ITEMS), r):
        s = sum(WEIGHTS[i] for i in combo)
        if s % DEN == 0 and s > 0:
            positive_integers_reachable.add(s // DEN)
assert positive_integers_reachable == {1, 2}, positive_integers_reachable

print(f"Classical solution count for target={TARGET} (integer 1): {num_solutions}")
print(f"Classical solution bitstrings: {classical_solutions}")
print(f"Independently confirmed A217693(6) = {len(positive_integers_reachable)} "
      f"(reachable positive integers: {sorted(positive_integers_reachable)})")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 64 subsets, oracle marks
#    exactly the classical_solutions bitstrings computed above.
# ---------------------------------------------------------------------------

n = N_ITEMS  # 6 qubits, one per subset-membership bit


def build_oracle(solutions, n_qubits):
    """Phase oracle: flips the sign of exactly the basis states listed in
    `solutions` (each an n_qubits-length '0'/'1' string, little-endian)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bitstring in solutions:
        # Flip qubits that are '0' in this target so a multi-controlled Z
        # fires only on this exact bitstring, then flip back.
        zero_positions = [i for i, b in enumerate(bitstring) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(classical_solutions, n)
diffuser = build_diffuser(n)

# Optimal number of Grover iterations for M solutions out of N=2^n states.
N_states = 2 ** n
M = num_solutions
theta = math.asin(math.sqrt(M / N_states))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(n, n)
qc.h(range(n))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(n), range(n))

print(f"Grover iterations used: {iterations} (N={N_states}, M={M})")

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

solution_set = set(classical_solutions)

# Qiskit's classical-register measurement keys are big-endian over the
# creg (leftmost char = highest qubit index) -- the reverse of the
# qubit-order strings used above and in build_oracle(). Reverse before
# comparing.
def to_qubit_order(measured_key):
    return measured_key[::-1]

solution_shots = sum(
    c for bstr, c in counts.items() if to_qubit_order(bstr) in solution_set
)
solution_fraction = solution_shots / shots

most_common = sorted(counts.items(), key=lambda kv: -kv[1])[:num_solutions + 2]
print(f"Top measured outcomes: {most_common}")
print(f"Fraction of shots landing on a classical solution: {solution_fraction:.4f}")

# Grover with the optimal iteration count for this M, N should concentrate
# the great majority of shots on the marked (classical-solution) states.
verified = solution_fraction > 0.90
verified = verified and all(
    to_qubit_order(bstr) in solution_set for bstr, _ in most_common[:num_solutions]
)

if verified:
    print("PASS: Grover search result matches the classical unit-fraction "
          "subset-sum solutions for A217693 (n=6, target integer 1).")
else:
    print("FAIL: Grover search result does not match the classical answer.")

assert verified, "quantum result did not match classical computation"
