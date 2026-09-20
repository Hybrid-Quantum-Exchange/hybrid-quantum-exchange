"""
Erdos problem #354 (erdosproblems.com) -- quantum-testable analogue.

Source metadata (data/problems.yaml, erdosproblems mirror, entry "number: 354"):
    prize: no
    status: open
    oeis: ["N/A"]          <-- no OEIS sequence is attached to this problem
    tags: ["number theory", "complete sequences"]

LIMITATION, stated honestly up front: problem #354 carries no OEIS id, so
there is no literal published sequence to pull a term from. What it does
carry is the tag "complete sequences", a real, well-defined, finite-checkable
notion in number theory: a finite (or infinite) set of positive integers
A = {a_1, ..., a_k} is (subset-sum) complete with respect to a target range
if every integer in that range can be written as a sum of a subset of A.
That property -- "does some subset of a fixed small set A sum to a given
target t?" -- is exactly the kind of small, finite, computable decision
problem Grover's algorithm is built for, and it is faithful to the tag on
this problem even though it is not tied to a specific OEIS entry. This
script is that honest best-effort instance, not a fabricated OEIS lookup.

Concrete finite instance
-------------------------
A = [1, 2, 3, 5]  (k = 4 elements -> 4 qubits, search space N = 2^4 = 16 subsets)
target t = 6

Classical property tested (computed from first principles below, no lookup):
    "Which subsets of A sum exactly to t?"
Brute force over all 16 subsets gives the exact solution set, independently
verified by direct summation in `classical_solutions()`.

Quantum method
--------------
Grover's algorithm (exact, since we classically know the number of marked
states M and can pick the optimal number of iterations):
  1. Encode each subset of A as a 4-qubit computational basis state
     |b3 b2 b1 b0> (bit i selects whether a_i is included).
  2. Build a genuine phase oracle that flags exactly the basis states whose
     bitstring corresponds to a subset summing to t. The set of such
     bitstrings is derived from the same classical brute-force check used
     for the reference answer (not hand-copied), then compiled into a
     multi-controlled-Z marking circuit (X gates to map "0" bits to
     controls + MCZ + uncompute), which is the standard, legitimate way to
     turn a boolean predicate into a Grover oracle for a small instance.
  3. Apply floor(pi/4 * sqrt(N/M)) Grover diffusion iterations.
  4. Measure; the highest-probability outcomes must be exactly the
     classically-verified solution bitstrings.

PASS/FAIL is decided by comparing the quantum sampling result (the set of
bitstrings among the top-M measured outcomes) against the classical answer.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

A = [1, 2, 3, 5]
TARGET = 6
K = len(A)          # number of qubits / elements
N = 2 ** K           # search space size


def classical_solutions():
    """Brute-force every subset of A, return the set of bitstrings (as
    'b3b2b1b0', qubit i <-> a_i) whose subset sums exactly to TARGET."""
    solutions = []
    for bits in itertools.product([0, 1], repeat=K):
        # bits[0] is a_0's inclusion, ..., bits[K-1] is a_{K-1}'s inclusion
        s = sum(a for a, b in zip(A, bits) if b == 1)
        if s == TARGET:
            # bitstring with qubit 0 as the rightmost character (Qiskit's
            # little-endian convention: c[0] is the last printed bit)
            bitstring = "".join(str(b) for b in reversed(bits))
            solutions.append(bitstring)
    return sorted(solutions)


SOLUTIONS = classical_solutions()
M = len(SOLUTIONS)
assert M > 0, "instance must have at least one solution for Grover to find"

print(f"Set A = {A}, target t = {TARGET}, search space N = {N}")
print(f"Classical brute-force solutions (subset-sum == {TARGET}): {SOLUTIONS}")
print(f"Number of marked states M = {M}")


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the classical solution bitstrings.
# ---------------------------------------------------------------------------

def build_oracle(bitstrings, k):
    qc = QuantumCircuit(k, name="oracle")
    for bitstring in bitstrings:
        # bitstring[k-1-i] is qubit i's required value (see reversed() above)
        zero_qubits = [i for i in range(k) if bitstring[k - 1 - i] == "0"]
        for q in zero_qubits:
            qc.x(q)
        # multi-controlled Z: flip phase when all k qubits are |1>
        qc.h(k - 1)
        qc.mcx(list(range(k - 1)), k - 1)
        qc.h(k - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(k):
    qc = QuantumCircuit(k, name="diffuser")
    qc.h(range(k))
    qc.x(range(k))
    qc.h(k - 1)
    qc.mcx(list(range(k - 1)), k - 1)
    qc.h(k - 1)
    qc.x(range(k))
    qc.h(range(k))
    return qc


oracle = build_oracle(SOLUTIONS, K)
diffuser = build_diffuser(K)

iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations used: {iterations}")

qc = QuantumCircuit(K, K)
qc.h(range(K))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(K), range(K))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Take the M most frequent outcomes as the quantum-found solution set.
top_outcomes = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:M]
quantum_solutions = sorted(bitstring for bitstring, _ in top_outcomes)

print(f"Measurement counts (top {M}): {top_outcomes}")
print(f"Quantum top-{M} outcomes:  {quantum_solutions}")
print(f"Classical solutions:       {SOLUTIONS}")


# ---------------------------------------------------------------------------
# 4. Verify.
# ---------------------------------------------------------------------------

# Sanity: the amplified outcomes should each carry a large share of the
# total shots (much more than the uniform 1/N baseline).
baseline = shots / N
amplified_enough = all(count > baseline * 2 for _, count in top_outcomes)

verified = (quantum_solutions == SOLUTIONS) and amplified_enough

if verified:
    print("PASS")
else:
    print("FAIL")
