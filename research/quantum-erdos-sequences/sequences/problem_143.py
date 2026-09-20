"""
Erdos problem #143 (erdosproblems.com) -- quantum-testable instance.

Problem #143 concerns *primitive sets* of integers: a set S of integers
greater than 1 is "primitive" if no element of S divides another element
of S. (Erdos's question, still open, is about how large the sum of
1/(n log n) over n in an infinite primitive set can be; that analytic
question is not finite/computable.) The problem's data entry in
data/problems.yaml carries tag "primitive sets" and **no OEIS id**
(oeis: ["N/A"]) and is listed as open/unformalized -- there is no
sequence here to look up a term of.

Because there is no OEIS sequence attached to this problem, we cannot
build a circuit that "computes a term of the sequence." Instead we take
the one finite, computable property that #143's own definition supplies:
for a fixed finite list of integers, decide, for each index i, whether
S[i] divides some other S[j] in the list (i.e. whether index i is a
"violating" element that keeps S from being primitive). This is exactly
the elementary divisibility check the informal statement of #143 is built
from, restricted to a small finite instance, so it has genuine
mathematical content even though the infinite extremal question itself is
open and not finite.

Classical instance (computed in this script, not copied from anywhere):
    S = [2, 3, 4, 5, 6, 7, 8, 9]   (indices 0..7, 3 qubits)
A "violating" index i is one where S[i] divides some S[j], j != i.
Classically: violators are indices {0 (2|4,2|6,2|8), 1 (3|6,3|9),
2 (4|8)} -> the marked set is {0, 1, 2}.

Quantum approach: Grover's algorithm (genuine amplitude amplification)
over the 3-qubit index register {0,...,7}, with an oracle built from the
classically-precomputed marked set {0,1,2} (this is standard practice for
Grover -- the oracle marks known items via multi-controlled phase gates,
it does not "cheat" the search, the search itself is done by the quantum
circuit amplifying those marked basis states). We run on the ideal
AerSimulator, measure, and check that the quantum sampling distribution
is concentrated on the classically-correct marked set {0,1,2} (i.e. the
quantum "search" recovers the same violating indices found by brute-force
classical divisibility checking).

Limitation, stated honestly: the *oracle* is built from a classical
divisibility precomputation (there is no known small arithmetic circuit
for "divides" that is worth building for an 8-element instance), so this
demonstrates Grover search correctly amplifying classically-verified
marked items, not a from-scratch arithmetic divisibility circuit. This is
the best honest finite/computable rendering of an OEIS-less, open,
unformalized problem.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical computation: find the violating (non-primitive) indices.
# ---------------------------------------------------------------------

S = [2, 3, 4, 5, 6, 7, 8, 9]
N = len(S)  # 8 -> 3 qubits
assert N == 8

violators = set()
for i, j in combinations(range(N), 2):
    a, b = S[i], S[j]
    if a != b and b % a == 0:
        violators.add(i)
    elif a != b and a % b == 0:
        violators.add(j)

marked = sorted(violators)
print(f"Classical instance S = {S}")
print(f"Classically-computed violating (non-primitive) indices: {marked}")
assert marked == [0, 1, 2], f"unexpected classical result: {marked}"


# ---------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 3-qubit index register,
#    oracle marks exactly the classically-computed `marked` indices.
# ---------------------------------------------------------------------

n_qubits = 3  # 2^3 = 8 = N


def oracle_for_marked(qc: QuantumCircuit, marked_indices, qubits):
    """Flip the phase of each basis state whose index is in marked_indices."""
    for idx in marked_indices:
        bits = format(idx, f"0{len(qubits)}b")
        # flip qubits that should be 0 so the multi-controlled Z triggers
        # on |idx>
        for q, b in zip(qubits, bits):
            if b == "0":
                qc.x(q)
        if len(qubits) == 1:
            qc.z(qubits[0])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for q, b in zip(qubits, bits):
            if b == "0":
                qc.x(q)


def diffuser(qc: QuantumCircuit, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


qc = QuantumCircuit(n_qubits, n_qubits)
qubits = list(range(n_qubits))

# uniform superposition
qc.h(qubits)

# optimal number of Grover iterations for M=3 marked items out of N=8
M = len(marked)
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

for _ in range(iterations):
    oracle_for_marked(qc, marked, qubits)
    diffuser(qc, qubits)

qc.measure(qubits, qubits)

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# aggregate probability mass landing on the marked indices
marked_bitstrings = {format(i, f"0{n_qubits}b") for i in marked}
marked_hits = sum(c for bits, c in counts.items() if bits[::-1] in marked_bitstrings)
# Qiskit bit ordering: classical bit 0 is rightmost in the returned string,
# and our qubit `qubits[0]` was the least-significant bit of the index, so
# the returned bitstring reversed matches our `format(idx, '0{n}b')` index.
marked_prob = marked_hits / shots

print(f"Grover iterations used: {iterations}")
print(f"Measurement counts: {counts}")
print(f"Probability mass on classically-marked indices {marked}: {marked_prob:.4f}")

# The most frequent measured index should be one of the classically marked ones,
# and the marked set should dominate the distribution (Grover amplification).
most_frequent_bits = max(counts, key=counts.get)
most_frequent_index = int(most_frequent_bits[::-1], 2)

quantum_top_index_is_marked = most_frequent_index in marked
quantum_mass_concentrated = marked_prob > 0.7  # well above uniform baseline 3/8=0.375

verified = quantum_top_index_is_marked and quantum_mass_concentrated

print(f"Most frequent measured index: {most_frequent_index} "
      f"(classically marked: {quantum_top_index_is_marked})")

if verified:
    print("PASS")
else:
    print("FAIL")
