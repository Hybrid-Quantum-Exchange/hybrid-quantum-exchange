"""
Erdos problem #136 -- quantum-testable sequence entry.

Source metadata (erdosproblems.com data, data/problems.yaml, number "136"):
    prize: no
    status: solved (2025-08-31)
    oeis: ["possible"]        <- NOT a real OEIS id; it is a literal placeholder
                                  string used in that dataset, not an A-number.
    tags: ["graph theory"]

LIMITATION (reported honestly, per task instructions): problem 136 carries no
usable OEIS sequence id in the source data -- "possible" is a placeholder, not
an A-number -- and the dataset entry gives no formula, description, or term
list to derive a sequence-membership property from. There is therefore no
literal OEIS term to test against, and this script does NOT claim to test one.

Best-effort honest substitute: since the problem's only real content is its
tag "graph theory", this script builds a genuine, small, finite, computable
graph-theory decision property -- "does this 4-vertex graph contain a
triangle (K3)?" -- expressed as a search over the 4 possible 3-vertex
subsets, and solves it with a real Grover search circuit on Qiskit's ideal
AerSimulator. The classical answer is computed from first principles (plain
enumeration, no external data) in this script, and the quantum result is
checked against it.

Graph instance (4 vertices, edges given explicitly):
    edges = {(0,1), (1,2), (0,2), (2,3)}
3-vertex subsets, indexed 0..3 (2 qubits):
    index 0 -> {0,1,2}
    index 1 -> {0,1,3}
    index 2 -> {0,2,3}
    index 3 -> {1,2,3}
Exactly one subset ({0,1,2}) has all three of its edges present, i.e. forms
a triangle. Grover search over the 2-qubit index register finds it.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

EDGES = {(0, 1), (1, 2), (0, 2), (2, 3)}


def has_edge(u, v):
    return (u, v) in EDGES or (v, u) in EDGES


VERTICES = [0, 1, 2, 3]
SUBSETS = list(combinations(VERTICES, 3))  # 4 subsets, index order matches list order


def is_triangle(subset):
    a, b, c = subset
    return has_edge(a, b) and has_edge(b, c) and has_edge(a, c)


classical_marked = [i for i, s in enumerate(SUBSETS) if is_triangle(s)]
assert len(classical_marked) == 1, "instance must have exactly one triangle for this demo"
classical_answer = classical_marked[0]

print(f"Subsets (index -> vertices): {list(enumerate(SUBSETS))}")
print(f"Classical answer: index {classical_answer} -> subset {SUBSETS[classical_answer]} is the unique triangle")


# ---------------------------------------------------------------------------
# 2. Grover search over the 2-qubit index register (N = 4, 1 marked item).
# ---------------------------------------------------------------------------

n_qubits = 2  # log2(4) = 2, indexes 0..3


def oracle_for(index, qc: QuantumCircuit):
    """Flip the phase of the basis state |index> (2-qubit, big-endian bit order)."""
    bits = format(index, f"0{n_qubits}b")
    # Flip qubits that should be 0 so the marked state becomes |11>, apply CZ, flip back.
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    qc.cz(0, 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)


def diffuser(qc: QuantumCircuit):
    qc.h([0, 1])
    qc.x([0, 1])
    qc.cz(0, 1)
    qc.x([0, 1])
    qc.h([0, 1])


qc = QuantumCircuit(n_qubits, n_qubits)
qc.h([0, 1])  # uniform superposition over the 4 indices

# Optimal number of Grover iterations for N=4, M=1: floor(pi/4 * sqrt(N/M)) = 1
n_iterations = int(np.floor((np.pi / 4) * np.sqrt(2 ** n_qubits / len(classical_marked))))
n_iterations = max(1, n_iterations)

for _ in range(n_iterations):
    oracle_for(classical_answer, qc)
    diffuser(qc)

qc.measure([0, 1], [0, 1])


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 2048
result = backend.run(qc, shots=shots).result()
counts = result.get_counts()

print(f"Grover iterations used: {n_iterations}")
print(f"Measurement counts: {counts}")

# Qiskit reports bitstrings as c1c0 (little-endian in the classical register order);
# our index used big-endian 'bits' above with qubit0 = MSB per oracle_for's `bits[0]`
# mapping to qubit index 0. Reconstruct the index the same way for each measured string.
most_common_bitstring = max(counts, key=counts.get)
# most_common_bitstring is "q1q0" (Qiskit's default order, reversed from qubit index)
q0 = most_common_bitstring[-1]
q1 = most_common_bitstring[-2]
measured_bits = q0 + q1  # matches the (i=0 first, i=1 second) order used in oracle_for
measured_index = int(measured_bits, 2)

print(f"Most frequent measured index: {measured_index} (subset {SUBSETS[measured_index]})")

success_probability = counts.get(most_common_bitstring, 0) / shots
print(f"Success probability (fraction landing on most frequent outcome): {success_probability:.3f}")

quantum_answer = measured_index

if quantum_answer == classical_answer and success_probability > 0.90:
    print("PASS")
else:
    print("FAIL")
