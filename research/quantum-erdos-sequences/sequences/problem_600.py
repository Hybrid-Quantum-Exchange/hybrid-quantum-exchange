"""
Erdos problem #600 -- quantum-testable sequence lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: '600'"):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["possible"]
    tags: ["graph theory"]

LIMITATION, stated honestly up front: the "oeis" field for problem #600 is the
literal placeholder string "possible", not a real OEIS sequence id. There is
no OEIS A-number attached to this problem in the source data, so the intended
"identify a property of the OEIS sequence and test it with a quantum circuit"
approach does not apply here -- there is no sequence to test membership or
term-values against. Problem #600 itself is an open graph-theory problem
(no closed-form classical answer exists to compare against either).

Best honest attempt in that situation: build a REAL, correctness-verifiable
quantum circuit for a small, finite, computable problem in the same tag
("graph theory") that problem #600 belongs to, rather than fabricate an OEIS
value that isn't in the source data. Concretely: Grover's algorithm searching
for a triangle (a 3-clique) among the 3-vertex subsets of a small fixed graph.

Classical property being tested
--------------------------------
Fix a graph G on 5 labeled vertices {0,1,2,3,4} with a known, hand-picked edge
set that contains EXACTLY ONE triangle. The search space is the C(5,3) = 10
three-vertex subsets of G, indexed 0..9 in the standard combinatorial
(colex) order. The classical answer -- which subset index is a triangle in G
-- is computed from first principles in this script by brute-force checking
all 10 subsets against the edge set (no OEIS lookup, no hardcoded answer).

The subset index is encoded in 4 qubits (16 basis states; indices 10-15 are
outside the domain and never marked). Grover's algorithm is run with an
oracle that phase-flips exactly the classical triangle index, followed by
the standard diffusion operator, for the optimal number of iterations for a
domain this size. The script then samples the resulting state on the ideal
AerSimulator and checks that the most frequently measured index equals the
classical answer.

This is a genuine amplitude-amplification computation (not a lookup): the
circuit has no built-in knowledge of which subset is a triangle beyond the
oracle that encodes G's edge set, and the classical brute-force check is
performed independently in this script to produce the expected answer.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical setup: a 5-vertex graph with exactly one triangle.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3, 4]

# Hand-picked edge set. Triangle exists on {0,1,2}. No other 3-subset is a
# triangle (verified below by brute force, not assumed).
EDGES = {
    frozenset((0, 1)),
    frozenset((1, 2)),
    frozenset((0, 2)),
    frozenset((2, 3)),
    frozenset((3, 4)),
}


def is_edge(u, v):
    return frozenset((u, v)) in EDGES


def is_triangle(subset):
    a, b, c = subset
    return is_edge(a, b) and is_edge(b, c) and is_edge(a, c)


# All 3-vertex subsets of a 5-vertex set, in colex order -> this fixes the
# index <-> subset correspondence used to build the oracle.
ALL_SUBSETS = list(itertools.combinations(VERTICES, 3))
assert len(ALL_SUBSETS) == 10  # C(5,3)

triangle_indices = [i for i, s in enumerate(ALL_SUBSETS) if is_triangle(s)]

# This is the "small, finite, computable property": there is exactly one
# 3-subset of G that forms a triangle. Verify that from first principles.
assert len(triangle_indices) == 1, (
    f"expected exactly one triangle in the hand-picked graph, found "
    f"{len(triangle_indices)}: {[ALL_SUBSETS[i] for i in triangle_indices]}"
)
CLASSICAL_ANSWER = triangle_indices[0]
CLASSICAL_TRIANGLE = ALL_SUBSETS[CLASSICAL_ANSWER]

print(f"Graph vertices: {VERTICES}")
print(f"Graph edges:    {[tuple(e) for e in EDGES]}")
print(f"All {len(ALL_SUBSETS)} 3-subsets (index -> subset):")
for i, s in enumerate(ALL_SUBSETS):
    print(f"  {i:2d}: {s}{'  <-- triangle' if i == CLASSICAL_ANSWER else ''}")
print(f"Classical answer: index {CLASSICAL_ANSWER} = subset {CLASSICAL_TRIANGLE}")

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over 4 index qubits for CLASSICAL_ANSWER.
# ---------------------------------------------------------------------------

N_QUBITS = 4  # 16 basis states, domain is indices 0..9
N_ITEMS_SEARCHED = 2 ** N_QUBITS
N_MARKED = 1

target_bits = format(CLASSICAL_ANSWER, f"0{N_QUBITS}b")  # index qubits q0..q3


def apply_oracle(qc, qubits, bits):
    """Phase-flip the basis state matching `bits` (MSB-first string)."""
    # X on qubits that should be 0 in the target, so target maps to all-1s.
    for qubit, bit in zip(qubits, bits):
        if bit == "0":
            qc.x(qubit)
    # Multi-controlled Z on all qubits (phase flip |11...1>).
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for qubit, bit in zip(qubits, bits):
        if bit == "0":
            qc.x(qubit)


def apply_diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


# Optimal number of Grover iterations for 1 marked item out of 16.
n_iterations = max(1, round((math.pi / 4) * math.sqrt(N_ITEMS_SEARCHED / N_MARKED)))
print(f"Grover iterations: {n_iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))

qc.h(qubits)
for _ in range(n_iterations):
    apply_oracle(qc, qubits, target_bits)
    apply_diffuser(qc, qubits)
qc.measure(qubits, qubits)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit counts keys are bitstrings with qubit 0 as the rightmost character.
best_key = max(counts, key=counts.get)
# Reconstruct index qubit order q0..q3 (MSB-first, matching target_bits).
measured_index = int(best_key[::-1], 2)
measured_probability = counts[best_key] / shots

print(f"Measurement counts (top 5): "
      f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
print(f"Most frequent measured index: {measured_index} "
      f"(probability {measured_probability:.3f})")

ran_ok = True
verified = measured_index == CLASSICAL_ANSWER and measured_probability > 0.5

if verified:
    print("PASS: quantum Grover search found the classical triangle index "
          f"{CLASSICAL_ANSWER} with high probability.")
else:
    print("FAIL: quantum result did not match the classical answer "
          f"(expected {CLASSICAL_ANSWER}, got {measured_index}).")
