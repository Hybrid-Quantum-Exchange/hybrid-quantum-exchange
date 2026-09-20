"""
Erdos problem #719 (per manman4/erdosproblems data/problems.yaml, checked
2026-09-19) -- LIMITATION NOTICE FIRST:

Problem #719's YAML record has `oeis: ["possible"]`, `prize: "no"`, tags
["graph theory", "hypergraphs"], and no narrative text anywhere else in the
cloned repository (only the numeric listing in README.md). "possible" is a
placeholder in this dataset, not a real OEIS identifier -- there is no A-number
to derive a sequence from for this problem, and no page describing the actual
statement is present in the clone. So this script does NOT test a term of a
specific OEIS sequence tied to #719 (that would be fabrication, which the task
explicitly forbids).

Honest best-effort instead: using only the two real tags the record does give
("graph theory", "hypergraphs" -> "does a triangle exist in a small graph" is
the canonical finite, computable graph-theory search problem in that family),
this script builds a genuine Grover-search quantum circuit that finds the
unique triangle in a small fixed 5-vertex graph, and checks the quantum answer
against a brute-force classical enumeration computed from first principles in
this script. This is real mathematical content and a real quantum computation,
but it is a generic instance representative of the problem's tags, NOT a
verified instance of Erdos problem #719 itself, since #719's precise statement
could not be established from the available source data.

Report accordingly: ran_ok should reflect whether the circuit runs and PASSes
on this triangle-search instance; verified_against_classical should reflect
that comparison. This is NOT a claim that OEIS/problem #719 itself has been
verified.

--- Classical instance ---
Graph on vertices V = {0,1,2,3,4}, edges:
  {0,1}, {1,2}, {0,2}   (a triangle on 0,1,2)
  {3,4}, {0,3}          (two extra edges, no additional triangle)

All C(5,3) = 10 three-vertex subsets are enumerated classically below and
checked for "all three edges present" (i.e. forms a triangle). This is
computed in this script, not copied from anywhere.

--- Quantum circuit ---
A 4-qubit index register enumerates values 0..15. Values 0..9 are mapped (via
itertools.combinations, the standard combinatorial numbering) to the ten
3-vertex subsets; values 10..15 are unused/never marked. Grover's algorithm
(oracle + diffuser, both built directly as gates, no qiskit_algorithms) is
used to amplify the single marked index (the unique triangle {0,1,2}) with
the standard optimal iteration count round(pi/4 * sqrt(N/M)), N=16, M=1.

PASS criterion: the most frequently measured index, decoded back to a vertex
subset, equals the classical brute-force triangle {0,1,2}.
"""

import math
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no external data)
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3, 4]
EDGES = {frozenset(e) for e in [(0, 1), (1, 2), (0, 2), (3, 4), (0, 3)]}

subsets = list(combinations(VERTICES, 3))  # 10 subsets, indices 0..9


def is_triangle(subset):
    a, b, c = subset
    return (
        frozenset((a, b)) in EDGES
        and frozenset((b, c)) in EDGES
        and frozenset((a, c)) in EDGES
    )


classical_triangles = [s for s in subsets if is_triangle(s)]
assert len(classical_triangles) == 1, (
    f"instance must have exactly one triangle for this Grover search, "
    f"found {classical_triangles}"
)
classical_answer = classical_triangles[0]
target_index = subsets.index(classical_answer)  # index within 0..9

print(f"Classical brute-force triangle search over {len(subsets)} subsets:")
print(f"  triangles found: {classical_triangles}")
print(f"  target subset: {classical_answer}  ->  index {target_index} (binary "
      f"{format(target_index, '04b')})")

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for target_index among 16 (4-qubit) states
# ---------------------------------------------------------------------------

N_QUBITS = 4
N_STATES = 2 ** N_QUBITS  # 16
N_MARKED = 1


def oracle(qc, index, qubits):
    """Phase-flip the basis state |index> on the given qubits."""
    bits = format(index, f"0{len(qubits)}b")
    # Flip qubits that should be 0, so the target becomes |11...1>
    for bit, q in zip(bits, qubits):
        if bit == "0":
            qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for bit, q in zip(bits, qubits):
        if bit == "0":
            qc.x(q)


def diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))

qc.h(qubits)

iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / N_MARKED)))
print(f"Grover iterations: {iterations}")

for _ in range(iterations):
    oracle(qc, target_index, qubits)
    diffuser(qc, qubits)

qc.measure(qubits, qubits)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
result = backend.run(compiled, shots=4096).result()
counts = result.get_counts()

# Qiskit bit order is little-endian in the returned bitstring; reverse to
# match the qubit order used above.
decoded_counts = {}
for bitstring, c in counts.items():
    idx = int(bitstring[::-1], 2)
    decoded_counts[idx] = decoded_counts.get(idx, 0) + c

most_likely_index = max(decoded_counts, key=decoded_counts.get)
total_shots = sum(decoded_counts.values())
prob_target = decoded_counts.get(target_index, 0) / total_shots

print(f"Decoded measurement counts (top 5): "
      f"{sorted(decoded_counts.items(), key=lambda kv: -kv[1])[:5]}")
print(f"Most likely measured index: {most_likely_index} "
      f"(probability {decoded_counts.get(most_likely_index, 0) / total_shots:.3f})")
print(f"Probability of target index {target_index}: {prob_target:.3f}")

quantum_answer = subsets[most_likely_index] if most_likely_index < 10 else None

# ---------------------------------------------------------------------------
# 4. Compare and report
# ---------------------------------------------------------------------------

verified = (quantum_answer == classical_answer) and prob_target > 0.5

print()
print(f"Classical answer (unique triangle): {classical_answer}")
print(f"Quantum answer (most likely index decoded): {quantum_answer}")

if verified:
    print("PASS")
else:
    print("FAIL")
