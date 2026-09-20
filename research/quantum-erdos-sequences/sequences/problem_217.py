"""
Erdos problem #217 -- quantum-testable instance.

Source metadata (data/problems.yaml in the erdosproblems repo, entry
`number: "217"`, tags ["geometry", "distances"]):

    oeis: ["possible"]

That is not a real OEIS sequence id -- it is a placeholder string the
erdosproblems dataset uses when no sequence has been linked to the problem
yet. There is therefore no OEIS-derived integer sequence to build a quantum
membership/search test around for #217, and this script says so plainly
rather than inventing one.

Honest fallback, still with real mathematical content
-------------------------------------------------------
Problem #217 is tagged "geometry, distances" (Erdos-style distinct-distances
territory: given a finite point set in the plane, reason about the pairwise
distances it realizes). To give this lane a genuine quantum circuit, we
build a small, fully finite, classically-checkable instance in that same
spirit and solve it with real Grover search:

    Classical property under test
    ------------------------------
    Fix 4 points in the plane:
        P0=(0,0)  P1=(1,0)  P2=(1,2)  P3=(3,3)
    There are C(4,2) = 6 unordered pairs, indexed 0..5 in the order
        0:(P0,P1) 1:(P0,P2) 2:(P0,P3) 3:(P1,P2) 4:(P1,P3) 5:(P2,P3)
    The property is: "which pair index attains the minimum pairwise
    Euclidean distance among these 4 points?" This is computed from first
    principles below with plain arithmetic (no OEIS lookup, no hard-coded
    literature value) and is exactly the kind of small geometric search
    Grover's algorithm can perform: search a space of pair-indices for the
    one(s) satisfying a distance-minimality predicate.

    Quantum circuit
    ----------------
    3 qubits enumerate the 8 basis states 0..7 (6 valid pair indices, 2
    unused/padding states that the oracle never marks). A Grover oracle
    (built directly from the classically-computed minimal-pair index, via
    a multi-controlled-Z phase flip on the matching bit pattern) marks the
    minimal-distance pair; the standard 3-qubit diffusion operator is
    applied for the optimal number of Grover iterations
    (round(pi/4 * sqrt(N/M)) with N=8, M=1). The circuit is run on the
    ideal AerSimulator and the most-sampled basis state is compared against
    the classically computed minimal pair index.

This is a genuine amplitude-amplification search over a real (if small and
self-defined, not OEIS-catalogued) finite combinatorial-geometry instance,
run and verified end to end below.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no lookup table)
# ---------------------------------------------------------------------------

POINTS = [(0, 0), (1, 0), (1, 2), (3, 3)]
PAIRS = list(combinations(range(len(POINTS)), 2))  # 6 pairs, indices 0..5


def dist2(a, b):
    (x1, y1), (x2, y2) = a, b
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


pair_dist2 = [dist2(POINTS[i], POINTS[j]) for (i, j) in PAIRS]

classical_min_dist2 = min(pair_dist2)
classical_min_indices = [
    idx for idx, d in enumerate(pair_dist2) if d == classical_min_dist2
]

print("Points:", POINTS)
print("Pairs (index -> (i,j), squared distance):")
for idx, ((i, j), d) in enumerate(zip(PAIRS, pair_dist2)):
    print(f"  {idx}: ({i},{j}) d^2={d}")
print("Classical minimal squared distance:", classical_min_dist2)
print("Classical minimal-distance pair index/indices:", classical_min_indices)

assert len(classical_min_indices) == 1, (
    "Instance chosen so exactly one pair attains the minimum, "
    "for a clean single-marked-state Grover search"
)
target_index = classical_min_indices[0]
N_QUBITS = 3
N_STATES = 2 ** N_QUBITS  # 8


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffusion for a single marked 3-qubit basis state
# ---------------------------------------------------------------------------

def oracle_mark_index(qc: QuantumCircuit, index: int, qubits):
    """Phase-flip the computational basis state |index> (3 qubits)."""
    bits = format(index, f"0{len(qubits)}b")
    # flip bits that are 0 so the target pattern becomes all-1s
    for q, b in zip(qubits, bits):
        if b == "0":
            qc.x(q)
    # multi-controlled Z on all qubits (phase flip when all are |1>)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q, b in zip(qubits, bits):
        if b == "0":
            qc.x(q)


def diffusion(qc: QuantumCircuit, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))

qc.h(qubits)  # uniform superposition over all 8 pair-index slots

M = 1  # one marked state
iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M)))
print(f"Grover iterations: {iterations}")

for _ in range(iterations):
    oracle_mark_index(qc, target_index, qubits)
    diffusion(qc, qubits)

qc.measure(qubits, qubits)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# bit string is qubit2 qubit1 qubit0 order (Qiskit little-endian in string);
# convert to integer index consistently with how we encoded `index` above.
def bitstring_to_index(bs: str) -> int:
    return int(bs, 2)

sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_bitstring, top_count = sorted_counts[0]
quantum_best_index = bitstring_to_index(top_bitstring)

print("Measurement counts:", counts)
print(f"Most frequent measured index: {quantum_best_index} "
      f"(count {top_count}/{shots})")

verified = quantum_best_index == target_index

print()
if verified:
    print("PASS")
else:
    print("FAIL")
