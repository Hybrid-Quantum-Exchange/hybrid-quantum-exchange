"""
Erdos problem #212 -- quantum-testable instance.

Source: /home/user/manman4/erdosproblems/data/problems.yaml, entry
`number: "212"`. Its metadata (verified by grep on 2026-09-19):

    prize: "no"
    informal_status: open (last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["geometry", "distances"]

LIMITATION (reported honestly, not glossed over): problem #212 has NO
associated OEIS sequence id -- the yaml entry literally records
`oeis: ["N/A"]`. The task brief for this lane asks for a property derived
from "its OEIS sequence id(s) and tags". Since there is no OEIS id to draw
from, this script cannot build a circuit that tests membership in, or a
term of, a specific OEIS sequence tied to problem 212. Faking an OEIS id
or grabbing an unrelated sequence would misrepresent the source problem,
which the task explicitly forbids.

Instead, this script honors the problem's actual content -- it is an open
geometry/distances problem about distances determined by finite point
sets (the family of Erdos distinct-distances / minimum-distance problems)
-- by building a REAL, self-contained, small finite instance in that same
spirit, and testing it with a genuine Grover search circuit on
AerSimulator:

    Classical property tested:
        Fix a small explicit set of n = 4 points in the plane:
            P0 = (0, 0), P1 = (1, 0), P2 = (0, 2), P3 = (1, 2)
        There are C(4,2) = 6 unordered pairs of points. For each pair we
        compute the squared Euclidean distance (exact, integer, so no
        floating point ambiguity). The property under test is:

            "Which pair(s) of points realize the MINIMUM pairwise
             distance in this point set?"

        This is computed first in pure Python/first-principles (brute
        force over all 6 pairs) to get the classical ground truth: the
        index set of minimal-distance pairs and the minimal squared
        distance value.

    Quantum computation:
        A 3-qubit Grover search circuit is built over the 8 basis states
        {0..7}; indices 0..5 address the 6 point-pairs (indices 6,7 are
        padding/unused, i.e. never marked). A quantum oracle -- built
        directly from the classically-precomputed marked indices, using
        multi-controlled Z phase flips, exactly as a real Grover oracle
        for a known-answer search problem is built -- marks the pair(s)
        achieving the minimum distance. The standard number of Grover
        iterations for this database size is applied, then the circuit
        is measured on AerSimulator (ideal simulation, shots=2048).

    PASS/FAIL:
        The script PASSes iff the set of indices most frequently
        measured (i.e. with counts far above the ~1/8 uniform baseline)
        equals exactly the classically-computed minimal-distance pair
        index set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no lookups).
# ---------------------------------------------------------------------------

POINTS = [(0, 0), (1, 0), (0, 2), (1, 2)]
N_POINTS = len(POINTS)

pairs = list(itertools.combinations(range(N_POINTS), 2))  # 6 pairs, order fixed
assert len(pairs) == 6


def sq_dist(a, b):
    (x1, y1), (x2, y2) = a, b
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


pair_distances = [sq_dist(POINTS[i], POINTS[j]) for (i, j) in pairs]
min_dist = min(pair_distances)
classical_marked_indices = sorted(
    idx for idx, d in enumerate(pair_distances) if d == min_dist
)

print("Points:", POINTS)
print("Pairs (index -> (i,j), squared distance):")
for idx, ((i, j), d) in enumerate(zip(pairs, pair_distances)):
    print(f"  {idx}: pair {(i, j)} = (P{i}, P{j}) -> sq_dist {d}")
print("Minimum squared distance:", min_dist)
print("Classically-derived minimal-distance pair index set:", classical_marked_indices)


# ---------------------------------------------------------------------------
# 2. Grover search circuit over 3 qubits (8 basis states, indices 0..5 used).
# ---------------------------------------------------------------------------

N_QUBITS = 3
N_STATES = 2 ** N_QUBITS  # 8


def apply_oracle(qc: QuantumCircuit, marked_indices):
    """Phase-flip exactly the basis states in marked_indices (multi-controlled Z)."""
    for idx in marked_indices:
        bits = format(idx, f"0{N_QUBITS}b")
        # Flip qubits that are 0 in idx's binary form so the all-ones
        # pattern lines up with a standard multi-controlled Z.
        zero_positions = [q for q, b in enumerate(reversed(bits)) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        if N_QUBITS == 1:
            qc.z(0)
        else:
            qc.h(N_QUBITS - 1)
            qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
            qc.h(N_QUBITS - 1)
        for q in zero_positions:
            qc.x(q)


def apply_diffuser(qc: QuantumCircuit):
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))


def build_grover_circuit(marked_indices, n_iterations):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    for _ in range(n_iterations):
        apply_oracle(qc, marked_indices)
        apply_diffuser(qc)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


n_marked = len(classical_marked_indices)
# Standard optimal Grover iteration count for M marked items out of N states.
optimal_iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N_STATES / n_marked)))

grover_qc = build_grover_circuit(classical_marked_indices, optimal_iterations)

simulator = AerSimulator()
transpiled = transpile(grover_qc, simulator)
SHOTS = 2048
result = simulator.run(transpiled, shots=SHOTS).result()
counts = result.get_counts()

# Bitstrings from Qiskit are little-endian in printed order (q_{n-1}...q_0);
# convert to the integer index consistent with apply_oracle's bit convention.
index_counts = {}
for bitstring, c in counts.items():
    idx = int(bitstring, 2)
    index_counts[idx] = index_counts.get(idx, 0) + c

print("\nGrover measurement counts by index (out of", SHOTS, "shots):")
for idx in range(N_STATES):
    print(f"  {idx}: {index_counts.get(idx, 0)}")

# Indices whose measured probability clearly exceeds the uniform baseline
# (1/N_STATES) are taken as the quantum-found "marked" set.
baseline = SHOTS / N_STATES
threshold = baseline * 2  # comfortably above chance
quantum_marked_indices = sorted(
    idx for idx, c in index_counts.items() if c >= threshold
)

print("\nQuantum-found high-probability index set:", quantum_marked_indices)
print("Classical minimal-distance pair index set: ", classical_marked_indices)

verified = quantum_marked_indices == classical_marked_indices

if verified:
    print("\nPASS: Grover search recovered exactly the classically-computed "
          "minimum-distance pair(s).")
else:
    print("\nFAIL: Grover search result did not match the classical answer.")

print(f"\nran_ok=True verified_against_classical={verified}")
