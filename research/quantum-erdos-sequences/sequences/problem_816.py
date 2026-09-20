#!/usr/bin/env python3
"""
Erdos problem #816 (per manman4/erdosproblems data/problems.yaml, entry
`number: "816"`) is tagged only `["graph theory"]` and its `oeis` field is
`["N/A"]` -- there is no OEIS sequence id attached to this problem. The task
instructions for this lane say: if no OEIS id exists, write the script anyway
with a best honest attempt at a genuine finite/computable property in the
same spirit (graph theory), note the limitation, and report accurately.

LIMITATION: this script is therefore NOT built from an OEIS sequence for
problem 816 (none exists in the source data). Instead it tests a small,
self-contained, classically-verified graph-theory fact that sits squarely in
problem 816's stated tag ("graph theory") and is a direct instance of the
Turan-type extremal question that underlies many Erdos graph-theory
problems: the maximum number of edges in a triangle-free graph on 4
vertices.

Classical property under test
------------------------------
Let K4 have vertex set {0,1,2,3} and the 6 candidate edges
(0,1) (0,2) (0,3) (1,2) (1,3) (2,3), encoded as 6 bits (1 = edge present).
Turan's theorem (ex(n, K3) = floor(n^2/4)) says the maximum size of a
triangle-free graph on 4 vertices is floor(16/4) = 4 edges, achieved
uniquely (up to isomorphism) by the complete bipartite graph K_{2,2} = C4.

The finite, computable property tested here: among all 2^6 = 64 edge
subsets of K4, find the subset(s) with exactly 4 edges that are
triangle-free. The script first computes this classically by brute force
(first principles, no lookups), then uses a real Grover search circuit on
6 qubits (one per candidate edge) over the AerSimulator to search the same
64-element space for exactly those marked bitstrings, and checks that the
circuit's most-likely measured outcome(s) match the classical answer.

Classically computed answer (recomputed in this script, not hard-coded):
exactly 3 such edge-sets exist, each isomorphic to C4 (a 4-cycle) -- the
three ways to choose which 2 of the 3 perfect matchings of K4 to keep out.

Circuit design
--------------
6 "edge" qubits index the 64 possible graphs. A classical oracle (built by
brute-force enumeration, i.e. from first principles) is compiled into a
multi-controlled phase-flip (Z) gate for each of the 3 marked bitstrings.
Grover diffusion is applied for the optimal number of iterations
round(pi/4 * sqrt(N/M)) with N=64, M=3. The circuit is run on AerSimulator
and the measurement histogram's most frequent outcomes are compared against
the classically-computed marked set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, brute force over K4 edge sets)
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges, index 0..5
N_QUBITS = len(EDGES)
TRIPLES = list(itertools.combinations(VERTICES, 3))  # the 4 possible triangles


def bits_to_edgeset(bits: tuple[int, ...]) -> set[frozenset[int]]:
    return {frozenset(EDGES[i]) for i, b in enumerate(bits) if b}


def is_triangle_free(edgeset: set[frozenset[int]]) -> bool:
    for tri in TRIPLES:
        tri_edges = {frozenset(p) for p in itertools.combinations(tri, 2)}
        if tri_edges.issubset(edgeset):
            return False
    return True


def classical_marked_bitstrings() -> list[tuple[int, ...]]:
    """Brute-force every one of the 2^6 edge subsets of K4."""
    marked = []
    for bits in itertools.product([0, 1], repeat=N_QUBITS):
        if sum(bits) != 4:
            continue
        if is_triangle_free(bits_to_edgeset(bits)):
            marked.append(bits)
    return marked


CLASSICAL_MARKED = classical_marked_bitstrings()
# Qiskit bit ordering: qubit 0 is the rightmost character of the bitstring.
CLASSICAL_MARKED_STRS = {
    "".join(str(b) for b in reversed(bits)) for bits in CLASSICAL_MARKED
}

print("Classical brute-force result:")
print(f"  Candidate edges of K4: {EDGES}")
print(f"  Total edge subsets searched: {2 ** N_QUBITS}")
print(f"  Triangle-free subsets with exactly 4 edges: {len(CLASSICAL_MARKED)}")
for bits in CLASSICAL_MARKED:
    edges_on = [EDGES[i] for i, b in enumerate(bits) if b]
    print(f"    bits={bits} -> edges {edges_on} (this is a 4-cycle / K_2,2)")

assert len(CLASSICAL_MARKED) == 3, "classical brute force disagrees with Turan's theorem"


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffusion, built directly from the classical marked set
# ---------------------------------------------------------------------------

def apply_multi_controlled_z(qc: QuantumCircuit, bitstring: str, qubits: list[int]) -> None:
    """Flip the phase of |bitstring> (Qiskit little-endian convention)."""
    zero_positions = [q for q, c in zip(qubits, reversed(bitstring)) if c == "0"]
    for q in zero_positions:
        qc.x(q)
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q in zero_positions:
        qc.x(q)


def oracle(qc: QuantumCircuit, qubits: list[int]) -> None:
    for bitstring in CLASSICAL_MARKED_STRS:
        apply_multi_controlled_z(qc, bitstring, qubits)


def diffusion(qc: QuantumCircuit, qubits: list[int]) -> None:
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


N = 2 ** N_QUBITS
M = len(CLASSICAL_MARKED)
iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"\nGrover iterations used: {iterations} (N={N}, M={M})")

qubits = list(range(N_QUBITS))
qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(qubits)
for _ in range(iterations):
    oracle(qc, qubits)
    diffusion(qc, qubits)
qc.measure(qubits, qubits)

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_outcomes = [c for c, _ in sorted_counts[:M]]

print("\nTop measured outcomes (quantum):")
for outcome, freq in sorted_counts[:M]:
    print(f"  {outcome} : {freq}/{shots} = {freq / shots:.3f}")

quantum_marked = set(top_outcomes)
classical_marked = CLASSICAL_MARKED_STRS

# Amplitude sanity check: the marked outcomes together should carry the
# large majority of measured probability after amplitude amplification.
marked_prob = sum(counts.get(s, 0) for s in classical_marked) / shots

print(f"\nClassical marked bitstrings (little-endian): {sorted(classical_marked)}")
print(f"Quantum top-{M} bitstrings:                    {sorted(quantum_marked)}")
print(f"Total measured probability on classical marked set: {marked_prob:.3f}")

verified = quantum_marked == classical_marked and marked_prob > 0.7

print("\nPASS" if verified else "\nFAIL")
