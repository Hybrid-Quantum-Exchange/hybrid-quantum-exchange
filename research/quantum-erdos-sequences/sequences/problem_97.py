"""
Erdos problem #97 (from manman4/erdosproblems, data/problems.yaml).

Metadata found for problem #97 in that dataset:
    prize: "$100"
    status: "falsifiable" (as of 2025-08-31)
    oeis: ["N/A"]
    tags: ["geometry", "distances", "convex"]

LIMITATION, stated honestly up front: problem #97 has no associated OEIS
sequence id in the source data (oeis: ["N/A"]). There is therefore no
"sequence" from this problem to make quantum-testable in the sense the
other lanes in this library use (checking membership/terms of a named
OEIS sequence). This script does NOT fabricate an OEIS id or claim a
value from one.

What this script does instead, honestly scoped to what the metadata does
support: the tags ("geometry", "distances", "convex") describe Erdos-style
combinatorial-geometry problems about pairwise distances among points in
convex position. We build a small, concrete, fully classically-computable
instance of that flavor of question -- "which pairs of points, among a
fixed small point set in convex position, realize the globally minimum
pairwise distance?" -- and use a genuine Grover search circuit to find a
pair achieving that minimum, verifying the quantum result against a
from-scratch classical computation.

Concrete finite instance:
    - 6 points placed at the vertices of a regular hexagon on the unit
      circle (a convex point set): angles 2*pi*k/6 for k = 0..5.
    - There are C(6,2) = 15 unordered pairs. Each pair is assigned an
      index 0..14 (computed in this script, not hard-coded from any
      external source) and encoded on 4 qubits (16 basis states; index 15
      is unused/invalid and is excluded from the marked set).
    - The classical target property: the pair(s) of points whose
      Euclidean distance equals the minimum pairwise distance over all 15
      pairs. For a regular hexagon on the unit circle this minimum is the
      edge length (distance between angularly-adjacent vertices), and by
      symmetry all 6 edges achieve it -- this is derived by direct
      computation below, not asserted.

Quantum method: Grover's algorithm. The oracle is a diagonal unitary
(phase flip) built directly from the classically-computed set of "good"
pair-indices, and the diffuser is the standard Grover diffusion operator.
We run on the ideal AerSimulator, measure, and check that the
most-frequently-measured 4-qubit index is one of the classically-known
minimum-distance pairs.

PASS/FAIL is decided purely by comparing the quantum measurement outcome
to the classical brute-force answer computed in this script.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator


def classical_min_distance_pairs():
    """Compute, from first principles, the minimum-distance pair(s) among
    6 points on a regular hexagon (unit circle), and enumerate all pairs.
    Returns (pairs, distances, min_dist, good_indices)."""
    n = 6
    points = [
        (math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n))
        for k in range(n)
    ]
    pairs = list(itertools.combinations(range(n), 2))  # 15 pairs, indices 0..14
    distances = []
    for (i, j) in pairs:
        dx = points[i][0] - points[j][0]
        dy = points[i][1] - points[j][1]
        distances.append(math.hypot(dx, dy))

    min_dist = min(distances)
    tol = 1e-9
    good_indices = [
        idx for idx, d in enumerate(distances) if abs(d - min_dist) < tol
    ]
    return pairs, distances, min_dist, good_indices


def build_oracle(num_qubits, marked_indices):
    """Diagonal phase-flip oracle: -1 phase on each marked computational
    basis index, +1 elsewhere. Returns an Operator (unitary, diagonal)."""
    dim = 2 ** num_qubits
    diag = np.ones(dim, dtype=complex)
    for idx in marked_indices:
        diag[idx] = -1.0
    return Operator(np.diag(diag))


def build_diffuser(num_qubits):
    """Standard Grover diffusion operator: 2|s><s| - I, where |s> is the
    equal superposition state."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover(num_qubits, marked_indices, shots=2048):
    dim = 2 ** num_qubits
    num_marked = len(marked_indices)
    # Optimal number of Grover iterations for this marked fraction.
    theta = math.asin(math.sqrt(num_marked / dim))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle_op = build_oracle(num_qubits, marked_indices)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.unitary(oracle_op, range(num_qubits), label="oracle")
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    pairs, distances, min_dist, good_indices = classical_min_distance_pairs()

    print("Erdos problem #97 -- metadata: oeis=['N/A'], tags=['geometry','distances','convex']")
    print("No OEIS sequence exists for this problem; using a directly-derived")
    print("finite geometric property instead (see module docstring).")
    print()
    print(f"6 points (regular hexagon), {len(pairs)} pairs, distances (classical):")
    for (i, j), d in zip(pairs, distances):
        print(f"  pair ({i},{j}) -> distance {d:.6f}")
    print(f"Classical minimum distance: {min_dist:.6f}")
    print(f"Classical minimum-distance pair indices (0..14): {good_indices}")

    num_qubits = 4  # covers indices 0..15, we only ever mark 0..14
    counts, iterations = run_grover(num_qubits, good_indices)

    # Most frequent measured outcome.
    best_bitstring = max(counts, key=counts.get)
    best_index = int(best_bitstring, 2)
    total_shots = sum(counts.values())
    prob_good = sum(
        c for bs, c in counts.items() if int(bs, 2) in good_indices
    ) / total_shots

    print()
    print(f"Grover iterations used: {iterations}")
    print(f"Most frequent measured index: {best_index} (bitstring {best_bitstring})")
    print(f"Fraction of shots landing on a classically-correct min-distance pair: {prob_good:.4f}")

    verified = (best_index in good_indices) and (prob_good > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
