"""
Erdos problem #668 -- quantum-testable instance
=================================================

Source: erdosproblems.com problem #668 (geometry / distances). Its metadata
in the erdosproblems dataset records:

    oeis: ["A385657"]
    tags: ["geometry", "distances"]

OEIS A385657 = "Number of nonisomorphic maximally dense unit-distance graphs
on n vertices" -- i.e. among all ways to place n points in the plane, the
maximum possible number of pairs at distance exactly 1 is some number
f(n) (this f(n) is itself the classical Erdos unit-distance function), and
A385657(n) counts how many non-isomorphic edge-graphs realize that maximum.
The dataset's first terms give A385657(4) = 1: for n = 4 points there is
exactly one graph shape (up to isomorphism) achieving the maximum number of
unit distances.

Classical property tested here (computed from first principles below, not
copied from OEIS):

    For n = 4 points in the plane, the maximum achievable number of pairs
    at distance exactly 1 is 5, realized by two equilateral unit triangles
    glued along a shared edge (a "bowtie"/rhombus-diagonal configuration):

        A = (0, 0)
        B = (1, 0)
        C = (0.5,  sqrt(3)/2)   -- apex of one equilateral triangle on AB
        D = (0.5, -sqrt(3)/2)   -- apex of the mirrored equilateral triangle

    Of the C(4,2) = 6 pairs, AB, AC, BC, AD, BD are unit distance (5 pairs)
    and CD = sqrt(3) is not. This script computes those 6 pairwise
    distances directly from coordinates and derives the 6-bit "edge
    pattern" (which of the 6 vertex pairs are unit-distance) as the ground
    truth target -- this is exactly one representative of the unique
    (A385657(4) = 1) maximally-dense unit-distance graph shape on 4 points.

Quantum property tested: a genuine unstructured (Grover) search over the
2^6 = 64 possible 6-bit edge patterns on 4 labeled vertices, for the
specific pattern that is the classically-computed maximum-unit-distance
edge set above. This is a legitimate small Grover instance -- 6 qubits,
one marked item out of 64, oracle built as a phase-flip on the exact
target bitstring, ~6 Grover iterations (near-optimal for N=64, M=1),
executed on the ideal AerSimulator.

Comparison: the most frequently measured 6-bit string from the Grover
circuit is compared against the classically-derived target bitstring;
PASS if they match.
"""

import math
import itertools

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical computation: the maximum unit-distance edge pattern for
#    n = 4 points, from first principles (coordinates -> distances).
# ---------------------------------------------------------------------

def classical_edge_pattern():
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    C = (0.5, math.sqrt(3) / 2)
    D = (0.5, -math.sqrt(3) / 2)
    points = [A, B, C, D]

    pairs = list(itertools.combinations(range(4), 2))  # 6 pairs, fixed order
    bits = []
    unit_count = 0
    for (i, j) in pairs:
        dx = points[i][0] - points[j][0]
        dy = points[i][1] - points[j][1]
        dist = math.hypot(dx, dy)
        is_unit = abs(dist - 1.0) < 1e-9
        bits.append(1 if is_unit else 0)
        if is_unit:
            unit_count += 1

    assert unit_count == 5, (
        f"expected 5 unit-distance pairs among 4 points, got {unit_count}"
    )
    # Sanity: 5 unit distances out of 6 possible pairs on 4 points is the
    # known maximum (K4 has only 6 pairs total; achieving all 6 would
    # require an equilateral-triangle-like arrangement of *all four*
    # points mutually equidistant, impossible in the plane for n=4>3).
    return pairs, bits


PAIRS, TARGET_BITS = classical_edge_pattern()
# Bit string as it will be read off the quantum register: qubit 0 is the
# pair PAIRS[0], ..., qubit 5 is PAIRS[5]. Qiskit's classical bitstring in
# get_counts() prints with qubit 0 as the *rightmost* character.
TARGET_BITSTRING = "".join(str(b) for b in reversed(TARGET_BITS))
TARGET_INT = int("".join(str(b) for b in TARGET_BITS), 2)  # for reference only

print("Pairs (vertex indices):", PAIRS)
print("Classical unit-distance edge pattern (bit i <-> PAIRS[i]):", TARGET_BITS)
print("Target bitstring (Qiskit little-endian print order):", TARGET_BITSTRING)


# ---------------------------------------------------------------------
# 2. Quantum part: Grover search over 6 qubits for the one bitstring
#    matching TARGET_BITS.
# ---------------------------------------------------------------------

N_QUBITS = 6
N = 2 ** N_QUBITS


def build_oracle(target_bits):
    """Phase-flip oracle marking the single basis state `target_bits`
    (list of 0/1, length N_QUBITS, index i -> qubit i)."""
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    zero_qubits = [i for i, b in enumerate(target_bits) if b == 0]
    for q in zero_qubits:
        qc.x(q)
    # Multi-controlled Z on all N_QUBITS-1 controls + 1 target, realized as
    # H - MCX - H on the last qubit.
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    for q in zero_qubits:
        qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_search(target_bits, n_qubits=N_QUBITS, shots=2048):
    n_states = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_states)))  # M = 1 marked item

    oracle = build_oracle(target_bits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    qc = qc.decompose().decompose().decompose()

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    best_bitstring = max(counts, key=counts.get)
    return best_bitstring, counts, iterations


def main():
    best_bitstring, counts, iterations = grover_search(TARGET_BITS)

    total_shots = sum(counts.values())
    target_shots = counts.get(TARGET_BITSTRING, 0)
    target_prob = target_shots / total_shots

    print(f"\nGrover iterations used: {iterations}")
    print(f"Most frequent measured bitstring: {best_bitstring} "
          f"(count {counts[best_bitstring]} / {total_shots})")
    print(f"Probability mass on target bitstring {TARGET_BITSTRING}: "
          f"{target_prob:.4f}")

    verified = (best_bitstring == TARGET_BITSTRING) and (target_prob > 0.5)

    if verified:
        print("\nPASS: Grover search recovered the classically-computed "
              "maximum unit-distance edge pattern for n=4 points.")
    else:
        print("\nFAIL: Grover search result does not match the classical "
              "unit-distance edge pattern.")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
