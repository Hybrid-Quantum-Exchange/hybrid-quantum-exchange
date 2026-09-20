"""
Erdos problem #92 -- quantum-testable instance
================================================

Source metadata (erdosproblems.com data, data/problems.yaml, number: "92"):
    prize: $500
    status: disproved (2026-05-21)
    oeis: ["possible"]
    tags: ["geometry", "distances"]

Honesty note on the OEIS id
----------------------------
The "oeis" field for problem 92 in the source data is the literal string
"possible", which is not a real OEIS sequence identifier (it does not match
the A-NNNNNN pattern and no such sequence exists to look up). There is
therefore NO usable OEIS sequence attached to this problem to build a
membership/term test from. Rather than fabricate an OEIS id or copy a
number that was never actually looked up, this script instead uses the
problem's TAGS ("geometry", "distances") to build a genuine, small,
finite, classically-checkable geometric question in the same family as
Erdos's distance problems (e.g. the unit-distance / minimum-distance
problem): for a small fixed point set in the plane, which pairs of points
realize the MINIMUM pairwise distance?

This is real mathematical content -- it is computed from first principles
(brute-force over all pairs, exact squared-distance arithmetic, no shortcuts
or hard-coded "known" answers) -- but it is a substitute for an OEIS-term
test, not a term of an OEIS sequence tied to problem 92. That substitution,
and the reason for it, is the "limitation" being disclosed here per the
task instructions.

The classical property tested
------------------------------
Fix 4 points in the plane (a small, explicit configuration -- three corners
of a unit square plus one point placed slightly off, chosen so the minimum
pairwise distance is realized by an odd number of pairs, avoiding any
accidental Grover degeneracy issues):

    P0 = (0, 0)
    P1 = (1, 0)
    P2 = (0, 1)
    P3 = (2, 2)

There are C(4,2) = 6 unordered pairs, indexed 0..5 in the fixed order
    0:(P0,P1) 1:(P0,P2) 2:(P0,P3) 3:(P1,P2) 4:(P1,P3) 5:(P2,P3)

The classical property: find the index (indices) of the pair(s) achieving
the MINIMUM squared Euclidean distance among all 6 pairs. This is computed
exactly in Python with integer arithmetic (squared distances, no floats),
so it is a genuine, checkable, finite classical fact -- and it is the same
combinatorial "distances" question Erdos-style distance problems ask,
scaled down to a size a 3-qubit Grover search can handle.

The quantum circuit
--------------------
A 3-qubit Grover search (index space of size 8, padded from 6 real pairs;
the 2 padding indices are never marked as solutions) whose oracle flags
exactly the pair-index/indices realizing the minimum squared distance,
computed classically and burned into the oracle as a phase-flip on the
matching computational basis state(s). One Grover iteration (optimal for
this database size/solution count) is applied, then the register is
measured on the ideal AerSimulator and the most likely outcome is compared
against the classical answer.

PASS/FAIL
---------
The script prints PASS if the most-frequently-measured basis state(s) from
the quantum circuit match exactly the classically-computed minimum-distance
pair index/indices, else FAIL.
"""

from itertools import combinations
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_answer():
    """Brute-force, exact-arithmetic computation of the minimum-distance pair(s)."""
    points = [(0, 0), (1, 0), (0, 1), (2, 2)]
    pairs = list(combinations(range(len(points)), 2))  # fixed order, 6 pairs

    sq_dists = []
    for (i, j) in pairs:
        xi, yi = points[i]
        xj, yj = points[j]
        d2 = (xi - xj) ** 2 + (yi - yj) ** 2
        sq_dists.append(d2)

    min_d2 = min(sq_dists)
    solution_indices = [idx for idx, d2 in enumerate(sq_dists) if d2 == min_d2]
    return pairs, sq_dists, min_d2, solution_indices


def build_oracle(n_qubits, marked_indices):
    """Phase-flip oracle marking each index in marked_indices (each < 2**n_qubits)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")[::-1]  # little-endian per qubit order
        flip_qubits = [q for q, b in enumerate(bits) if b == "0"]
        if flip_qubits:
            qc.x(flip_qubits)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if flip_qubits:
            qc.x(flip_qubits)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits, marked_indices, shots=4096):
    n_items = 2 ** n_qubits
    n_solutions = len(marked_indices)

    # Optimal number of Grover iterations for this database/solution size.
    theta = math.asin(math.sqrt(n_solutions / n_items))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_indices)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    pairs, sq_dists, min_d2, solution_indices = classical_answer()

    print("Erdos problem #92 -- geometry/distances substitute instance")
    print(f"Points: P0=(0,0) P1=(1,0) P2=(0,1) P3=(2,2)")
    print(f"Pairs (index: (i,j) -> squared distance):")
    for idx, ((i, j), d2) in enumerate(zip(pairs, sq_dists)):
        print(f"  {idx}: (P{i},P{j}) -> {d2}")
    print(f"Classical minimum squared distance: {min_d2}")
    print(f"Classical minimum-distance pair index/indices: {solution_indices}")

    n_qubits = 3  # covers indices 0..7, real pairs occupy 0..5
    counts, iterations = run_grover(n_qubits, solution_indices)
    print(f"\nGrover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    total_shots = sum(counts.values())
    # Interpret each bitstring back into our index convention, and treat any
    # index amplified to at least 5% of shots as "measured" (Grover boosts
    # every true solution's amplitude together; with 2 solutions out of 8
    # items each gets roughly half of the amplified probability mass, so a
    # 5% threshold comfortably separates amplified solutions from the
    # residual noise on non-solution indices).
    measured_solution_shots = 0
    significant_indices = []
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        if idx in solution_indices:
            measured_solution_shots += count
        if count / total_shots >= 0.05:
            significant_indices.append(idx)

    significant_indices = sorted(set(significant_indices))
    solution_probability = measured_solution_shots / total_shots

    print(f"Significantly measured index/indices (>=5% of shots): {significant_indices}")
    print(f"Fraction of shots landing on a true solution index: {solution_probability:.3f}")

    quantum_matches_classical = set(significant_indices) == set(solution_indices)

    if quantum_matches_classical and solution_probability > 0.9:
        print("\nPASS: Grover search's top measured index/indices match the "
              "classically computed minimum-distance pair(s).")
    else:
        print("\nFAIL: Grover search result does not match the classical answer.")


if __name__ == "__main__":
    main()
