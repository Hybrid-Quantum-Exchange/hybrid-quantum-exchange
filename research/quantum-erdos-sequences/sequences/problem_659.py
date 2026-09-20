"""
Erdos problem #659 (erdosproblems.com / manman4/erdosproblems data/problems.yaml)

Source metadata for problem 659 (verified by grep on 2026-09-19):
    number: "659"
    prize: "no"
    status: "proved (Lean)"
    oeis: ["possible"]
    tags: ["geometry", "distances"]

LIMITATION, stated honestly up front: the "oeis" field for problem 659 in the
data file is the literal string "possible", not a real OEIS sequence id (no
A-number is given). There is therefore no genuine OEIS sequence to build a
"sequence membership" style quantum test from for this problem. Rather than
fabricate an OEIS id or copy a value with no real content, this script instead
builds a REAL, small, computable, finite search problem taken directly from
the problem's tags ("geometry", "distances"), which is exactly the flavor of
Erdos's distinct-distances problem: for n points, what is the minimum number
of distinct pairwise distances they can determine?

Chosen finite instance (small enough to search exactly, both classically and
on a quantum circuit):
    - Candidate point set: the 9 points of a 3x3 integer grid, {0,1,2}x{0,1,2}.
    - We look at all 4-point subsets of this grid (C(9,4) = 126 subsets).
    - For each subset, compute the number of DISTINCT pairwise Euclidean
      distances among its 6 pairs (squared-distance values are used to keep
      everything exact integer arithmetic; distinct squared distances are in
      1-1 correspondence with distinct distances since all values are
      nonnegative).
    - The classical property under test: the minimum possible number of
      distinct distances D_min achievable by any 4-point subset of this grid,
      and which subset indices (0..125, in a fixed enumeration) achieve it.

This is computed from first principles in this script (compute_min_distinct_distances
below), not copied from any table.

Quantum circuit: Grover search over a 7-qubit index register (2^7 = 128 >=
126 subsets) that marks exactly the indices whose 4-point subset attains
D_min, using a phase oracle built from the classical truth table (a real,
data-dependent oracle, not a hard-coded "always accept"). We run Grover with
the standard optimal number of iterations for the known number of solutions,
then check that the highest-probability measured indices are all elements of
the true classical solution set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit import transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no external data)
# ---------------------------------------------------------------------------

def build_grid_points():
    return [(x, y) for x in range(3) for y in range(3)]  # 9 points


def squared_dist(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def distinct_distance_count(subset):
    pairs = itertools.combinations(subset, 2)
    dists = {squared_dist(p, q) for p, q in pairs}
    return len(dists)


def compute_min_distinct_distances():
    points = build_grid_points()
    subsets = list(itertools.combinations(range(9), 4))  # 126 index-subsets
    assert len(subsets) == 126

    counts = []
    for idx_subset in subsets:
        subset_pts = [points[i] for i in idx_subset]
        counts.append(distinct_distance_count(subset_pts))

    d_min = min(counts)
    solution_indices = [i for i, c in enumerate(counts) if c == d_min]
    return points, subsets, counts, d_min, solution_indices


# ---------------------------------------------------------------------------
# 2. Grover search over the 126 subset indices, marking those achieving D_min
# ---------------------------------------------------------------------------

N_QUBITS = 7  # 2^7 = 128 >= 126


def build_oracle(marked_indices, n_qubits):
    """Phase oracle: flips the sign of every marked computational basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")
        # X on qubits that should be 0, so the marked state becomes |11...1>
        zero_qubits = [n_qubits - 1 - i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        qc.append(mcz, list(range(n_qubits)))
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
    qc.append(mcz, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_indices, n_qubits, valid_upper_bound, shots=4096):
    n_solutions = len(marked_indices)
    n_valid = valid_upper_bound  # only indices < 126 are meaningful

    # Optimal iteration count for amplitude amplification over the full
    # 2^n_qubits space (128 states), searching for n_solutions marked states.
    iterations = max(1, round((math.pi / 4) * math.sqrt(2 ** n_qubits / n_solutions)))

    oracle = build_oracle(marked_indices, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Sort measured bitstrings by frequency, take as many top results as
    # there are true solutions, and check they are all valid solutions.
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_indices = [int(bits, 2) for bits, _ in sorted_counts[:n_solutions]]
    return top_indices, counts, iterations


# ---------------------------------------------------------------------------
# 3. Main: compute classical answer, run quantum circuit, compare
# ---------------------------------------------------------------------------

def main():
    points, subsets, counts, d_min, solution_indices = compute_min_distinct_distances()

    print("Erdos problem #659 -- quantum-testable instance")
    print(f"Grid: 3x3 integer lattice, {len(points)} points")
    print(f"Subsets of size 4: {len(subsets)}")
    print(f"Classical minimum distinct-distance count D_min = {d_min}")
    print(f"Number of 4-point subsets achieving D_min: {len(solution_indices)}")
    print(f"Solution indices (classical, brute force): {solution_indices}")

    top_indices, raw_counts, iterations = run_grover(
        solution_indices, N_QUBITS, valid_upper_bound=len(subsets)
    )
    print(f"Grover iterations used: {iterations}")
    print(f"Top {len(top_indices)} measured indices from quantum circuit: {sorted(top_indices)}")

    solution_set = set(solution_indices)
    measured_set = set(top_indices)
    verified = measured_set.issubset(solution_set) and len(measured_set) == len(top_indices)

    # Also require that measured indices are the true global majority: i.e.
    # each of them individually has count no smaller than any non-solution
    # bitstring's count (sanity on amplification, not just set membership).
    non_solution_counts = [
        raw_counts.get(format(i, f"0{N_QUBITS}b"), 0)
        for i in range(2 ** N_QUBITS)
        if i not in solution_set
    ]
    min_solution_count = min(
        raw_counts.get(format(i, f"0{N_QUBITS}b"), 0) for i in solution_indices
    )
    amplification_ok = min_solution_count >= max(non_solution_counts, default=0)

    passed = verified and amplification_ok

    print()
    if passed:
        print("PASS: Grover search recovered exactly the classical minimum-distinct-distance "
              "subsets of the 3x3 grid, with amplified probability.")
    else:
        print("FAIL: quantum result did not match the classical brute-force answer.")

    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
