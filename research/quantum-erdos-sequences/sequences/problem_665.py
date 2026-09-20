"""
Erdos problem #665 -- quantum-testable sequence entry (best-effort, LIMITED).

Source metadata (data/problems.yaml in manman4/erdosproblems, read-only clone):
    number: "665"
    prize: no
    status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["combinatorics"]

LIMITATION (read this before trusting the PASS below as "problem 665 verified"):
Problem #665 has no OEIS id in the source data (oeis: ["N/A"]) and no formal
statement is given in the metadata file -- it is marked "unformalized". There
is therefore no genuine, specific sequence or property belonging to problem
665 that this script can derive and check. Per the task instructions, rather
than fabricate a property and falsely attribute it to problem 665, this
script honestly documents that gap and instead builds a REAL, correctly
verified quantum circuit for a small, well-defined combinatorial search
problem of the same general shape as the "combinatorics" tag (subset-sum
existence), so the quantum-circuit machinery itself is genuine and checked
classically -- but the specific numeric instance is a stand-in, NOT a term of
problem 665's sequence, because no such sequence exists in the source data.

Chosen finite instance (stand-in combinatorial search, not OEIS-derived):
    Set S = {1, 2, 3, 4} (n = 4 elements -> search space size N = 2^4 = 16
    subsets, indexed by a 4-qubit register).
    Target sum T = 5.
    Property: "does there exist a subset of S whose elements sum to T?"
    This is exactly the kind of small, finite, computable combinatorial
    decision/search problem Grover's algorithm is built for.

Classical ground truth (computed here from first principles, brute force
over all 2^4 = 16 subsets):
    Subsets of {1,2,3,4} summing to 5: {1,4} and {2,3} -> marked indices are
    those two subsets. There are exactly 2 solutions out of 16.

Quantum approach: Grover's algorithm.
    - 4-qubit register enumerates all subsets of S (bit i = 1 means element
      i+1 is included).
    - A classically-derived oracle (built from the brute-force solution set,
      not hand-waved) marks exactly the subset-sum solutions with a phase
      flip.
    - Standard Grover diffusion operator amplifies the marked states.
    - With 2 solutions out of 16, the optimal number of Grover iterations is
      round(pi/4 * sqrt(16/2)) = 2.
    - Run on the ideal AerSimulator (statevector, no noise) and check that
      the measurement distribution is concentrated (essentially all
      probability mass) on the two classically-verified solution indices.

PASS/FAIL: compares the set of high-probability measured bitstrings against
the classically brute-forced solution set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_subset_sum_solutions(values, target):
    """Brute-force every subset of `values`; return the set of bitmask
    indices (bit i set <=> values[i] included) whose subset sums to target.
    """
    n = len(values)
    solutions = set()
    for mask in range(2 ** n):
        total = sum(values[i] for i in range(n) if (mask >> i) & 1)
        if total == target:
            solutions.add(mask)
    return solutions


def build_oracle(n_qubits, solution_masks):
    """Phase-flip oracle: for each solution bitmask, apply a multi-controlled
    Z (via X-sandwiching for 0-bits) so that exactly those computational
    basis states receive a -1 phase.
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    for mask in solution_masks:
        zero_bits = [i for i in range(n_qubits) if not (mask >> i) & 1]
        for i in zero_bits:
            qc.x(i)
        # multi-controlled Z on all n_qubits (flip phase of |11..1>)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_bits:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(values, target, shots=4096):
    n = len(values)
    solutions = classical_subset_sum_solutions(values, target)
    assert len(solutions) > 0, "instance must have at least one solution"

    n_solutions = len(solutions)
    N = 2 ** n
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / n_solutions)))

    oracle = build_oracle(n, solutions)
    diffuser = build_diffuser(n)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n), range(n))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    return counts, solutions, iterations


def main():
    values = [1, 2, 3, 4]
    target = 5

    classical_solutions = classical_subset_sum_solutions(values, target)
    print(f"Classical brute-force solutions (bitmask over {values}, "
          f"target sum {target}): {sorted(classical_solutions)}")
    for mask in sorted(classical_solutions):
        chosen = [values[i] for i in range(len(values)) if (mask >> i) & 1]
        print(f"  mask={mask:04b} -> subset {chosen} sums to {sum(chosen)}")

    counts, solutions, iterations = run_grover(values, target)
    print(f"\nGrover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    total_shots = sum(counts.values())
    # Qiskit bit order: rightmost char is qubit 0 -> convert to our mask
    # convention (bit i = qubit i) by reversing the bitstring.
    measured_masks = {}
    for bitstring, cnt in counts.items():
        mask = int(bitstring[::-1], 2)
        measured_masks[mask] = measured_masks.get(mask, 0) + cnt

    # High-probability outcomes: anything carrying at least 5% of the shots.
    threshold = 0.05 * total_shots
    high_prob_masks = {m for m, c in measured_masks.items() if c >= threshold}

    solution_mass = sum(
        c for m, c in measured_masks.items() if m in solutions
    )
    solution_fraction = solution_mass / total_shots

    print(f"\nHigh-probability measured masks: {sorted(high_prob_masks)}")
    print(f"Classical solution masks:        {sorted(solutions)}")
    print(f"Fraction of shots landing on a true solution: "
          f"{solution_fraction:.3f}")

    passed = (
        high_prob_masks == solutions
        and solution_fraction > 0.90
    )

    print("\nPASS" if passed else "\nFAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
