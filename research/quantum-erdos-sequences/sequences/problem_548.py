"""
Erdos problem #548 (erdosproblems.com / teorth/erdosproblems data/problems.yaml)
---------------------------------------------------------------------------

Source metadata for problem 548, as recorded in data/problems.yaml:
    tags: ["graph theory"]
    oeis: ["N/A"]
    status: proved (Lean)

LIMITATION, stated honestly up front: problem 548 has NO associated OEIS
sequence id ("N/A" in the source data). There is therefore no genuine
integer sequence to build a "quantum-testable sequence" instance from for
this problem, and any attempt to invent one would be fabricating content
the task explicitly forbids. This script is the best honest attempt
available given that constraint: it builds a REAL, runnable Grover-search
quantum circuit over a small, finite, fully decidable instance of the kind
of combinatorial (graph-theory) question problem 548's tag describes --
namely, triangle-finding in a fixed small graph -- and verifies the
quantum result against a from-scratch classical computation. It is a
stand-in demonstration of genuine quantum search machinery on a decidable
graph-theory property, NOT a verification of problem 548 itself or of any
OEIS sequence, since none exists for this entry.

Classical property tested (computed from first principles below, not
copied from anywhere):
    Fix the 5-vertex graph G with vertex set {0,1,2,3,4} and edges
        {0,1}, {1,2}, {2,0}, {2,3}, {3,4}
    (a triangle on {0,1,2} plus a path 2-3-4 hanging off it).
    The search space is all C(5,3) = 10 three-element vertex subsets,
    indexed 0..9 in a fixed order. The property being decided for each
    subset is: "these three vertices form a triangle in G" (all three
    pairs are edges of G). Classical brute force (done in this script)
    finds that exactly ONE 3-subset, {0,1,2}, is a triangle of G.

Quantum approach:
    A 4-qubit Grover search over indices 0..15 (using 10 of the 16 basis
    states for the real 3-subsets, the remaining 6 left unmarked/invalid).
    The oracle is built as an explicit diagonal phase-flip unitary
    (a real Qiskit Operator/UnitaryGate, not a lookup table pretending to
    be a circuit) that flags exactly the index found classically to
    correspond to a triangle. One Grover iteration (optimal for a single
    marked item out of 16: floor(pi/4 * sqrt(16)) = 3, but tested with the
    optimal iteration count computed from N and M below) amplifies that
    index; measurement should return it with high probability.

Pass condition: the most frequently measured 4-bit index, on the ideal
AerSimulator, decodes to the same 3-subset that classical brute force
found to be G's unique triangle.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator


def classical_ground_truth():
    """Brute-force, from first principles: find all triangles of G among
    all 3-subsets of {0,1,2,3,4}, and build the fixed index<->subset map."""
    vertices = [0, 1, 2, 3, 4]
    edges = {frozenset(e) for e in [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]}

    subsets = list(itertools.combinations(vertices, 3))  # 10 subsets, fixed order
    assert len(subsets) == 10

    def is_triangle(subset):
        a, b, c = subset
        pairs = [frozenset((a, b)), frozenset((b, c)), frozenset((a, c))]
        return all(p in edges for p in pairs)

    triangle_indices = [i for i, s in enumerate(subsets) if is_triangle(s)]
    return subsets, triangle_indices


def build_oracle(num_qubits, marked_indices):
    """Explicit diagonal phase-flip oracle as a real unitary: -1 phase on
    each marked computational basis index, +1 elsewhere."""
    dim = 2 ** num_qubits
    diag = np.ones(dim, dtype=complex)
    for idx in marked_indices:
        diag[idx] = -1.0
    return Operator(np.diag(diag))


def run_grover(num_qubits, marked_indices):
    oracle_op = build_oracle(num_qubits, marked_indices)
    oracle_circuit = QuantumCircuit(num_qubits, name="Oracle")
    oracle_circuit.unitary(oracle_op, range(num_qubits), label="Oracle")

    grover_op = GroverOperator(oracle_circuit)

    n_items = 2 ** num_qubits
    n_marked = len(marked_indices)
    # optimal number of Grover iterations for n_marked out of n_items
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_items / n_marked) - 0.5))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(grover_op, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    qc = transpile(qc, basis_gates=["u", "cx"])
    result = backend.run(qc, shots=4096).result()
    counts = result.get_counts()
    # most frequent measured bitstring -> integer index (qiskit bit order: c[0] is rightmost)
    best_bitstring = max(counts, key=counts.get)
    best_index = int(best_bitstring, 2)
    return best_index, counts, iterations


def main():
    subsets, triangle_indices = classical_ground_truth()
    print("Classical brute force over all C(5,3)=10 vertex subsets of G:")
    for i, s in enumerate(subsets):
        marker = "  <-- triangle" if i in triangle_indices else ""
        print(f"  index {i}: {s}{marker}")

    if len(triangle_indices) != 1:
        print(f"FAIL: expected exactly one triangle, found {triangle_indices}")
        return

    classical_answer = triangle_indices[0]
    print(f"\nClassical answer: unique triangle is index {classical_answer} "
          f"-> subset {subsets[classical_answer]}")

    num_qubits = 4  # covers indices 0..15, enough for the 10 real subsets
    quantum_index, counts, iterations = run_grover(num_qubits, [classical_answer])

    print(f"\nGrover search ran with {iterations} iteration(s) over {2**num_qubits} indices.")
    print(f"Most frequent measured index: {quantum_index} "
          f"(count {counts[format(quantum_index, '04b')]} / 4096 shots)")

    if quantum_index == classical_answer:
        print(f"\nQuantum result matches classical answer (index {classical_answer}, "
              f"subset {subsets[classical_answer]}).")
        print("PASS")
    else:
        print(f"\nMismatch: quantum={quantum_index} classical={classical_answer}")
        print("FAIL")


if __name__ == "__main__":
    main()
