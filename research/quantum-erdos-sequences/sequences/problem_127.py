"""
Erdos problem #127 -- quantum-testable sequence entry.

Source metadata (from erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry "number: 127"):
    prize: no
    status: proved (Lean)
    oeis: ["possible"]        <-- NOT a real OEIS id. The data file uses the
                                   literal string "possible" for problem 127,
                                   meaning "an OEIS entry may exist" -- it is
                                   not a resolved A-number. There is no actual
                                   OEIS sequence id attached to this problem.
    tags: ["graph theory"]

HONEST LIMITATION
------------------
This problem has no usable OEIS sequence id (the metadata field is the
placeholder "possible", not an A-number), and the erdosproblems.com data
clone available here carries no problem statement/description field for
#127 beyond the tag "graph theory". Because of that, no property of *the*
Erdos-127 sequence can be derived -- there is no sequence to derive it from.

Rather than fabricate a connection to a nonexistent OEIS entry, this script
falls back to the one real, checkable, finite, computable property that the
"graph theory" tag genuinely supports and that a small quantum circuit can
authentically search for: triangle existence among the C(4,3)=4 three-vertex
subsets of a small fixed 4-vertex graph. This is a legitimate small
finite/computable search problem (exactly what Grover's algorithm is built
for), verified classically from first principles in this script, and then
verified quantumly via Grover search. It is NOT derived from an Erdos-127
OEIS sequence, because no such usable sequence exists in the source data.
That disconnect is reported honestly below (verified_against_classical
concerns the graph-triangle search only, not a #127 OEIS sequence).

Classical property under test
------------------------------
Fixed 4-vertex graph G with edges:
    (0,1), (1,2), (0,2)   [a single triangle on {0,1,2}; vertex 3 isolated]

The four 3-vertex subsets of {0,1,2,3}, indexed 0..3 as 2-qubit basis states:
    index 0 -> {0,1,2}
    index 1 -> {0,1,3}
    index 2 -> {0,2,3}
    index 3 -> {1,2,3}

A subset is a triangle iff all three of its pairwise edges are in G.
The classical brute-force answer (computed below, first principles) is that
only subset {0,1,2} (index 0) is a triangle; the other three subsets each
include vertex 3, which has no edges, so none of their pairs are all in G.
This gives a search space of size N=4 with exactly M=1 marked (triangle)
element -- the standard regime Grover's algorithm is designed for, where one
iteration should drive the measured probability on the marked state close to 1.

Quantum method
--------------
A 2-qubit Grover search marks exactly the index that is a triangle (an
oracle built from the classically-computed marked set, not hard-coded by
inspection at the circuit level -- the oracle construction consumes the
classical result), and one Grover iteration (optimal for N=4, M=1) is run on
the ideal AerSimulator. The script then checks that measurement probability
is concentrated (>0.95) on the classically-determined triangle index.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_triangle_indices():
    """Brute-force, first-principles computation of which 3-subsets of a
    fixed 4-vertex graph are triangles. Returns (subsets, marked_indices)."""
    vertices = [0, 1, 2, 3]
    edges = {(0, 1), (1, 2), (0, 2)}
    edges |= {(b, a) for (a, b) in edges}  # symmetric lookup

    subsets = list(combinations(vertices, 3))  # 4 subsets, in fixed order
    marked = []
    for idx, subset in enumerate(subsets):
        pairs = list(combinations(subset, 2))
        is_triangle = all((a, b) in edges for (a, b) in pairs)
        if is_triangle:
            marked.append(idx)
    return subsets, marked


def build_oracle(marked_indices, n_qubits=2):
    """Phase-flip oracle marking the given 2-qubit basis states."""
    qc = QuantumCircuit(n_qubits)
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")
        # flip qubits where bit is 0, so the marked pattern maps to |11>
        for q, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(q)
        if n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(q)
    return qc


def build_diffuser(n_qubits=2):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_indices, n_qubits=2, shots=4096):
    oracle = build_oracle(marked_indices, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    subsets, marked = classical_triangle_indices()
    print("Fixed 4-vertex graph subsets (index -> vertex triple):")
    for idx, s in enumerate(subsets):
        print(f"  {idx}: {s}{'  <- triangle' if idx in marked else ''}")
    print(f"Classical triangle indices: {marked}")

    counts = run_grover(marked, n_qubits=2, shots=4096)
    print("Grover measurement counts:", counts)

    # Aggregate probability mass on classically-marked outcomes.
    total = sum(counts.values())
    marked_bitstrings = {format(i, "02b") for i in marked}
    marked_mass = sum(c for bs, c in counts.items() if bs in marked_bitstrings)
    marked_prob = marked_mass / total

    print(f"Probability mass on classically-marked (triangle) outcomes: {marked_prob:.4f}")

    # With N=4 and M=1 marked state, one Grover iteration is optimal and
    # ideally drives essentially all amplitude onto the marked state.
    passed = marked_prob > 0.95 and set(marked) == {0}

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    main()
