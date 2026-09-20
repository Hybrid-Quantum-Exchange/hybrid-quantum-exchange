"""
Erdos problem #717 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, data/problems.yaml, entry
`number: "717"`):
    tags: ["graph theory"]
    oeis: ["N/A"]
    status: proved (informal), unformalized (formal)

LIMITATION, stated honestly up front: problem #717 carries no OEIS sequence
id in the source data (`oeis: ["N/A"]`) and the repository's metadata file
gives no problem statement, only the tag "graph theory". There is therefore
no specific integer sequence tied to problem #717 to build a quantum test
around, and no way to derive a property "of the sequence" because there is
no sequence. Per the task's own fallback instructions, this script is a
best-honest-attempt rather than a genuine test of problem #717's content:
it implements a real, finite, classically-verifiable graph-theory decision
property (triangle existence in a small fixed graph) -- chosen because it
sits squarely inside the "graph theory" tag attached to #717 -- and searches
for it with a real Grover quantum circuit on the ideal AerSimulator. This is
NOT a derivation of anything specific to problem #717's actual (unstated)
content; it is offered only so the lane produces a genuine, runnable quantum
computation rather than a fabricated one.

Classical property under test
------------------------------
Fix a 5-vertex graph G with vertex set {0,1,2,3,4} and edge set
    E = {(0,1), (1,2), (0,2), (2,3), (3,4)}
(so {0,1,2} is the graph's unique triangle; no other 3-subset of vertices
is pairwise connected). Enumerate all C(5,3) = 10 three-vertex subsets in a
fixed order and index them 0..9 (4 qubits, values 0..15; indices 10..15 are
unused padding and are never marked). Define the property

    triangle(i) = True iff the i-th 3-subset's three vertices are pairwise
                  adjacent in G

The classical answer (computed here from first principles, no lookup) is
the set of indices i for which triangle(i) holds. For this G that set is
computed by brute force below.

Quantum circuit
----------------
A 4-qubit Grover search over the 16 basis states of the index register.
The oracle is a phase-flip (multi-controlled Z, with X-gates to match each
marked index's bit pattern) applied to exactly the indices found triangle
in the classical brute-force scan above -- so the oracle is built directly
from the classically-verified answer, not asserted independently. One
Grover diffusion step follows (N=16, ~1 marked state => optimal iterations
round(pi/4 * sqrt(16/1)) = 3, computed below rather than hard-coded).

Pass/fail
---------
Run the circuit on AerSimulator (statevector-exact, no noise), take the
most-probable measured index, and PASS iff it equals the classically
computed unique triangle index.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_triangle_indices():
    """Brute-force, from first principles: which 3-subsets of {0..4} form
    a triangle in G? Returns (ordered list of all 10 triples, list of
    indices into that list that are triangles)."""
    vertices = [0, 1, 2, 3, 4]
    edges = {(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)}

    def adjacent(a, b):
        return (a, b) in edges or (b, a) in edges

    triples = list(combinations(vertices, 3))  # 10 triples, fixed order
    triangle_idx = []
    for i, (a, b, c) in enumerate(triples):
        if adjacent(a, b) and adjacent(b, c) and adjacent(a, c):
            triangle_idx.append(i)
    return triples, triangle_idx


def build_oracle(qc, qubits, marked_index, n_qubits):
    """Phase-flip the single computational basis state |marked_index>
    (an n_qubits-bit binary pattern) using X-sandwiched multi-controlled Z."""
    bits = format(marked_index, f"0{n_qubits}b")
    # bits[0] is qubit n_qubits-1 (MSB) in this convention; apply to all qubits
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    for pos in zero_positions:
        qc.x(qubits[n_qubits - 1 - pos])
    if n_qubits == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for pos in zero_positions:
        qc.x(qubits[n_qubits - 1 - pos])


def build_diffuser(qc, qubits, n_qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def run_grover(marked_index, n_qubits=4, shots=2048):
    n_states = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_states / 1)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))
    qc.h(qubits)
    for _ in range(iterations):
        build_oracle(qc, qubits, marked_index, n_qubits)
        build_diffuser(qc, qubits, n_qubits)
    qc.measure(qubits, qubits)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    best_bitstring = max(counts, key=counts.get)
    measured_index = int(best_bitstring, 2)
    return measured_index, counts, iterations


def main():
    triples, triangle_indices = classical_triangle_indices()
    print(f"Triples (index: vertices): {list(enumerate(triples))}")
    print(f"Classically found triangle indices: {triangle_indices}")

    assert len(triangle_indices) == 1, (
        "Instance must have exactly one triangle for a clean single-target "
        f"Grover search; got {triangle_indices}"
    )
    classical_answer = triangle_indices[0]

    measured_index, counts, iterations = run_grover(classical_answer)
    total_shots = sum(counts.values())
    prob_correct = counts.get(format(classical_answer, "04b"), 0) / total_shots

    print(f"Grover iterations used: {iterations}")
    print(f"Measured (most frequent) index: {measured_index}")
    print(f"Classical answer (unique triangle index): {classical_answer}")
    print(f"P(measure classical answer) over {total_shots} shots: {prob_correct:.4f}")

    passed = (measured_index == classical_answer) and (prob_correct > 0.5)
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
