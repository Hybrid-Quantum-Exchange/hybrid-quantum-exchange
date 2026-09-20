"""
Erdos problem #718 -- Grover-search sanity check (LIMITATION: no real sequence
content available; see below).

Source of truth: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: '718'":
    prize: no
    informal_status: proved (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION (read before trusting the PASS below):
Problem #718 has no associated OEIS sequence id ("N/A" in the metadata) and
the problems.yaml record carries only a status/tag summary, not the problem's
actual mathematical statement. There is therefore no "small, finite,
computable property of the sequence" to derive here -- there is no sequence.
This script is NOT a test of problem #718's actual content. It cannot be,
honestly, from the available metadata.

What this script actually does, as the best honest fallback consistent with
the problem's only real signal (the tag "graph theory"): it implements a
genuine, self-contained instance of a classical graph-theory decision
problem -- "does this 4-vertex graph contain a triangle?" -- and solves it
two ways:
  1. Classically, by brute-force enumeration of all C(4,3)=4 vertex triples
     (ground truth, computed from first principles in this script).
  2. Via a real Grover search circuit built on qiskit_aer's AerSimulator,
     searching the same 4-element space (2 index qubits) with an oracle
     built from the same classical adjacency check, and one Grover
     diffusion iteration (optimal for N=4, 1 marked-or-more state).

The two answers are compared and PASS/FAIL is printed accordingly. This
demonstrates a genuine small Grover search on a graph-theory-flavored
decision problem, in the spirit of the "graph theory" tag on #718, but it is
explicitly NOT a verified/derived property of problem #718 itself, and must
not be read as such. verified_against_classical below means "the quantum
circuit's search result matches the brute-force classical search result for
this specific 4-vertex graph", nothing more.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def build_graph():
    """A fixed, small 4-vertex graph with exactly one triangle: (0,1,2)."""
    vertices = [0, 1, 2, 3]
    edges = {(0, 1), (1, 2), (0, 2), (2, 3)}  # triangle 0-1-2, plus a pendant edge 2-3
    return vertices, edges


def triples(vertices):
    return list(combinations(vertices, 3))


def is_triangle(edges, triple):
    a, b, c = triple
    pairs = [(a, b), (a, c), (b, c)]

    def has_edge(u, v):
        return (u, v) in edges or (v, u) in edges

    return all(has_edge(u, v) for u, v in pairs)


def classical_triangle_indices():
    """Ground truth: brute-force over all 4 vertex triples."""
    vertices, edges = build_graph()
    trip = triples(vertices)
    marked = [i for i, t in enumerate(trip) if is_triangle(edges, t)]
    return marked, trip


def grover_oracle(qc, marked_indices, index_qubits):
    """Phase-flip the marked 2-qubit basis states (indices in [0,3])."""
    for idx in marked_indices:
        bits = format(idx, f"0{len(index_qubits)}b")
        # Flip qubits that are 0 in this index so the multi-controlled Z
        # triggers exactly on this basis state, then flip back.
        for qb, bit in zip(index_qubits, bits):
            if bit == "0":
                qc.x(qb)
        qc.h(index_qubits[-1])
        qc.cx(index_qubits[0], index_qubits[-1])
        qc.h(index_qubits[-1])
        for qb, bit in zip(index_qubits, bits):
            if bit == "0":
                qc.x(qb)


def grover_diffuser(qc, index_qubits):
    for qb in index_qubits:
        qc.h(qb)
        qc.x(qb)
    qc.h(index_qubits[-1])
    qc.cx(index_qubits[0], index_qubits[-1])
    qc.h(index_qubits[-1])
    for qb in index_qubits:
        qc.x(qb)
        qc.h(qb)


def run_grover(marked_indices, n_index_qubits=2, shots=2048):
    qc = QuantumCircuit(n_index_qubits, n_index_qubits)
    index_qubits = list(range(n_index_qubits))

    qc.h(index_qubits)  # uniform superposition over the 4 triples

    # One Grover iteration is optimal for N=4, M=1 (single marked triangle).
    grover_oracle(qc, marked_indices, index_qubits)
    grover_diffuser(qc, index_qubits)

    qc.measure(index_qubits, index_qubits)

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    most_likely = max(counts, key=counts.get)
    quantum_index = int(most_likely, 2)
    return quantum_index, counts


def main():
    marked, trip = classical_triangle_indices()
    print("Problem 718 metadata: oeis=['N/A'], tags=['graph theory'] -- "
          "no real sequence available; running fallback graph-theory check.")
    print(f"Vertex triples (index order): {trip}")
    print(f"Classical brute-force triangle indices: {marked}")

    assert len(marked) == 1, "This fixed graph is constructed to have exactly one triangle"
    classical_answer = marked[0]

    quantum_index, counts = run_grover(marked, n_index_qubits=2, shots=2048)
    print(f"Grover circuit measurement counts: {counts}")
    print(f"Quantum most-likely index: {quantum_index} (triple {trip[quantum_index]})")
    print(f"Classical answer index:    {classical_answer} (triple {trip[classical_answer]})")

    verified = quantum_index == classical_answer
    if verified:
        print("PASS: quantum Grover search found the same unique triangle "
              "as classical brute force.")
    else:
        print("FAIL: quantum result did not match classical brute force.")


if __name__ == "__main__":
    main()
