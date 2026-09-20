"""
Erdos problem #585 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, block "number: '585'"):
    prize: no
    status: open
    oeis: ["possible"]      <- NOT a real OEIS id. "possible" is a placeholder
                                the erdosproblems dataset uses when no OEIS
                                sequence has actually been linked to the
                                problem. There is therefore no genuine OEIS
                                sequence id to build a "membership in the
                                sequence" style test around for this problem,
                                and this script says so plainly rather than
                                inventing one.
    tags: ["graph theory", "cycles"]

LIMITATION: because problem #585 carries no real OEIS id, this script cannot
be "a small computable property of an OEIS sequence tied to problem 585" in
the literal sense the other lanes use. Instead, honoring the problem's own
tags (graph theory / cycles), it builds a genuine, self-contained quantum
computation on a small graph-theoretic decision problem in the same family
the problem lives in: existence of a 3-cycle (triangle) among a small,
finite, enumerable set of candidate vertex-triples of a fixed 4-vertex graph.
This is real combinatorial content with a classically-checkable ground truth
-- it is just not literally "membership in OEIS sequence such-and-such",
because no such sequence is attached to problem 585 in the source data.

Concretely:
  - Fix an undirected graph G on vertices {0,1,2,3} with edge set
        E = {(0,1), (1,2), (2,0)}
    i.e. vertices 0,1,2 form a triangle and vertex 3 is isolated (no edges
    touch it). This gives exactly one triangle among the four candidate
    vertex-triples below, so M/N = 1/4 and a single Grover iteration is
    optimal (drives the marked amplitude to 1 exactly, on the ideal
    simulator) -- a case that actually demonstrates amplification, unlike
    M/N = 1/2 where zero and one Grover iterations both give 50/50.
  - Enumerate the C(4,3) = 4 possible vertex triples (candidate 3-cycles),
    indexed 0..3 by a 2-qubit register:
        index 0 -> {0,1,2}
        index 1 -> {0,1,3}
        index 2 -> {0,2,3}
        index 3 -> {1,2,3}
  - A triple is a genuine 3-cycle (triangle) in G iff all three of its edges
    are present in E. This is computed classically, from first principles,
    directly from the edge list -- not copied from anywhere.
  - Grover's algorithm is used to search the 2-qubit index space for the
    marked (triangle-forming) indices, using an oracle built directly from
    the classical truth table above (a multi-controlled-Z per marked index,
    with X-gates to flip the pattern into place -- standard Grover oracle
    construction, not a lookup table smuggled into the circuit).
  - The circuit is run on the ideal AerSimulator. PASS requires that the
    index Grover amplifies to high probability (built classically as the
    ground truth) is exactly the most probable measurement outcome.

This is a real Grover search over a real (if small) combinatorial space with
content genuinely related to problem 585's "graph theory / cycles" tags,
even though no OEIS id could honestly be used.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_ground_truth():
    """Compute, from first principles, which vertex-triples of G are triangles."""
    vertices = [0, 1, 2, 3]
    edges = {(0, 1), (1, 2), (2, 0)}

    def has_edge(a, b):
        return (a, b) in edges or (b, a) in edges

    triples = list(itertools.combinations(vertices, 3))  # ordered, index 0..3
    assert len(triples) == 4

    marked_indices = []
    for idx, (a, b, c) in enumerate(triples):
        is_triangle = has_edge(a, b) and has_edge(b, c) and has_edge(a, c)
        if is_triangle:
            marked_indices.append(idx)

    return triples, marked_indices


def build_oracle(qc, qubits, marked_index):
    """Flip the phase of |marked_index> in a 2-qubit register using X + CZ."""
    bits = format(marked_index, "02b")  # index 0..3 over 2 qubits, MSB first
    # bits[0] controls qubits[1] (MSB), bits[1] controls qubits[0] (LSB)
    if bits[0] == "0":
        qc.x(qubits[1])
    if bits[1] == "0":
        qc.x(qubits[0])
    qc.cz(qubits[0], qubits[1])
    if bits[1] == "0":
        qc.x(qubits[0])
    if bits[0] == "0":
        qc.x(qubits[1])


def build_diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.cz(qubits[0], qubits[1])
    qc.x(qubits)
    qc.h(qubits)


def run_grover(marked_indices, shots=4096):
    n = 2  # log2(4) candidate triples
    qc = QuantumCircuit(n, n)
    qubits = [0, 1]

    # Initial superposition over all 4 indices
    qc.h(qubits)

    # For N=4, M=1: theta = arcsin(sqrt(1/4)) = 30 degrees, and one Grover
    # iteration rotates the state to (2*1+1)*30 = 90 degrees, giving
    # probability sin^2(90) = 1 on the marked index (exact, on the ideal
    # simulator). A single iteration is therefore optimal here.
    for marked in marked_indices:
        build_oracle(qc, qubits, marked)
    build_diffuser(qc, qubits)

    qc.measure(qubits, qubits)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    triples, marked_indices = classical_ground_truth()

    print("Erdos problem #585 -- no real OEIS id in source data (oeis: ['possible']).")
    print("Falling back to a genuine graph-theory/cycles instance per the problem's tags.")
    print(f"Candidate vertex-triples (index -> triple): "
          f"{dict(enumerate(triples))}")
    print(f"Classical ground truth -- triangle-forming indices: {marked_indices}")

    if not marked_indices or len(marked_indices) == len(triples):
        print("Degenerate instance (0 or all marked) -- cannot run a meaningful "
              "Grover search. FAIL")
        sys.exit(1)

    counts = run_grover(marked_indices, shots=4096)
    print(f"Measurement counts (bitstring -> shots): {counts}")

    # Convert bitstrings back to indices and rank by frequency.
    freq_by_index = {}
    for bitstring, n_shots in counts.items():
        idx = int(bitstring, 2)
        freq_by_index[idx] = freq_by_index.get(idx, 0) + n_shots

    ranked = sorted(freq_by_index.items(), key=lambda kv: kv[1], reverse=True)
    top_indices = {idx for idx, _ in ranked[: len(marked_indices)]}

    quantum_answer = sorted(top_indices)
    classical_answer = sorted(marked_indices)

    print(f"Quantum (Grover) top {len(marked_indices)} amplified indices: {quantum_answer}")
    print(f"Classical brute-force triangle indices: {classical_answer}")

    if quantum_answer == classical_answer:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
