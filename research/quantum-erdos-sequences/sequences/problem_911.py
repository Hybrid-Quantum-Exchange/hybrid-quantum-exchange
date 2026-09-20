"""
Erdos problem #911 -- quantum-testable sequence lane.

Source metadata (data/problems.yaml, entry "number: \"911\""):
    tags: ["graph theory", "ramsey theory"]
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #911 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no OEIS-derived
numeric property to hand to a quantum circuit for this problem specifically.
Per instructions, rather than fabricate an OEIS-backed property, this script
falls back to the best honest attempt available from the problem's own tags
("graph theory", "ramsey theory"): it builds a genuine, small, finite,
computable decision problem drawn directly from Ramsey theory -- the same
area problem #911 is tagged with -- and solves it with a real Grover search
circuit run on AerSimulator, checked against a from-scratch classical
computation.

Classical property being tested
--------------------------------
Let K5 be the complete graph on 5 labeled vertices (10 edges). Each edge is
colored red (0) or blue (1), giving a 10-bit string ("coloring"). A coloring
is GOOD if none of the 10 triangles of K5 is monochromatic (all three of its
edges the same color). This is exactly the finite instance behind the
classical Ramsey fact R(3,3) = 6: K5 admits a 2-coloring with no
monochromatic triangle, while K6 does not (any 2-coloring of K6 forces one).

This script:
  1. Computes classically, from first principles (brute force over all
     2^10 = 1024 edge colorings, testing all C(5,3) = 10 triangles), the
     exact set of GOOD colorings of K5, and their count M.
  2. Builds a Grover search circuit over 10 qubits (one per edge) whose
     oracle is the exact diagonal unitary flipping the phase of precisely
     the GOOD colorings (computed classically in step 1 -- not guessed),
     with the standard number of Grover iterations for search space
     N = 1024 and M marked items.
  3. Runs the circuit on the ideal AerSimulator, measures, and checks that
     the most frequently sampled bitstrings are all GOOD colorings (i.e.
     the quantum search actually finds real solutions to the classical
     Ramsey problem), and that no bad (monochromatic-triangle) coloring
     appears among the top results.
  4. Prints PASS or FAIL based on that comparison.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate, MCXGate
from qiskit_aer import AerSimulator


def classical_good_colorings(n_vertices=5):
    """Brute-force, from first principles, all 2-colorings of K_n with no
    monochromatic triangle. Returns (edges, list_of_good_bitstrings_as_int).
    """
    edges = list(itertools.combinations(range(n_vertices), 2))
    edge_index = {e: i for i, e in enumerate(edges)}
    triangles = list(itertools.combinations(range(n_vertices), 3))

    def triangle_edge_indices(t):
        a, b, c = t
        return (edge_index[(a, b)], edge_index[(a, c)], edge_index[(b, c)])

    tri_idx = [triangle_edge_indices(t) for t in triangles]

    n_edges = len(edges)
    good = []
    for bits in range(2 ** n_edges):
        ok = True
        for e1, e2, e3 in tri_idx:
            c1 = (bits >> e1) & 1
            c2 = (bits >> e2) & 1
            c3 = (bits >> e3) & 1
            if c1 == c2 == c3:
                ok = False
                break
        if ok:
            good.append(bits)
    return edges, good


def build_diagonal_oracle(n_qubits, marked_states):
    """Exact diagonal phase oracle: -1 on marked computational basis states,
    +1 elsewhere. Built directly from the classically-computed marked set,
    not approximated.
    """
    diag = [1.0] * (2 ** n_qubits)
    for s in marked_states:
        diag[s] = -1.0
    return DiagonalGate(diag)


def build_diffuser(n_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits - 1 >= 1:
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    else:
        qc.z(0)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    n_vertices = 5
    edges, good = classical_good_colorings(n_vertices)
    n_edges = len(edges)
    N = 2 ** n_edges
    M = len(good)
    good_set = set(good)

    print(f"Instance: K{n_vertices}, edges={n_edges}, search space N={N}")
    print(f"Classical result: {M} monochromatic-triangle-free colorings out of {N}")
    assert M > 0, "classical search found no solutions -- cannot build Grover instance"

    # Standard optimal iteration count for Grover with N items, M marked.
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / 4 / theta) - 0.5))

    oracle = build_diagonal_oracle(n_edges, good)
    diffuser = build_diffuser(n_edges)

    qc = QuantumCircuit(n_edges, n_edges)
    qc.h(range(n_edges))
    for _ in range(iterations):
        qc.append(oracle, range(n_edges))
        qc.append(diffuser, range(n_edges))
    qc.measure(range(n_edges), range(n_edges))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Bit ordering: qiskit prints classical register with qubit 0 as the
    # rightmost character.
    def bitstring_to_int(bs):
        return int(bs, 2)

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_k = min(M, 10)
    top_states = [bitstring_to_int(bs) for bs, _ in sorted_counts[:top_k]]

    hits = sum(1 for s in top_states if s in good_set)
    total_top_shots = sum(c for _, c in sorted_counts[:top_k])
    amplified_prob = total_top_shots / shots

    print(f"Grover iterations used: {iterations}")
    print(f"Top-{top_k} most frequent measured colorings are GOOD: {hits}/{top_k}")
    print(f"Fraction of shots landing on the top-{top_k} bucket: {amplified_prob:.3f}")

    quantum_found_valid_solution = any(s in good_set for s, _ in
                                        [(bitstring_to_int(bs), c) for bs, c in sorted_counts[:1]])
    verified = (hits == top_k) and quantum_found_valid_solution

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
