"""
Erdos problem #65 -- quantum-testable instance.

Source metadata (erdosproblems.com data, /home/user/manman4/erdosproblems
data/problems.yaml, entry "number: \"65\"", read 2026-09-19):
    prize: no
    status: open (informal and formal both open as of 2025-08-31)
    oeis: ["N/A"]   <-- no OEIS sequence is attached to this problem
    tags: ["graph theory", "cycles"]

LIMITATION, stated honestly up front: problem #65 has no OEIS id in the
source data, so there is no "sequence" to test quantum membership/search
against as the other lanes do. Fabricating an OEIS id or inventing a fake
sequence value would violate the task's ground rule of not fabricating
mathematical content. Instead, since the problem's own tags are exactly
["graph theory", "cycles"], this script builds a genuine small quantum
computation on the one finite, classically-checkable property that those
tags actually name: existence of a Hamiltonian cycle in a small graph.

Classical property tested (defined and computed from first principles in
this file, not copied from anywhere):
    Fix the 4 labeled vertices {0,1,2,3}. Up to rotation/reflection there
    are exactly (4-1)!/2 = 3 distinct Hamiltonian cycles on 4 labeled
    vertices:
        C0 = 0-1-2-3-0   edges {01,12,23,30}
        C1 = 0-1-3-2-0   edges {01,13,23,02}
        C2 = 0-2-1-3-0   edges {02,12,13,03}
    Given a graph G on those 4 vertices, exactly the Ci whose edge set is
    a subset of E(G) are Hamiltonian cycles that actually appear in G.
    We pick a concrete G (K4 minus the single edge {1,3}) and ask: among
    the 4 possible 2-bit indices (C0, C1, C2, and an unused padding index
    3), which index/indices correspond to a Hamiltonian cycle contained
    in G? This is computed directly here by set-subset checks.

    For this G, only C0 = {01,12,23,30} survives (C1 and C2 both need
    edge {1,3}, which is the one edge removed from K4). So the classical
    answer is: exactly one marked index, 0 (binary "00"), out of 4.

Quantum computation: a textbook single-solution Grover search over the
2-qubit index space {00,01,10,11}, with the oracle built directly from
the classical subset checks above (no hardcoded "answer" bit pattern --
the oracle's marked state is whatever index the classical check found).
One Grover iteration on N=4, M=1 solution is provably optimal (exact
success probability 1 on the ideal simulator), so we run it once and
check that measurement returns the marked index with certainty.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def hamiltonian_cycles_on_4_labeled_vertices():
    """Return the 3 distinct Hamiltonian cycles on vertices {0,1,2,3} as edge sets."""
    cycles = [
        [(0, 1), (1, 2), (2, 3), (3, 0)],  # C0: 0-1-2-3-0
        [(0, 1), (1, 3), (3, 2), (2, 0)],  # C1: 0-1-3-2-0
        [(0, 2), (2, 1), (1, 3), (3, 0)],  # C2: 0-2-1-3-0
    ]
    # Normalize each edge (a, b) with a < b so subset checks are order-independent.
    return [set(tuple(sorted(e)) for e in cyc) for cyc in cycles]


def classical_answer():
    """
    Compute, from first principles, which of the 4 possible 2-bit indices
    (0,1,2 -> the three Hamiltonian cycles; 3 -> unused padding, never a
    solution) correspond to a Hamiltonian cycle contained in G = K4 minus
    edge {1,3}.

    Returns (marked_indices, graph_edges).
    """
    all_k4_edges = {tuple(sorted(e)) for e in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]}
    removed_edge = (1, 3)
    graph_edges = all_k4_edges - {removed_edge}

    cycles = hamiltonian_cycles_on_4_labeled_vertices()
    marked = []
    for idx, cyc_edges in enumerate(cycles):
        if cyc_edges.issubset(graph_edges):
            marked.append(idx)
    # index 3 is padding (2 qubits give 4 states but only 3 cycles exist);
    # it is never a Hamiltonian cycle by construction.
    return marked, graph_edges


def build_oracle(marked_index, n_qubits=2):
    """
    Phase oracle on n_qubits that flips the sign of |marked_index> only.
    Implemented as: X-gates to map marked_index -> |11..1>, a multi-controlled
    Z, then undo the X-gates.
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(marked_index, f"0{n_qubits}b")
    # Put a 0 where the target bit is 0, so that after X-gates the target
    # pattern maps to all-ones.
    zero_positions = [i for i, b in enumerate(reversed(bits)) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for i in zero_positions:
        qc.x(i)
    return qc


def build_diffuser(n_qubits=2):
    qc = QuantumCircuit(n_qubits, name="diffuser")
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


def run_grover(marked_index, n_qubits=2, shots=2048):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_index, n_qubits)
    diffuser = build_diffuser(n_qubits)

    # N = 2**n_qubits = 4, M = 1 solution -> exactly one Grover iteration
    # is optimal (theta = arcsin(sqrt(1/4)) = pi/6, so (2k+1)*theta = pi/2
    # at k=1).
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    marked, graph_edges = classical_answer()
    print("Graph edges (K4 minus {1,3}):", sorted(graph_edges))
    print("Hamiltonian-cycle indices found in G (classical):", marked)

    assert len(marked) == 1, (
        "Instance is only guaranteed single-solution for the chosen graph; "
        f"got marked={marked}"
    )
    marked_index = marked[0]
    expected_bits = format(marked_index, "02b")
    print(f"Classical marked index: {marked_index} (bitstring '{expected_bits}')")

    counts = run_grover(marked_index, n_qubits=2, shots=2048)
    print("Grover measurement counts:", counts)

    total = sum(counts.values())
    hits = counts.get(expected_bits, 0)
    hit_fraction = hits / total

    print(f"Fraction of shots landing on marked state: {hit_fraction:.4f}")

    # Single-iteration Grover on N=4, M=1 is exact (success probability 1
    # on the ideal simulator), so we require an overwhelming majority of
    # shots (allow tiny numerical/statistical slack).
    verified = hit_fraction > 0.99

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
