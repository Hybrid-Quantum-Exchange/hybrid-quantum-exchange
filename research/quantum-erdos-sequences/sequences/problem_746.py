"""
Erdos problem #746 (erdosproblems.com), quantum-testable lane.

Source metadata (from manman4/erdosproblems data/problems.yaml, entry
`number: "746"`): prize "no", status "proved" (last_update 2025-08-31),
tags ["graph theory"], oeis: ["N/A"]. There is NO OEIS sequence id attached
to this problem -- the listed value is the literal string "N/A", and no
statement text is present in that data file for this entry either. Per the
task instructions, an honest best attempt is written here rather than a
fabricated sequence property, and the limitation is stated explicitly: this
script does NOT test any real term membership of an actual
Erdos-problem-746 OEIS sequence, because none exists in the source data.

Given the only real signal available -- the tag "graph theory" -- the best
small, finite, computable stand-in property genuinely in that spirit is a
basic existence question on small graphs:

    Property tested: among all labeled simple graphs on 4 vertices
    (there are C(4,2) = 6 possible edges, so 2^6 = 64 graphs, encoded
    directly as 6-qubit basis states, one qubit per potential edge),
    find every graph that contains a triangle (a set of 3 mutually
    adjacent vertices). "Does a small graph contain a triangle" is a
    genuine, classical graph-theory decision property with real
    combinatorial content, and a natural fit for a problem tagged
    "graph theory".

Classical answer (computed here in the script, from first principles):
enumerate all 64 edge-subsets of K4, and for each, check all four possible
triangles ({0,1,2}, {0,1,3}, {0,2,3}, {1,2,3}) for full adjacency. This
enumeration is the ground truth used both to build the Grover oracle and to
check the quantum result against.

Quantum approach: Grover's search algorithm on 6 qubits (one per edge of
K4). The oracle marks exactly the basis states (edge-subsets) that contain
at least one triangle; the diffusion operator amplifies those marked
amplitudes; the ideal AerSimulator counts are checked against the classical
enumeration -- every classically-marked graph must be among the top
measured outcomes, with strictly higher counts than every unmarked graph.

Limitation: this is a faithful small Grover search over a "contains a
triangle" property on 4-vertex graphs, not a test of a specific documented
OEIS integer sequence for Erdos problem #746 (none is listed, and no
problem statement text is present in the source data to derive one from
instead). ran_ok and verified_against_classical are reported honestly for
what this script actually checks.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_VERTICES = 4
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 6 edges, index = qubit
N_EDGES = len(EDGES)
assert N_EDGES == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(range(N_VERTICES), 3))  # 4 possible triangles


def edge_of(u: int, v: int):
    return EDGE_INDEX[(u, v) if u < v else (v, u)]


def graph_has_triangle(edge_bits: str) -> bool:
    """edge_bits[k] == '1' means edge EDGES[k] is present."""
    present = lambda u, v: edge_bits[edge_of(u, v)] == "1"
    for (a, b, c) in TRIANGLES:
        if present(a, b) and present(a, c) and present(b, c):
            return True
    return False


def classical_marked_graphs():
    """Enumerate all 2^6 edge-subsets of K4 and mark those containing a triangle."""
    marked = []
    for bits_int in range(1 << N_EDGES):
        edge_bits = format(bits_int, f"0{N_EDGES}b")  # edge_bits[0] = EDGES[0], ..., MSB-first string
        if graph_has_triangle(edge_bits):
            marked.append(edge_bits)
    return marked


def build_oracle_phase(marked_bitstrings, total_qubits: int) -> QuantumCircuit:
    """Multi-controlled-Z oracle: applies -1 phase to each marked basis state.

    marked_bitstrings[k] uses index 0 -> EDGES[0] (i.e. string position 0 = qubit 0).
    """
    qc = QuantumCircuit(total_qubits, name="oracle")
    for pattern in marked_bitstrings:
        zero_qubits = [q for q, b in enumerate(pattern) if b == "0"]
        for q in zero_qubits:
            qc.x(q)

        qc.h(total_qubits - 1)
        qc.append(MCXGate(total_qubits - 1), list(range(total_qubits - 1)) + [total_qubits - 1])
        qc.h(total_qubits - 1)

        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(total_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(total_qubits, name="diffuser")
    qc.h(range(total_qubits))
    qc.x(range(total_qubits))
    qc.h(total_qubits - 1)
    qc.append(MCXGate(total_qubits - 1), list(range(total_qubits - 1)) + [total_qubits - 1])
    qc.h(total_qubits - 1)
    qc.x(range(total_qubits))
    qc.h(range(total_qubits))
    return qc


def run():
    total_qubits = N_EDGES  # 6

    marked_edge_bits = classical_marked_graphs()  # strings, position k -> EDGES[k]
    assert marked_edge_bits, "classical search found no triangle-containing graphs — degenerate instance"
    print(f"Classical answer: {len(marked_edge_bits)} of {1 << N_EDGES} labeled 4-vertex graphs contain a triangle.")
    print("Edge order (qubit index -> vertex pair):", list(enumerate(EDGES)))

    num_marked = len(marked_edge_bits)
    search_space_size = 1 << N_EDGES

    # Oracle pattern needs qubit-index-ordered strings: position q -> qubit q.
    # edge_bits as built already has position k == EDGES[k] == qubit k, so use directly.
    oracle_patterns = marked_edge_bits

    theta = math.asin(math.sqrt(num_marked / search_space_size))
    optimal_iters = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle_phase(oracle_patterns, total_qubits)
    diffuser = build_diffuser(total_qubits)

    qc = QuantumCircuit(total_qubits, total_qubits)
    qc.h(range(total_qubits))
    for _ in range(optimal_iters):
        qc.append(oracle.to_instruction(), range(total_qubits))
        qc.append(diffuser.to_instruction(), range(total_qubits))
    qc.measure(range(total_qubits), range(total_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=4096, seed_simulator=42).result()
    counts = result.get_counts()

    def counts_key_to_edge_bits(bs: str) -> str:
        # Qiskit counts keys print qubit(total-1) ... qubit0, left to right.
        # Our patterns use position k == qubit k, so reverse the counts key.
        return bs[::-1]

    graph_counts = {}
    for bs, c in counts.items():
        edge_bits = counts_key_to_edge_bits(bs)
        graph_counts[edge_bits] = graph_counts.get(edge_bits, 0) + c

    marked_set = set(marked_edge_bits)
    sorted_graphs = sorted(graph_counts.items(), key=lambda kv: -kv[1])
    print("\nTop measured graphs (edge bitstrings) by frequency:")
    for g, c in sorted_graphs[:10]:
        marker = " <-- MARKED (has triangle)" if g in marked_set else ""
        print(f"  {g}: {c}{marker}")

    top_n = sorted_graphs[:num_marked]
    top_graphs = set(g for g, _ in top_n)

    all_marked_on_top = top_graphs == marked_set
    min_marked_count = min(graph_counts.get(g, 0) for g in marked_set)
    max_unmarked_count = max(
        (c for g, c in graph_counts.items() if g not in marked_set), default=0
    )
    dominance_ok = min_marked_count > max_unmarked_count

    verified = all_marked_on_top and dominance_ok

    print(f"\nGrover iterations used: {optimal_iters}")
    print(f"Number of marked (triangle-containing) graphs: {num_marked} / {search_space_size}")
    print(f"Top-{num_marked} measured graphs match marked set exactly: {all_marked_on_top}")
    print(f"Every marked graph outcounts every unmarked graph: {dominance_ok}")

    if verified:
        print("\nPASS")
    else:
        print("\nFAIL")

    return verified


if __name__ == "__main__":
    ok = run()
    if not ok:
        raise SystemExit(1)
