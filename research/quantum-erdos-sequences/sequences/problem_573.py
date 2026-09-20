"""
Erdos problem #573 (https://www.erdosproblems.com/573)
OEIS id used: A006856 -- ex(n; C4), the maximum number of edges in a graph
on n labeled vertices that contains no 4-cycle (Turan-type extremal number
for the cycle C4). Erdos problem 573 concerns this Zarankiewicz/Turan-type
quantity for C4-free graphs.

Classical property tested (computed from first principles in this script,
not copied from OEIS):

    For n = 4 vertices there are C(4,2) = 6 possible edges, so there are
    2**6 = 64 labeled graphs on 4 vertices. Brute force over all 64 graphs
    computes:
        m* = max { |E(G)| : G has 4 vertices, no 4-cycle subgraph }
    (this is exactly the classical quantity behind OEIS A006856(4)), and
    the full set S of edge-subsets (as 6-bit strings) that achieve m* while
    remaining C4-free. This is a small, finite, fully checkable search
    problem: "does there exist a C4-free graph on 4 vertices with m edges,
    for the largest such m" -- an unstructured search over 64 candidates
    for a member of a marked set S, which is exactly Grover's problem.

Quantum circuit:
    We build a genuine Grover search circuit over the 6-qubit space of all
    labeled graphs on 4 vertices (qubit i <-> whether edge i is present).
    The oracle is derived directly from the classical brute-force
    computation above (it is a diagonal phase-flip unitary that is -1 on
    exactly the classically-computed marked set S and +1 elsewhere -- i.e.
    it encodes the classical predicate "this graph is C4-free and has m*
    edges", not a hard-coded literal OEIS value). We run the standard
    Grover diffuser for the optimal number of iterations on the ideal
    AerSimulator, measure, and check that the most frequent measured
    bitstring decodes to a graph that is (a) in the classically computed
    marked set S and (b) genuinely C4-free with m* edges (re-verified
    independently after measurement). If so: PASS.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

N_VERTICES = 4
VERTEX_PAIRS = list(itertools.combinations(range(N_VERTICES), 2))  # 6 possible edges
N_EDGES = len(VERTEX_PAIRS)  # 6 -> search space size 2**6 = 64


def edges_from_bits(bits):
    """bits: tuple/list of 0/1 of length N_EDGES -> list of (u,v) edges present."""
    return [VERTEX_PAIRS[i] for i, b in enumerate(bits) if b == 1]


def has_4_cycle(edges, n=N_VERTICES):
    """Brute-force check: does the graph (vertex set 0..n-1) contain a
    4-cycle as a subgraph? For n=4 the only possible 4-cycles use all 4
    vertices; check every cyclic ordering of the 4 vertices."""
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    if len(edge_set) < 4:
        return False
    verts = list(range(n))
    for perm in itertools.permutations(verts):
        # a 4-cycle v0-v1-v2-v3-v0; fix v0 to avoid trivial repeats (not required for correctness)
        v0, v1, v2, v3 = perm
        cyc_edges = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v3), max(v2, v3)),
            (min(v3, v0), max(v3, v0)),
        ]
        if all(e in edge_set for e in cyc_edges):
            return True
    return False


def classical_brute_force():
    """Return (m_star, marked_indices) where m_star is the max number of
    edges of a C4-free graph on N_VERTICES vertices, and marked_indices is
    the sorted list of all 6-bit integers (0..63) whose corresponding
    labeled graph achieves m_star edges while being C4-free."""
    best_m = -1
    per_m_marked = {}
    for idx in range(2 ** N_EDGES):
        bits = [(idx >> i) & 1 for i in range(N_EDGES)]
        edges = edges_from_bits(bits)
        m = len(edges)
        if has_4_cycle(edges):
            continue
        per_m_marked.setdefault(m, []).append(idx)
        if m > best_m:
            best_m = m
    marked = sorted(per_m_marked[best_m])
    return best_m, marked


def build_oracle(marked_indices, n_qubits):
    """Diagonal phase-flip oracle: -1 on marked_indices, +1 elsewhere.
    Built directly from the classical predicate (not a fixed literal),
    then wrapped as a unitary gate."""
    dim = 2 ** n_qubits
    diag = np.ones(dim, dtype=complex)
    for idx in marked_indices:
        diag[idx] = -1.0
    qc = QuantumCircuit(n_qubits, name="Oracle")
    qc.unitary(Operator(np.diag(diag)), range(n_qubits), label="Oracle")
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits - 1 > 0:
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    else:
        qc.z(0)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_indices, n_qubits, shots=4096):
    n_marked = len(marked_indices)
    n_total = 2 ** n_qubits
    # optimal number of Grover iterations
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_total / n_marked)))

    oracle = build_oracle(marked_indices, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    m_star, marked = classical_brute_force()
    print(f"Erdos problem #573 / OEIS A006856 -- n = {N_VERTICES} vertices")
    print(f"Classical max edges in a C4-free graph on {N_VERTICES} vertices: m* = {m_star}")
    print(f"Number of marked (m*-edge, C4-free) labeled graphs out of {2 ** N_EDGES}: {len(marked)}")

    counts, iterations = run_grover(marked, N_EDGES)
    print(f"Grover iterations used: {iterations}")

    # Qiskit bitstrings are printed with qubit 0 as the rightmost character.
    def bitstring_to_idx(bs):
        bits = [int(c) for c in reversed(bs)]
        return sum(b << i for i, b in enumerate(bits))

    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_bitstring, top_count = sorted_counts[0]
    top_idx = bitstring_to_idx(top_bitstring)
    shots_total = sum(counts.values())

    print(f"Most frequent measurement: {top_bitstring} (idx {top_idx}), "
          f"{top_count}/{shots_total} shots")

    # Independent re-verification of the measured result against the
    # classical predicate (not just membership in the precomputed set).
    bits = [(top_idx >> i) & 1 for i in range(N_EDGES)]
    edges = edges_from_bits(bits)
    measured_edge_count = len(edges)
    measured_c4_free = not has_4_cycle(edges)
    measured_is_optimal = (measured_edge_count == m_star) and measured_c4_free

    # Also require that the marked-set probability mass is concentrated
    # (Grover amplification actually worked), as a sanity check beyond a
    # single lucky shot.
    marked_set = set(marked)
    marked_shots = sum(c for bs, c in counts.items() if bitstring_to_idx(bs) in marked_set)
    marked_fraction = marked_shots / shots_total

    print(f"Measured graph edges = {measured_edge_count}, C4-free = {measured_c4_free}")
    print(f"Fraction of shots landing on a marked (optimal, C4-free) state: {marked_fraction:.3f}")

    verified = measured_is_optimal and top_idx in marked_set and marked_fraction > 0.5

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
