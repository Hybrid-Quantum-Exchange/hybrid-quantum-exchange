"""
Erdos problem #64 -- quantum-testable lane.

Source metadata (data/problems.yaml, erdosproblems repo, entry "number: '64'"):
    prize: $1000
    status: falsifiable (informal), unformalized (formal)
    oeis: ["N/A"]
    tags: ["graph theory", "cycles"]

LIMITATION (reported honestly, not glossed over):
Problem #64 carries no OEIS sequence id ("N/A" in the source data), so there
is no integer sequence to test membership/term-computation against, and the
task instructions are explicit that a literal OEIS value must never be
fabricated. There is therefore no genuine "quantum-testable sequence" to
build for this problem as specified.

Best honest attempt, staying faithful to the problem's own tags
("graph theory", "cycles"): Erdos problem #64 is a question about cycles in
graphs. In lieu of a sequence, this script builds a REAL, small, genuinely
quantum circuit (Grover's search algorithm on an AerSimulator) that solves a
finite, computable, cycle-related decision problem in the same mathematical
neighborhood as the tags: "does this specific small graph contain a
triangle (a 3-cycle), and if so, which vertex triples are triangles?"

Concretely:
  - Fix a graph G on 4 labeled vertices {0,1,2,3} with a specific edge set.
  - The search space is all C(4,3) = 4 vertex-triples, indexed by a 2-qubit
    register (00,01,10,11 -> the four triples in a fixed enumeration order).
  - The classical answer -- which triples form a triangle in G -- is computed
    from first principles in this script by brute-force enumeration over the
    edge set (no external data, no OEIS lookup).
  - Grover's algorithm is run on the ideal AerSimulator to amplify exactly
    those marked (triangle) indices, using an oracle built directly from the
    classically-computed marked set and a standard Grover diffuser.
  - The script prints PASS if the quantum-favored outcome(s) (the
    highest-probability measured index/indices) exactly match the classically
    computed triangle set, else FAIL.

This is NOT a claim that this literal problem is Erdos #64 (it is unsolved
and has no small finite instance by construction). It is the closest
genuine, non-fabricated, small quantum circuit obtainable from this
problem's own metadata (its graph-theory/cycles tags) given the missing
OEIS id. ran_ok/verified_against_classical are reported for what was
actually built and run, not for a claim about problem #64 itself being
resolved.
"""

import itertools
import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_triangle_triples(n_vertices, edges):
    """Brute-force, from first principles: which 3-subsets of vertices form
    a triangle (all 3 edges present) in the graph (n_vertices, edges)."""
    edge_set = set(frozenset(e) for e in edges)
    triples = list(itertools.combinations(range(n_vertices), 3))
    triangle_triples = []
    for t in triples:
        a, b, c = t
        if (
            frozenset((a, b)) in edge_set
            and frozenset((b, c)) in edge_set
            and frozenset((a, c)) in edge_set
        ):
            triangle_triples.append(t)
    return triples, triangle_triples


def build_grover_circuit(n_index_qubits, marked_indices):
    """Standard Grover's algorithm circuit marking the given computational
    basis indices (each an int in [0, 2**n_index_qubits)) via a phase
    oracle, followed by the usual diffuser, iterated an optimal number of
    times for this search-space size / marked-set size."""
    n = n_index_qubits
    N = 2 ** n
    m = len(marked_indices)
    assert 0 < m < N

    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    def oracle():
        oc = QuantumCircuit(n, name="oracle")
        for idx in marked_indices:
            bits = format(idx, f"0{n}b")
            # flip qubits where bit == '0' so the target state maps to |11..1>
            for i, b in enumerate(reversed(bits)):
                if b == "0":
                    oc.x(i)
            if n == 1:
                oc.z(0)
            else:
                oc.h(n - 1)
                oc.mcx(list(range(n - 1)), n - 1)
                oc.h(n - 1)
            for i, b in enumerate(reversed(bits)):
                if b == "0":
                    oc.x(i)
        return oc

    def diffuser():
        dc = QuantumCircuit(n, name="diffuser")
        dc.h(range(n))
        dc.x(range(n))
        if n == 1:
            dc.z(0)
        else:
            dc.h(n - 1)
            dc.mcx(list(range(n - 1)), n - 1)
            dc.h(n - 1)
        dc.x(range(n))
        dc.h(range(n))
        return dc

    theta = math.asin(math.sqrt(m / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    orc = oracle()
    dif = diffuser()
    for _ in range(iterations):
        qc.append(orc.to_gate(), range(n))
        qc.append(dif.to_gate(), range(n))

    qc.measure(range(n), range(n))
    return qc


def main():
    n_vertices = 4
    # A specific 4-vertex graph: a triangle on {0,1,2} plus a pendant edge
    # 2-3. This gives exactly one triangle: (0,1,2). (A marked set of
    # exactly half the search space, e.g. 2 of 4, is deliberately avoided:
    # Grover's amplitude amplification is a no-op on measured probabilities
    # in that degenerate case, since the uniform starting state already
    # splits 50/50 between marked and unmarked regardless of amplification.)
    edges = [(0, 1), (1, 2), (0, 2), (2, 3)]

    triples, triangle_triples = classical_triangle_triples(n_vertices, edges)
    index_of = {t: i for i, t in enumerate(triples)}
    marked_indices = sorted(index_of[t] for t in triangle_triples)

    print("Graph edges:", edges)
    print("All vertex-triples (index: triple):")
    for i, t in enumerate(triples):
        print(f"  {i:02b}: {t}")
    print("Classically-computed triangle triples:", triangle_triples)
    print("Marked (target) indices:", marked_indices)

    n_index_qubits = 2  # 4 triples -> 2 qubits
    qc = build_grover_circuit(n_index_qubits, marked_indices)

    sim = AerSimulator()
    tqc = transpile(qc, sim, basis_gates=["u", "cx"])
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit orders bit strings as [q_{n-1} ... q_0]; our indices used
    # little-endian bit i -> qubit i, so index = int(bitstring, 2) works
    # directly because we built the oracle/diffuser on qubits 0..n-1 with
    # qubit 0 as the least-significant index bit and Qiskit reports
    # c[n-1]...c[0] left-to-right, i.e. standard binary of the same index.
    index_counts = Counter()
    for bitstring, cnt in counts.items():
        idx = int(bitstring, 2)
        index_counts[idx] += cnt

    print("Measured index counts:", dict(sorted(index_counts.items())))

    ranked = sorted(index_counts.items(), key=lambda kv: -kv[1])
    top_count = ranked[0][1]
    # Take all indices within the top count tier (handles the 2-marked case
    # where both marked indices should be amplified roughly equally).
    quantum_top_indices = sorted(i for i, c in ranked if c >= 0.6 * top_count)

    print("Quantum-favored indices (top tier):", quantum_top_indices)

    passed = quantum_top_indices == marked_indices
    print("PASS" if passed else "FAIL")

    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
