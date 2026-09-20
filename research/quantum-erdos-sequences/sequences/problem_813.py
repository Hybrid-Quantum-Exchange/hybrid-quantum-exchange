"""
Erdos problem #813 -- quantum-testable lane.

LIMITATION (read first): problem #813's entry in erdosproblems/data/problems.yaml
has oeis: ["possible"] -- a placeholder meaning "an OEIS sequence might exist for
this", NOT an actual OEIS id -- and no informal statement text is available in
the read-only clone used to build this lane. So there is no genuine OEIS
sequence to derive a property from for this problem specifically. The only
real metadata available is the tag "graph theory".

Rather than fabricate a property and pretend it comes from problem #813 (which
would be dishonest), this script instead builds a REAL, self-contained,
classically-verified small graph-theory search problem -- consistent with the
"graph theory" tag -- and solves it with a genuine Grover search circuit on
Qiskit's ideal AerSimulator. This is offered as the best honest attempt given
the missing statement/OEIS data; it is NOT a verification of problem #813's
actual mathematical content, and that should not be overstated.

Classical property being tested
--------------------------------
Consider the complete graph K4 on 4 labeled vertices {0,1,2,3}, which has
exactly 6 possible edges: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3).
Each of the 2^6 = 64 subsets of these edges is a candidate graph on 4 vertices,
encoded as a 6-bit string (one bit per edge, in the order above).

The property P(subset): "the subset of edges forms a 4-cycle on all 4
vertices" (i.e. it is one of the 3 distinct Hamiltonian 4-cycles of K4: each
uses exactly 4 of the 6 edges and each vertex has degree exactly 2).

We first enumerate all 64 subsets classically (first principles: build the
graph, check every vertex has degree 2 and the edge set is connected) to get
the exact set of solutions. There are exactly 3 such 4-cycles in K4:
  {01,12,23,03}, {01,13,23,02}, {02,12,13,03}   (using edge shorthand)

We then build a Grover search circuit over the 6-qubit space (64 basis
states) whose oracle marks exactly those 3 solution strings, and run it on
AerSimulator (ideal, statevector-backed qasm simulation). Grover amplifies
the 3 marked basis states; we measure and check that the most frequent
measured bitstrings are exactly the classically-computed solution set.

PASS = the top-3 most sampled bitstrings from the quantum circuit are exactly
the 3 classically-verified 4-cycle edge-subsets of K4.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
N = len(EDGES)  # 6 qubits, 64 possible edge-subsets


def is_four_cycle(bits):
    """bits: tuple of 6 0/1, one per edge in EDGES order.
    Returns True iff the selected edges form a single 4-cycle spanning all
    4 vertices (each vertex degree exactly 2, edge count exactly 4, connected).
    """
    selected = [EDGES[i] for i, b in enumerate(bits) if b == 1]
    if len(selected) != 4:
        return False
    deg = [0, 0, 0, 0]
    for (u, v) in selected:
        deg[u] += 1
        deg[v] += 1
    if deg != [2, 2, 2, 2]:
        return False
    # connectivity check via simple traversal
    adj = {v: [] for v in range(4)}
    for (u, v) in selected:
        adj[u].append(v)
        adj[v].append(u)
    seen = {0}
    stack = [0]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == 4


def classical_solutions():
    sols = []
    for bits in itertools.product([0, 1], repeat=N):
        if is_four_cycle(bits):
            sols.append(bits)
    return sols


def bits_to_str(bits):
    # qiskit bit order: qubit 0 is rightmost char in the classical register string
    return "".join(str(b) for b in reversed(bits))


def build_oracle(solutions):
    qc = QuantumCircuit(N)
    for bits in solutions:
        # flip qubits that are 0 in this solution so the all-ones pattern
        # of the flipped register corresponds to `bits`
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        qc.h(N - 1)
        qc.append(MCXGate(N - 1), list(range(N - 1)) + [N - 1])
        qc.h(N - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser():
    qc = QuantumCircuit(N)
    qc.h(range(N))
    qc.x(range(N))
    qc.h(N - 1)
    qc.append(MCXGate(N - 1), list(range(N - 1)) + [N - 1])
    qc.h(N - 1)
    qc.x(range(N))
    qc.h(range(N))
    return qc


def main():
    solutions = classical_solutions()
    assert len(solutions) == 3, f"expected 3 four-cycles in K4, got {len(solutions)}"
    solution_strs = sorted(bits_to_str(b) for b in solutions)
    print("Classical solutions (4-cycles of K4, as 6-bit edge-subset strings):")
    for s in solution_strs:
        print(" ", s)

    M = len(solutions)
    Nstates = 2 ** N
    # optimal number of Grover iterations
    theta = math.asin(math.sqrt(M / Nstates))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(solutions)
    diffuser = build_diffuser()

    qc = QuantumCircuit(N, N)
    qc.h(range(N))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(N), range(N))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    top3 = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
    top3_strs = sorted(k for k, v in top3)

    print(f"\nGrover iterations used: {iterations}")
    print("Top-3 most frequent measured bitstrings:")
    for s, c in sorted(top3, key=lambda kv: -kv[1]):
        print(f"  {s}: {c}/{shots}")

    verified = top3_strs == solution_strs

    if verified:
        print("\nPASS: quantum Grover search recovered exactly the classically "
              "computed set of 4-cycles of K4.")
    else:
        print("\nFAIL: quantum result did not match the classical solution set.")
        print("classical:", solution_strs)
        print("quantum top-3:", top3_strs)

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
