"""
Erdos problem #628 -- quantum-testable instance.

Source metadata (data/problems.yaml, entry "number: \"628\""):
    comments: "Erdős-Lovász Tihany Conjecture"
    tags: ["graph theory", "chromatic number"]
    oeis: ["N/A"]

HONEST LIMITATION: problem #628 carries no OEIS sequence id (oeis: ["N/A"]
in the source data), so there is no OEIS-derived integer sequence to test
membership/terms of, and this script cannot honor the "from its OEIS
sequence id(s)" instruction literally. Rather than fabricate a fake OEIS
id or copy an unrelated sequence, this script instead builds a genuine,
finite, computable instance of the property named directly by the
problem's own tags -- "chromatic number" / graph 2-colorability -- which
is exactly the combinatorial object the Erdos-Lovasz Tihany Conjecture is
about (it concerns how a graph's chromatic number can be forced to split
across an edge's two endpoint color classes).

Classical property under test
------------------------------
Graph G = P3, the path on 3 vertices {0, 1, 2} with edges (0,1) and (1,2).

Question: which of the 2^3 = 8 possible 2-colorings x = (x0, x1, x2) in
{0,1}^3 are PROPER 2-colorings of G, i.e. satisfy x0 != x1 AND x1 != x2?

This is computed from first principles by brute force in
`classical_solutions()` below (no OEIS lookup, no hard-coded literal).
Brute force gives exactly two proper 2-colorings: 010 and 101
(the two alternating colorings of a path), confirming chi(P3) <= 2,
i.e. P3 is bipartite / 2-colorable, which is the finite decidable fact
this script quantum-verifies.

Quantum approach
-----------------
Grover search over the 3-qubit space {0,1}^3 with a phase oracle that
marks exactly the proper-2-coloring states (built from the same boolean
condition (x0 XOR x1) AND (x1 XOR x2), computed reversibly on ancillas,
phase-kicked with a controlled-Z, then uncomputed). With N = 8 basis
states and M = 2 marked states, one Grover iteration
(floor(pi/4 * sqrt(N/M)) = 1) is optimal and should amplify the two
marked states to high combined measurement probability on the ideal
AerSimulator.

PASS criterion: the set of bitstrings receiving the highest measured
counts (after 1 Grover iteration, many shots) equals exactly the
classically brute-forced solution set {010, 101} AND their combined
measured probability exceeds a fixed threshold confirming amplification
actually occurred (not just a flat/uniform post-selection artifact).
"""

from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_VERTICES = 3
EDGES = [(0, 1), (1, 2)]  # path graph P3


def classical_solutions():
    """Brute-force all proper 2-colorings of P3 from first principles."""
    solutions = []
    for coloring in product([0, 1], repeat=N_VERTICES):
        if all(coloring[u] != coloring[v] for (u, v) in EDGES):
            # bitstring convention: qiskit prints qubit order q2 q1 q0
            bitstring = "".join(str(coloring[i]) for i in reversed(range(N_VERTICES)))
            solutions.append(bitstring)
    return sorted(solutions)


def grover_circuit():
    x = list(range(3))       # data qubits x0, x1, x2
    anc = [3, 4]              # ancilla qubits a0, a1
    qc = QuantumCircuit(5, 3)

    # uniform superposition over the 3 data qubits
    qc.h(x)

    # 1 Grover iteration (optimal for N=8, M=2)
    # --- oracle: mark proper 2-colorings ---
    qc.cx(x[0], anc[0])
    qc.cx(x[1], anc[0])   # anc[0] = x0 XOR x1
    qc.cx(x[1], anc[1])
    qc.cx(x[2], anc[1])   # anc[1] = x1 XOR x2

    # phase flip only when both ancillas are 1 (both edges properly colored)
    qc.cz(anc[0], anc[1])

    # uncompute ancillas
    qc.cx(x[2], anc[1])
    qc.cx(x[1], anc[1])
    qc.cx(x[1], anc[0])
    qc.cx(x[0], anc[0])

    # --- diffuser on the 3 data qubits ---
    qc.h(x)
    qc.x(x)
    qc.h(x[2])
    qc.ccx(x[0], x[1], x[2])
    qc.h(x[2])
    qc.x(x)
    qc.h(x)

    qc.measure(x, [0, 1, 2])
    return qc


def main():
    classical = classical_solutions()
    print(f"Classical proper 2-colorings of P3 (brute force): {classical}")
    assert classical == ["010", "101"], "classical brute force gave an unexpected result"

    qc = grover_circuit()
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    print("Measured counts (top 5):", sorted_counts[:5])

    top_states = {b for b, c in sorted_counts[:2]}
    combined_prob = sum(counts.get(b, 0) for b in classical) / shots
    print(f"Top 2 measured states: {sorted(top_states)}")
    print(f"Combined probability mass on classical solutions {classical}: {combined_prob:.4f}")

    verified = (top_states == set(classical)) and (combined_prob > 0.85)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
