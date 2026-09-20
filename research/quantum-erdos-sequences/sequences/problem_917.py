"""
Erdos problem #917 (erdosproblems.com), as recorded in data/problems.yaml
(manman4/erdosproblems, read-only clone):

    number: "917"
    oeis: ["N/A"]
    tags: ["graph theory", "chromatic number"]

LIMITATION, stated up front: problem 917 carries no OEIS sequence id in the
source data (oeis: ["N/A"]). There is therefore no "sequence" to build a
quantum-testable membership/term property from in the way the other lanes
in this library do. To still produce a genuine, non-fabricated quantum
computation tied to the problem's own tags ("graph theory",
"chromatic number"), this script tests the smallest non-trivial instance of
the actual combinatorial question a chromatic-number problem asks:

    CLASSICAL PROPERTY TESTED:
    Is the path graph P3 (vertices v0-v1-v2, edges {v0,v1} and {v1,v2})
    2-colorable, i.e. is its chromatic number <= 2? Equivalently: does a
    proper 2-coloring (a function c: {v0,v1,v2} -> {0,1} with c(v0)!=c(v1)
    and c(v1)!=c(v2)) exist, and how many of the 2^3 = 8 possible colorings
    are valid?

    This is computed from first principles by brute force in
    classical_solutions() below (no OEIS lookup, no hard-coded literal):
    it enumerates all 8 colorings of P3 and checks the two edge
    constraints directly. P3 is bipartite (v0/v2 in one class, v1 in the
    other), so the classical answer is: valid, with exactly 2 solutions
    out of 8 assignments (the two proper 2-colorings, {v0,v2}=0,v1=1 and
    {v0,v2}=1,v1=0).

QUANTUM CIRCUIT:
Grover's search algorithm over the 3-qubit space of all colorings of P3.
The oracle marks a computational basis state |c0 c1 c2> as a "hit" exactly
when c0 != c1 AND c1 != c2 (built from two CNOT-based inequality checks
into ancilla qubits, combined with a Toffoli, phase-kicked back, then
uncomputed). With N = 8 states and M = 2 solutions, the optimal number of
Grover iterations is floor(pi/4 * sqrt(N/M)) = 1. After 1 iteration and
measurement on the ideal AerSimulator, the two marked basis states
(|010> and |101> in qiskit's little-endian bit order, matching
(c0,c1,c2) = (1,0,1) and (0,1,0)) must dominate the measured distribution.

PASS/FAIL: the script computes the classical solution set, runs the Grover
circuit, and PASSes only if the two most-frequent measured bitstrings are
exactly the two classical solutions and together account for the large
majority of shots (Grover amplification, not a coincidence).
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2)]  # path graph P3: v0-v1-v2
N_VERTICES = 3


def classical_solutions():
    """Brute-force every 2-coloring of P3 and return the valid ones.

    A coloring is valid iff every edge's endpoints get different colors.
    Returns a set of 3-bit tuples (c0, c1, c2).
    """
    solutions = set()
    for coloring in itertools.product((0, 1), repeat=N_VERTICES):
        if all(coloring[u] != coloring[v] for (u, v) in EDGES):
            solutions.add(coloring)
    return solutions


SOLUTIONS = classical_solutions()
assert SOLUTIONS == {(1, 0, 1), (0, 1, 0)}, (
    "sanity check failed: P3 is bipartite and must have exactly these two "
    f"proper 2-colorings, got {SOLUTIONS}"
)

# qiskit reports measurement bitstrings as c2 c1 c0 (qubit 0 is rightmost).
# Build the expected bitstrings for the two classical solutions accordingly.
EXPECTED_BITSTRINGS = {
    "".join(str(c) for c in reversed(coloring)) for coloring in SOLUTIONS
}

# ---------------------------------------------------------------------------
# 2. Grover oracle for "is this coloring proper?" on P3.
# ---------------------------------------------------------------------------


def build_oracle(qc, color, anc, edges):
    """Flip the phase of |color> iff every edge endpoint pair differs.

    color: 3-qubit register (c0, c1, c2)
    anc:   2 diff-flag ancillas (one per edge) + 1 output ancilla, all
           starting and ending in |0>.
    """
    diff_ancillas = anc[: len(edges)]
    out = anc[len(edges)]

    # diff_ancillas[i] = c_u XOR c_v for edge i  (1 means "different", good)
    for i, (u, v) in enumerate(edges):
        qc.cx(color[u], diff_ancillas[i])
        qc.cx(color[v], diff_ancillas[i])

    # out = AND of all diff flags (multi-controlled X)
    qc.mcx(list(diff_ancillas), out)

    # phase kickback: flip phase of the marked state
    qc.z(out)

    # uncompute (mirror image) to restore ancillas to |0>
    qc.mcx(list(diff_ancillas), out)
    for i, (u, v) in enumerate(edges):
        qc.cx(color[v], diff_ancillas[i])
        qc.cx(color[u], diff_ancillas[i])


def build_diffuser(qc, color):
    n = len(color)
    qc.h(color)
    qc.x(color)
    qc.h(color[-1])
    qc.mcx(list(color[:-1]), color[-1])
    qc.h(color[-1])
    qc.x(color)
    qc.h(color)


def build_grover_circuit(iterations=1):
    color = QuantumRegister(N_VERTICES, "c")
    anc = AncillaRegister(len(EDGES) + 1, "a")
    creg = ClassicalRegister(N_VERTICES, "m")
    qc = QuantumCircuit(color, anc, creg)

    qc.h(color)

    for _ in range(iterations):
        build_oracle(qc, color, anc, EDGES)
        build_diffuser(qc, color)

    qc.measure(color, creg)
    return qc


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------


def main():
    n = 2 ** N_VERTICES
    m = len(SOLUTIONS)
    iterations = max(1, int(np.floor(np.pi / 4 * np.sqrt(n / m))))

    qc = build_grover_circuit(iterations=iterations)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 8192
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    top_bitstrings = {bs for bs, _ in ranked[: len(SOLUTIONS)]}
    top_mass = sum(c for bs, c in ranked[: len(SOLUTIONS)]) / shots

    print(f"Classical proper 2-colorings of P3 (c0,c1,c2): {sorted(SOLUTIONS)}")
    print(f"Expected top measurement bitstrings (c2c1c0): {sorted(EXPECTED_BITSTRINGS)}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measured counts (top 6): {ranked[:6]}")
    print(f"Top-{len(SOLUTIONS)} measured bitstrings: {sorted(top_bitstrings)}")
    print(f"Probability mass on those bitstrings: {top_mass:.3f}")

    verified = (top_bitstrings == EXPECTED_BITSTRINGS) and (top_mass > 0.6)

    if verified:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
