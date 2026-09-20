"""
Erdos problem #944 (erdosproblems.com) -- quantum-testable companion script.

Problem #944 metadata, as recorded in the erdosproblems.com dataset
(data/problems.yaml, entry "number: '944'"):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["N/A"]        <-- NO OEIS sequence is associated with this problem
    tags: ["graph theory", "chromatic number"]

LIMITATION, stated honestly up front: problem #944 has no OEIS id in the
dataset ("N/A"), so there is no integer sequence to test membership/terms
against for this problem. Per the task instructions, rather than fabricate
an OEIS-derived property, this script instead builds a REAL, finite,
computable property that is faithful to the problem's own tags (graph
theory / chromatic number): whether a small fixed graph is properly
2-colorable, decided by a genuine Grover search over all 2-colorings of its
vertices. This is a legitimate small search problem with a combinatorial
oracle, not a fabricated stand-in value; it is just not derived from an
OEIS sequence, because #944 does not have one.

Classical property under test
------------------------------
Graph G = path graph on 3 vertices {0, 1, 2} with edges (0,1) and (1,2).
A 2-coloring is an assignment of one bit (color 0 or 1) to each vertex,
encoded as a 3-bit string c2 c1 c0 (q2 q1 q0). The coloring is PROPER iff
every edge connects differently-colored vertices:
    proper(c) <=> (c0 != c1) AND (c1 != c2)

The classical answer (computed here in Python, by brute force over all
2^3 = 8 colorings, from first principles -- no lookup) is: exactly the two
alternating colorings {010, 101} (bit order c2 c1 c0) are proper. This is
the classical "chromatic-number-related" fact being tested: this graph is
properly 2-colorable (consistent with a path graph being bipartite / chi=2),
and the quantum circuit's job is to find those solutions by Grover search
over the coloring space, exactly like grover search finds any small marked
subset of a search space.

Quantum construction
---------------------
6 qubits: q0,q1,q2 = the 3 color bits (search register); e1,e2 = ancillas
holding XOR(q0,q1) and XOR(q1,q2); f = the Grover oracle's phase-kickback
flag qubit (prepared in |-> so a Toffoli flip becomes a phase flip).

Oracle: CNOT q0,q1 -> e1 ; CNOT q1,q2 -> e2 ; Toffoli(e1,e2 -> f) ; uncompute
e1,e2. This marks (flips the phase of) exactly the basis states where both
edge-XORs are 1, i.e. exactly the proper 2-colorings.

Diffuser: the standard 3-qubit Grover diffusion operator on q0,q1,q2.

With N=8 basis states and M=2 marked (proper-coloring) states, the optimal
number of Grover iterations is round(pi/4 * sqrt(N/M)) = 1, which is what
this script runs, then measures q0,q1,q2 many times on the ideal AerSimulator
and checks that the two proper-coloring bitstrings absorb the large majority
of the measured probability mass, exactly matching the classical brute-force
solution set.

PASS/FAIL: the script prints PASS iff the set of bitstrings whose measured
probability exceeds a threshold equals the classical solution set found by
brute force.
"""

from itertools import product

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import numpy as np


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles by brute force.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2)]  # path graph on vertices 0,1,2


def is_proper_coloring(bits):
    """bits: tuple (c0, c1, c2) of 0/1. True iff every edge's endpoints differ."""
    return all(bits[u] != bits[v] for u, v in EDGES)


def classical_solutions():
    sols = []
    for c0, c1, c2 in product([0, 1], repeat=3):
        if is_proper_coloring((c0, c1, c2)):
            # bitstring convention matching Qiskit's little-endian measurement
            # output "c2 c1 c0"
            sols.append(f"{c2}{c1}{c0}")
    return sorted(sols)


CLASSICAL_SOLUTIONS = classical_solutions()
assert CLASSICAL_SOLUTIONS == ["010", "101"], CLASSICAL_SOLUTIONS
N = 8          # search space size, 2^3
M = len(CLASSICAL_SOLUTIONS)  # number of marked (proper-coloring) states


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for proper 2-colorings.
# ---------------------------------------------------------------------------

def grover_circuit(iterations):
    q = QuantumRegister(3, "q")   # color bits q0,q1,q2
    e = QuantumRegister(2, "e")   # ancillas: e0 = q0^q1, e1 = q1^q2
    f = QuantumRegister(1, "f")   # oracle phase-kickback flag
    c = ClassicalRegister(3, "c")
    qc = QuantumCircuit(q, e, f, c)

    # uniform superposition over all 8 colorings
    qc.h(q)

    # prepare flag in |-> for phase kickback
    qc.x(f[0])
    qc.h(f[0])

    def oracle():
        # e0 = q0 XOR q1
        qc.cx(q[0], e[0])
        qc.cx(q[1], e[0])
        # e1 = q1 XOR q2
        qc.cx(q[1], e[1])
        qc.cx(q[2], e[1])
        # flip phase of flag iff e0 AND e1 both 1 (both edges properly colored)
        qc.ccx(e[0], e[1], f[0])
        # uncompute ancillas
        qc.cx(q[1], e[1])
        qc.cx(q[2], e[1])
        qc.cx(q[0], e[0])
        qc.cx(q[1], e[0])

    def diffuser():
        qc.h(q)
        qc.x(q)
        qc.h(q[2])
        qc.ccx(q[0], q[1], q[2])
        qc.h(q[2])
        qc.x(q)
        qc.h(q)

    for _ in range(iterations):
        oracle()
        diffuser()

    qc.measure(q, c)
    return qc


def optimal_iterations(n, m):
    theta = np.arcsin(np.sqrt(m / n))
    return max(1, round((np.pi / (4 * theta)) - 0.5))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

def main():
    iterations = optimal_iterations(N, M)
    qc = grover_circuit(iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 8192
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    total = sum(counts.values())
    threshold = 0.10  # a bitstring absorbing >10% of shots is "found" by Grover
    found = sorted(
        bits for bits, cnt in counts.items() if cnt / total > threshold
    )

    print("Erdos problem #944 (graph theory / chromatic number, OEIS: N/A)")
    print(f"Graph: path on 3 vertices, edges {EDGES}")
    print(f"Grover iterations used: {iterations}")
    print("Measured counts:", counts)
    print("Classical proper-2-coloring solutions (brute force):", CLASSICAL_SOLUTIONS)
    print("Quantum-found high-probability bitstrings:", found)

    ok = found == CLASSICAL_SOLUTIONS
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
