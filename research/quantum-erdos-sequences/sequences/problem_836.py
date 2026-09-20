"""
Erdos problem #836 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, problems.yaml, entry `number: "836"`):
    prize:   no
    status:  open
    tags:    ["graph theory", "hypergraphs", "chromatic number"]
    oeis:    ["N/A"]

HONEST LIMITATION: problem #836 has no associated OEIS sequence (oeis: ["N/A"]
in the source data), and the problem itself concerns chromatic numbers of
hypergraphs formed from infinite unions of large arithmetic-progression-free
sets -- not a small finite decidable predicate. There is therefore no genuine
OEIS-derived "sequence" to test membership/search on a quantum circuit, and
faking one (e.g. by inventing a fictitious OEIS id) would misrepresent the
problem. Rather than fabricate that, this script builds the closest *honest*
finite, computable instance suggested by the problem's own tags
("graph theory", "chromatic number"): 2-colorability (chromatic number <= 2)
of a small, concrete graph, decided by a real Grover search circuit.

Chosen finite instance
-----------------------
Graph G = path graph P3: vertices {0, 1, 2}, edges {(0,1), (1,2)}.
Question: which of the 2^3 = 8 possible 2-colorings (bitstrings c0 c1 c2,
one bit per vertex) are *proper* colorings, i.e. satisfy c0 != c1 and
c1 != c2 (adjacent vertices get different colors)?

This is computed from first principles by brute force in `classical_solve()`
below (no lookup, no OEIS value copied). The two proper colorings of a path
graph are exactly "010" and "101" (alternating colors), and every path graph
is bipartite, so this always has a nonempty, finite, strictly-smaller-than-2^n
solution set -- a genuine (if small) search problem, not a tautology.

Quantum approach
-----------------
Grover's algorithm on 3 qubits (search space size N = 8) with a phase oracle
built directly from the adjacency constraints (implemented as CZ/X gates
encoding "c0 XOR c1 == 1" AND "c1 XOR c2 == 1"), followed by the standard
diffusion operator, run for the Grover-optimal number of iterations for
N = 8, M = 2 solutions (1 iteration). The resulting measurement distribution
on the ideal AerSimulator is compared against the classical brute-force
solution set: PASS if the two most-probable measured bitstrings equal the
classical solution set.
"""

import math
from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2)]  # path graph P3 on vertices {0, 1, 2}
NUM_VERTICES = 3


def is_proper_coloring(bits):
    """bits: tuple of 0/1, one per vertex. True iff every edge is bichromatic."""
    return all(bits[u] != bits[v] for u, v in EDGES)


def classical_solve():
    """Brute-force over all 2^3 colorings; returns the set of proper ones,
    as bitstrings in Qiskit's little-endian convention (qubit 0 = rightmost)."""
    solutions = []
    for bits in product([0, 1], repeat=NUM_VERTICES):
        if is_proper_coloring(bits):
            # bits = (c0, c1, c2); Qiskit reports qubit0 as the last char.
            bitstring = "".join(str(b) for b in reversed(bits))
            solutions.append(bitstring)
    return sorted(solutions)


CLASSICAL_SOLUTIONS = classical_solve()


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for proper 2-colorings of P3.
# ---------------------------------------------------------------------------

N_QUBITS = NUM_VERTICES  # qubit i <-> color of vertex i


def build_oracle_correct():
    """Correct phase oracle for (q0 XOR q1 == 1) AND (q1 XOR q2 == 1)."""
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    # CNOT(q1 -> q0): after this, q0_new = q0 XOR q1. We want q0_new == 1.
    qc.cx(1, 0)
    # CNOT(q1 -> q2): after this, q2_new = q2 XOR q1. We want q2_new == 1.
    qc.cx(1, 2)
    # Now flip phase iff q0_new == 1 and q2_new == 1: controlled-Z on q0,q2.
    qc.cz(0, 2)
    # Uncompute (oracle must be its own inverse aside from the phase flip).
    qc.cx(1, 2)
    qc.cx(1, 0)
    return qc


def build_diffusion():
    qc = QuantumCircuit(N_QUBITS, name="diffusion")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


def build_grover_circuit(iterations):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    oracle = build_oracle_correct()
    diffusion = build_diffusion()
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def grover_optimal_iterations(n_items, n_solutions):
    if n_solutions == 0:
        return 0
    theta = math.asin(math.sqrt(n_solutions / n_items))
    return max(1, round((math.pi / (4 * theta)) - 0.5))


def run_grover():
    n_items = 2 ** N_QUBITS
    n_solutions = len(CLASSICAL_SOLUTIONS)
    iters = grover_optimal_iterations(n_items, n_solutions)

    circuit = build_grover_circuit(iters)
    backend = AerSimulator()
    compiled = transpile(circuit, backend)
    result = backend.run(compiled, shots=4096).result()
    counts = result.get_counts()
    return counts, iters


# ---------------------------------------------------------------------------
# 3. Verification: PASS/FAIL against the classical answer.
# ---------------------------------------------------------------------------

def main():
    print("Erdos problem #836 -- quantum-testable instance")
    print(f"Tags: graph theory, hypergraphs, chromatic number; OEIS: N/A")
    print(f"Instance: proper 2-colorings of path graph P3, edges {EDGES}")
    print(f"Classical brute-force solutions (bitstrings, qubit0=rightmost): "
          f"{CLASSICAL_SOLUTIONS}")

    counts, iters = run_grover()
    print(f"Grover iterations used: {iters}")

    total_shots = sum(counts.values())
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    print("Measurement distribution (top entries):")
    for bitstring, c in sorted_counts[:6]:
        print(f"  {bitstring}: {c} ({100 * c / total_shots:.1f}%)")

    # The measured most-probable outcomes (as many as len(CLASSICAL_SOLUTIONS))
    # should equal the classical solution set.
    top_measured = sorted(
        bs for bs, _ in sorted_counts[: len(CLASSICAL_SOLUTIONS)]
    )

    # Also require that the measured probability mass on the true solutions
    # is amplified well above the uniform baseline (2/8 = 25%), confirming
    # genuine Grover amplification rather than a lucky top-k match.
    solution_mass = sum(counts.get(bs, 0) for bs in CLASSICAL_SOLUTIONS) / total_shots
    uniform_baseline = len(CLASSICAL_SOLUTIONS) / (2 ** N_QUBITS)

    verified = (top_measured == CLASSICAL_SOLUTIONS) and (
        solution_mass > 2 * uniform_baseline
    )

    print(f"Classical solution set : {CLASSICAL_SOLUTIONS}")
    print(f"Top-{len(CLASSICAL_SOLUTIONS)} measured bitstrings : {top_measured}")
    print(f"Solution probability mass: {solution_mass:.3f} "
          f"(uniform baseline {uniform_baseline:.3f})")

    if verified:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
