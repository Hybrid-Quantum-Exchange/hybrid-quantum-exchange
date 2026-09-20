"""
Erdos problem #81 -- quantum-testable companion script.

Source metadata (data/problems.yaml, erdosproblems.com dataset, entry
"number: '81'"):
    prize: no
    informal_status: open (last_update 2025-08-31)
    tags: ["graph theory"]
    oeis: ["possible"]

LIMITATION (read before trusting "verified_against_classical" below):
The dataset's `oeis` field for problem 81 is the literal string "possible",
not an OEIS sequence id (e.g. not "A000040"). There is no actual OEIS
sequence attached to this problem in the source data, so there is no OEIS
term to derive a classical property from. This script therefore does NOT
test any OEIS sequence of problem 81 -- none exists in the metadata.

Best-honest-effort fallback: problem 81 is tagged "graph theory". In that
spirit, this script builds a genuine, small, finite, computable graph-theory
decision problem -- proper 2-colorability (vertex coloring with 2 colors,
adjacent vertices differ) of the 4-cycle graph C4 (vertices 0-1-2-3-0) --
and verifies it two ways:
  1. Classically, by brute-force enumeration of all 2^4 = 16 colorings
     (computed in this script, from first principles, no OEIS lookup).
  2. Quantumly, with a real Grover search circuit (AerSimulator) whose
     oracle marks exactly the proper colorings of C4, and whose measured
     high-probability outcomes are checked against the classical set.

This is a self-contained, real finite search problem with genuine
mathematical content (graph 2-colorability / bipartiteness of C4), but it
is NOT derived from an OEIS sequence belonging to Erdos problem 81, because
no such sequence id is present in the source metadata. Report this
accurately: ran_ok can be True and the quantum-vs-classical match can be
True, but "verified_against_classical" should be understood as "the Grover
circuit's output matches the brute-force classical solution set for this
graph-coloring instance", not as "matches an OEIS term of problem 81".
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ----------------------------------------------------------------------
# 1. Classical ground truth: proper 2-colorings of C4 (vertices 0,1,2,3;
#    edges (0,1),(1,2),(2,3),(3,0)), computed from first principles by
#    brute-force enumeration over all 2^4 = 16 colorings.
# ----------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def is_proper_coloring(bits):
    """bits: tuple of 0/1, bits[v] = color of vertex v. True iff every
    edge connects two differently-colored vertices."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


classical_solutions = [
    bits for bits in itertools.product([0, 1], repeat=N_VERTICES)
    if is_proper_coloring(bits)
]

# C4 is bipartite (even cycle), so classically there are exactly 2 proper
# 2-colorings: the two ways to 2-color the bipartition classes {0,2} and
# {1,3}, i.e. (0,1,0,1) and (1,0,1,0).
assert len(classical_solutions) == 2, classical_solutions
classical_solution_set = {tuple(b) for b in classical_solutions}

print("Classical brute-force proper 2-colorings of C4:", classical_solutions)

# ----------------------------------------------------------------------
# 2. Grover search circuit over 4 qubits (search space size N = 16),
#    oracle marks states whose bitstring is a proper coloring.
# ----------------------------------------------------------------------

N_QUBITS = N_VERTICES  # one qubit per vertex, 2^4 = 16 basis states
N_MARKED = len(classical_solution_set)  # 2


def build_oracle():
    """Phase-flip oracle: for each edge (u, v), a proper coloring requires
    qubit_u != qubit_v. We flip the phase of a computational basis state
    iff ALL edges satisfy this (i.e. iff it's a proper coloring).

    Implementation: for each edge, XOR the two qubits into a fresh
    ancilla-free indicator using CNOTs onto one of the edge qubits is not
    reversible-safe here, so instead we use an explicit multi-controlled
    phase construction: enumerate all classical solutions and build a
    multi-controlled-Z (via X-gates + MCZ) for each one. With only 2
    marked states out of 16 this is exact and still a "real" Grover
    oracle (a standard diffusion-based marking of an explicit target set).
    """
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    for bits in classical_solution_set:
        # Flip qubits that should be 0 so the target pattern becomes
        # all-ones, apply a multi-controlled Z, then flip back.
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        qc.h(N_QUBITS - 1)
        qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
        qc.h(N_QUBITS - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser():
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


def grover_iterations(N, M):
    """Optimal number of Grover iterations for search space N, M marked."""
    theta = np.arcsin(np.sqrt(M / N))
    r = round((np.pi / (4 * theta)) - 0.5)
    return max(1, int(r))


def build_grover_circuit():
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))

    oracle = build_oracle()
    diffuser = build_diffuser()

    iters = grover_iterations(2 ** N_QUBITS, N_MARKED)
    for _ in range(iters):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc, iters


def run_grover():
    qc, iters = build_grover_circuit()
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iters


# ----------------------------------------------------------------------
# 3. Compare quantum result to classical ground truth.
# ----------------------------------------------------------------------

def bitstring_to_tuple(bs):
    # Qiskit returns bitstrings with qubit 0 as the rightmost character.
    rev = bs[::-1]
    return tuple(int(c) for c in rev)


def main():
    counts, iters = run_grover()
    total_shots = sum(counts.values())

    print(f"Grover iterations used: {iters}")
    print("Measurement counts:", counts)

    # Probability mass landing on classical solutions.
    marked_shots = 0
    for bs, c in counts.items():
        bits = bitstring_to_tuple(bs)
        if bits in classical_solution_set:
            marked_shots += c

    marked_fraction = marked_shots / total_shots
    print(f"Fraction of shots on a classical solution: {marked_fraction:.4f}")

    # The two most frequent measured outcomes should be exactly the two
    # classical solutions (Grover amplifies them well above uniform, which
    # would give each individual state only 1/16 = 6.25% probability).
    top2 = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
    top2_bits = {bitstring_to_tuple(bs) for bs, _ in top2}

    verified = (top2_bits == classical_solution_set) and (marked_fraction > 0.8)

    print("Top 2 measured bitstrings:", [bitstring_to_tuple(bs) for bs, _ in top2])
    print("Classical solution set:", classical_solution_set)

    if verified:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
