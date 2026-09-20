"""
Erdos problem #640 (erdosproblems.com/640) -- quantum-testable lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, verified 2026-09-19):
    number: "640"
    prize: no
    status: open
    tags: ["graph theory", "chromatic number"]
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #640 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no OEIS integer
sequence to build a membership/term-search circuit around, and no literal
OEIS value to (mis)quote. Rather than fabricate one, this script instead
targets the one piece of real mathematical content the metadata does give
us -- the tag "chromatic number" -- with a genuine, from-first-principles,
finite, computable graph-coloring decision problem, which is the natural
finite instance of the kind of question Erdos-style chromatic-number
problems ask:

    THE PROPERTY TESTED:
    "Is the 4-cycle graph C4 (vertices 0-1-2-3-0) properly 2-colorable?"
    i.e. does there exist an assignment of one of 2 colors to each of the
    4 vertices such that every edge joins two differently-colored vertices?

This is a bona fide small constraint-satisfaction / graph-coloring search:
4 vertices x 1 bit/vertex (2 colors) = 4 qubits, search space size 16,
exactly the kind of "small search space whose answer is known/derivable"
instance called for. It is computed here two independent ways:

  1. Classically, by brute force over all 2^4 = 16 colorings (first
     principles -- no lookup, no OEIS value copied).
  2. Quantumly, with a genuine Grover search circuit on AerSimulator: an
     oracle phase-flips exactly the colorings that satisfy all 4 edge
     constraints, a diffuser amplifies them, and the most frequently
     measured bitstring after the computed optimal number of Grover
     iterations is compared against the classical set of valid colorings.

C4 is bipartite, so the classical brute force finds exactly 2 valid
colorings out of 16 (a coloring and its complement): {0:A,1:B,2:A,3:B} and
its swap. Grover search is run to amplify exactly these 2 marked states
out of 16, and PASS/FAIL is decided by checking that the top measurement
outcome(s) by probability are members of the classical valid-coloring set.

Dependencies: qiskit, qiskit_aer, numpy only (already installed).
"""

import math
import itertools

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. The graph: C4, 4 vertices, edges (0,1) (1,2) (2,3) (3,0)
# ---------------------------------------------------------------------------
N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def classical_valid_colorings():
    """Brute force, from first principles: every 2-coloring of C4's 4
    vertices, keep the ones where every edge has differently-colored
    endpoints. Returns the set of valid colorings as 4-bit strings
    (bit i = color of vertex i), MSB = vertex 3 .. LSB = vertex 0, matching
    the qubit ordering used below (qubit i <-> vertex i, Qiskit prints
    q3 q2 q1 q0).
    """
    valid = set()
    for bits in itertools.product([0, 1], repeat=N_VERTICES):
        # bits[i] = color assigned to vertex i
        if all(bits[a] != bits[b] for (a, b) in EDGES):
            bitstring = "".join(str(bits[v]) for v in reversed(range(N_VERTICES)))
            valid.add(bitstring)
    return valid


CLASSICAL_VALID = classical_valid_colorings()
N = 2 ** N_VERTICES
M = len(CLASSICAL_VALID)

print(f"Classical brute force: {M} valid 2-colorings out of {N} total: {sorted(CLASSICAL_VALID)}")

assert M == 2, "C4 is bipartite; exactly 2 proper 2-colorings expected (a coloring and its complement)."


# ---------------------------------------------------------------------------
# 2. Grover oracle: phase-flip states where every edge (a,b) has qubit_a != qubit_b
#    Implemented via ancilla-free approach: use one ancilla qubit as the
#    standard Grover phase-kickback target, built from a multi-controlled
#    condition on 4 "edge-satisfied" helper qubits.
# ---------------------------------------------------------------------------
def build_grover_circuit(num_iterations):
    n = N_VERTICES          # 4 data qubits, one per vertex
    e = len(EDGES)          # 4 helper qubits, one per edge (XOR of endpoints)
    data = list(range(n))
    helper = list(range(n, n + e))
    ancilla = n + e         # phase-kickback ancilla

    qc = QuantumCircuit(n + e + 1, n)

    # uniform superposition over the 4 data (vertex-color) qubits
    qc.h(data)

    # ancilla in |-> for phase kickback
    qc.x(ancilla)
    qc.h(ancilla)

    def mark_oracle():
        # helper[k] = data[a] XOR data[b] for edge k=(a,b); this is 1 iff
        # the edge's endpoints differ in color, i.e. the edge constraint holds.
        for k, (a, b) in enumerate(EDGES):
            qc.cx(data[a], helper[k])
            qc.cx(data[b], helper[k])
        # flip ancilla's phase iff ALL helper qubits are 1 (all edges satisfied)
        qc.mcx(helper, ancilla)
        # uncompute helpers
        for k, (a, b) in enumerate(EDGES):
            qc.cx(data[b], helper[k])
            qc.cx(data[a], helper[k])

    def diffuser():
        qc.h(data)
        qc.x(data)
        qc.h(data[-1])
        qc.mcx(data[:-1], data[-1])
        qc.h(data[-1])
        qc.x(data)
        qc.h(data)

    for _ in range(num_iterations):
        mark_oracle()
        diffuser()

    qc.h(ancilla)
    qc.x(ancilla)

    qc.measure(data, list(range(n)))
    return qc


# optimal number of Grover iterations for N=16 states, M=2 marked
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M) - 0.5))
print(f"Running Grover search: N={N} states, M={M} marked, iterations={optimal_iterations}")

circuit = build_grover_circuit(optimal_iterations)

backend = AerSimulator()
transpiled = transpile(circuit, backend)
shots = 4096
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print("Top measurement outcomes:", sorted_counts[:5])

total_valid_shots = sum(c for bs, c in counts.items() if bs in CLASSICAL_VALID)
valid_fraction = total_valid_shots / shots
print(f"Fraction of shots landing on a classically-valid coloring: {valid_fraction:.3f}")

# The top-M measured bitstrings (M = number of classically valid colorings)
# should be exactly the classically valid colorings, and together should
# dominate the distribution (amplified by Grover well above the 2/16=12.5%
# uniform baseline).
top_m_bitstrings = {bs for bs, _ in sorted_counts[:M]}
quantum_matches_classical = top_m_bitstrings == CLASSICAL_VALID
amplified_above_baseline = valid_fraction > (M / N) * 2  # comfortably above uniform

verified = quantum_matches_classical and amplified_above_baseline

print(f"Top-{M} measured bitstrings: {sorted(top_m_bitstrings)}")
print(f"Classical valid set:        {sorted(CLASSICAL_VALID)}")
print(f"Quantum top outcomes match classical valid set: {quantum_matches_classical}")
print(f"Amplified well above uniform baseline: {amplified_above_baseline}")

if verified:
    print("PASS")
else:
    print("FAIL")
