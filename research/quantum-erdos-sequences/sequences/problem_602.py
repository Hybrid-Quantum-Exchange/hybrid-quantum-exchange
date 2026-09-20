"""
Erdos problem #602 (as recorded in manman4/erdosproblems, data/problems.yaml,
entry `number: "602"`) concerns Property B: the minimum number m(n) of edges
in an n-uniform hypergraph that has no proper 2-coloring (a 2-coloring of the
vertices leaving no edge monochromatic). That entry carries no OEIS id
(`oeis: ["N/A"]`) and tags `["combinatorics", "set theory"]`, comment
"Property B". Because there is no OEIS sequence to anchor a term-membership
test, this script does NOT fabricate one. Instead it tests a small, genuinely
finite, computable fact that sits directly inside the Property B statement
problem 602 is about: for the classical fact m(2) = 3 (the smallest
non-2-colorable 2-uniform hypergraph, i.e. graph, is the triangle K3, and any
2-edge graph on 3 vertices IS properly 2-colorable), a graph with fewer than
3 edges among {0,1,2} always admits a proper 2-coloring.

Concrete instance used here: the path graph on 3 vertices with edges
E = {(0,1), (1,2)} (2 edges, one fewer than the m(2)=3 threshold).

Classical property being tested
--------------------------------
"There exists an assignment of colors c: {0,1,2} -> {0,1} such that for
every edge (u,v) in E, c(u) != c(v)."

This is computed from first principles below by brute-force enumeration of
all 2^3 = 8 colorings (no OEIS lookup, no hard-coded literal): the script
counts how many of the 8 colorings are proper, and lists them explicitly.

Quantum circuit
----------------
A genuine Grover search circuit is built over 3 qubits (one per vertex,
|0> = color 0, |1> = color 1). The oracle marks exactly the computational
basis states that are proper colorings of E = {(0,1),(1,2)}: it flips the
phase of a state iff qubit0 != qubit1 AND qubit1 != qubit2. This is built
with elementary gates (CNOT-based inequality flags, a Toffoli-style
multi-controlled-Z, uncompute), not a hard-coded diagonal unitary. Since the
number of solutions (4 out of 8) is known in advance from the classical
enumeration, the optimal number of Grover iterations is computed via the
standard formula and applied. The circuit is run on the ideal AerSimulator
and the measured, highest-probability outcomes are checked against the
classical solution set.

PASS iff:
  1. classical brute force finds >0 proper colorings for this 2-edge graph
     (confirming instances below the m(2)=3 threshold are 2-colorable), and
  2. the quantum Grover search's most-sampled outcomes are exactly the
     classical proper-coloring set.
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2)]  # 2 edges on 3 vertices: below the m(2)=3 threshold
N_VERTICES = 3


def is_proper_coloring(bits, edges):
    """bits: tuple of 0/1 per vertex. True iff no edge is monochromatic."""
    return all(bits[u] != bits[v] for (u, v) in edges)


def classical_search(edges, n_vertices):
    solutions = []
    for bits in product([0, 1], repeat=n_vertices):
        if is_proper_coloring(bits, edges):
            solutions.append(bits)
    return solutions


classical_solutions = classical_search(EDGES, N_VERTICES)
print(f"Edges: {EDGES}")
print(f"Classical brute-force proper 2-colorings ({len(classical_solutions)} "
      f"of {2 ** N_VERTICES}):")
for s in classical_solutions:
    print(f"  {s}")

assert len(classical_solutions) > 0, (
    "Expected the 2-edge instance (below the m(2)=3 Property B threshold) "
    "to admit a proper 2-coloring."
)

# Basis-state bitstrings (Qiskit little-endian: qubit0 is the rightmost bit).
# Vertex i -> qubit i, so bitstring q2 q1 q0 corresponds to (v0, v1, v2) with
# v_i = bit at position i.
classical_bitstrings = set()
for (v0, v1, v2) in classical_solutions:
    bitstring = f"{v2}{v1}{v0}"  # q2 q1 q0, MSB first as Qiskit prints it
    classical_bitstrings.add(bitstring)


# ---------------------------------------------------------------------------
# 2. Quantum Grover search circuit built from elementary gates.
# ---------------------------------------------------------------------------

def build_oracle(n_vertices, edges):
    """Phase-flip oracle marking proper 2-colorings of `edges` on `n_vertices`
    qubits, using ancillas that hold the per-edge inequality flags."""
    qc = QuantumCircuit(n_vertices + len(edges), name="oracle")
    anc_offset = n_vertices

    # For each edge (u, v): ancilla_e = qubit_u XOR qubit_v (1 iff differ).
    for i, (u, v) in enumerate(edges):
        anc = anc_offset + i
        qc.cx(u, anc)
        qc.cx(v, anc)

    # Multi-controlled Z on all ancillas: flips phase iff every ancilla is 1,
    # i.e. iff every edge's endpoints differ -> proper coloring.
    ancillas = list(range(anc_offset, anc_offset + len(edges)))
    if len(ancillas) == 1:
        qc.z(ancillas[0])
    else:
        qc.h(ancillas[-1])
        qc.mcx(ancillas[:-1], ancillas[-1])
        qc.h(ancillas[-1])

    # Uncompute ancillas.
    for i, (u, v) in enumerate(edges):
        anc = anc_offset + i
        qc.cx(v, anc)
        qc.cx(u, anc)

    return qc


def build_diffuser(n_vertices):
    qc = QuantumCircuit(n_vertices, name="diffuser")
    qc.h(range(n_vertices))
    qc.x(range(n_vertices))
    qc.h(n_vertices - 1)
    qc.mcx(list(range(n_vertices - 1)), n_vertices - 1)
    qc.h(n_vertices - 1)
    qc.x(range(n_vertices))
    qc.h(range(n_vertices))
    return qc


n_solutions = len(classical_solutions)
search_space = 2 ** N_VERTICES
# Standard optimal Grover iteration count.
theta = np.arcsin(np.sqrt(n_solutions / search_space))
n_iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
print(f"\nGrover iterations used: {n_iterations} "
      f"(solutions={n_solutions}, space={search_space})")

n_ancilla = len(EDGES)
qc = QuantumCircuit(N_VERTICES + n_ancilla, N_VERTICES)
qc.h(range(N_VERTICES))

oracle = build_oracle(N_VERTICES, EDGES)
diffuser = build_diffuser(N_VERTICES)

for _ in range(n_iterations):
    qc.append(oracle.to_instruction(), range(N_VERTICES + n_ancilla))
    qc.append(diffuser.to_instruction(), range(N_VERTICES))

qc.measure(range(N_VERTICES), range(N_VERTICES))

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

print("\nMeasurement counts:")
for bitstring, count in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {bitstring}: {count}")

# The states Grover amplifies should be exactly the classical solution set.
# Take the top len(classical_solutions) most-frequent outcomes and compare.
sorted_outcomes = sorted(counts.items(), key=lambda kv: -kv[1])
top_outcomes = {bitstring for bitstring, _ in sorted_outcomes[:len(classical_bitstrings)]}

total_solution_shots = sum(c for b, c in counts.items() if b in classical_bitstrings)
solution_fraction = total_solution_shots / shots

print(f"\nClassical solution bitstrings: {sorted(classical_bitstrings)}")
print(f"Top quantum outcomes:          {sorted(top_outcomes)}")
print(f"Fraction of shots landing on a classical solution: {solution_fraction:.3f}")

verified = (top_outcomes == classical_bitstrings) and (solution_fraction > 0.9)

if verified:
    print("\nPASS: Grover search on the ideal AerSimulator amplified exactly "
          "the classically-verified proper 2-colorings of the 2-edge "
          "instance below Erdos problem #602's Property B threshold m(2)=3.")
else:
    print("\nFAIL: quantum outcomes did not match the classical solution set.")

assert verified
