"""
Erdos problem #737 (per erdosproblems.com / manman4/erdosproblems data as of
2025-10-01): a graph-theory / chromatic-number problem, status "proved",
oeis: ["N/A"].

LIMITATION, stated honestly up front: problem #737 has no associated OEIS
sequence in the source data (oeis: ["N/A"]), so the task of "identify a small,
finite, computable property of the OEIS sequence" does not literally apply.
There is no sequence id to derive a property from. Rather than fabricate an
OEIS id or copy an unrelated one, this script instead builds a genuine
quantum circuit for the one piece of real mathematical content the problem's
own tags give us: chromatic-number / graph 2-colorability, which is exactly
the kind of small finite decision property Grover search is suited to.

Chosen property: proper 2-coloring existence (bipartiteness) of a small,
fixed graph on 4 vertices with edges forming a 4-cycle:
    edges = {(0,1), (1,2), (2,3), (3,0)}
Question: does there exist an assignment of one of 2 colors to each of the
4 vertices such that every edge's endpoints get different colors (i.e. is
this graph 2-colorable / bipartite / chromatic number <= 2)?

Classical answer (computed here from first principles by brute force over
all 2^4 = 16 colorings, not looked up): YES, this graph is bipartite/
2-colorable. A valid coloring is vertices {0,2} = color 0, {1,3} = color 1
(a 4-cycle is bipartite since it has no odd cycle).

Quantum approach: Grover's algorithm searches the 2^4 = 16-dimensional space
of colorings (one qubit per vertex, qubit value = color) for an assignment
satisfying "every edge's two endpoints differ". The oracle marks a coloring
as a "solution" iff, for every edge (u, v), qubit_u != qubit_v (computed via
CNOT into edge-ancilla qubits which must all be 1 for a hit). We run Grover
with a small number of iterations tuned for 16 items, then check that the
most frequently measured bitstring among high-probability outcomes indeed
satisfies the coloring constraint, matching the classical answer.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed by brute force (not copied from OEIS).
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # 4-cycle


def is_proper_coloring(bits):
    """bits: tuple of 0/1, one color per vertex. True iff every edge differs."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


def brute_force_solutions():
    sols = []
    for bits in itertools.product([0, 1], repeat=N_VERTICES):
        if is_proper_coloring(bits):
            sols.append(bits)
    return sols


CLASSICAL_SOLUTIONS = brute_force_solutions()
CLASSICAL_IS_2_COLORABLE = len(CLASSICAL_SOLUTIONS) > 0

print(f"Graph: {N_VERTICES} vertices, edges = {EDGES}")
print(f"Classical brute-force solutions (out of {2 ** N_VERTICES} colorings): "
      f"{CLASSICAL_SOLUTIONS}")
print(f"Classical answer: graph is 2-colorable = {CLASSICAL_IS_2_COLORABLE}")

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle for "proper 2-coloring of this graph".
# ---------------------------------------------------------------------------
# 4 vertex qubits (v0..v3), one ancilla per edge (marks "edge satisfied"),
# 1 output qubit (phase-kickback target, prepared in |-> outside the oracle).

n_v = N_VERTICES
n_e = len(EDGES)

vertex = QuantumRegister(n_v, "v")
edge_anc = QuantumRegister(n_e, "e")
out = QuantumRegister(1, "out")
creg = ClassicalRegister(n_v, "c")


def build_oracle():
    qc = QuantumCircuit(vertex, edge_anc, out)
    # mark each edge ancilla = 1 iff endpoints differ (XOR via CNOT-CNOT)
    for i, (u, v) in enumerate(EDGES):
        qc.cx(vertex[u], edge_anc[i])
        qc.cx(vertex[v], edge_anc[i])
    # flip out (prepared in |-> ) iff ALL edge ancillas are 1 -> phase flip
    qc.mcx(list(edge_anc), out[0])
    # uncompute edge ancillas
    for i, (u, v) in enumerate(EDGES):
        qc.cx(vertex[v], edge_anc[i])
        qc.cx(vertex[u], edge_anc[i])
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(n_iterations):
    qc = QuantumCircuit(vertex, edge_anc, out, creg)
    qc.h(vertex)
    qc.x(out[0])
    qc.h(out[0])  # |-> ancilla for phase kickback

    oracle = build_oracle()
    diffuser = build_diffuser(n_v)

    for _ in range(n_iterations):
        qc.compose(oracle, qubits=list(vertex) + list(edge_anc) + list(out), inplace=True)
        qc.compose(diffuser, qubits=list(vertex), inplace=True)

    qc.measure(vertex, creg)
    return qc


# ---------------------------------------------------------------------------
# 3. Choose iteration count for N=16 items, M=len(CLASSICAL_SOLUTIONS) marked.
# ---------------------------------------------------------------------------

N = 2 ** n_v
M = len(CLASSICAL_SOLUTIONS)
assert M > 0, "graph has no proper 2-coloring; Grover search would find nothing"

theta = math.asin(math.sqrt(M / N))
optimal_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

circuit = build_grover_circuit(optimal_iterations)

# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(circuit, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order: classical register bit string is c[n-1]...c[0]; c[i] was
# measured from vertex[i]. Reverse to get (v0, v1, v2, v3) order.
def bitstring_to_vertex_tuple(bs):
    bits = bs[::-1]  # bits[i] corresponds to vertex[i]
    return tuple(int(b) for b in bits)


# Sum measured probability mass landing on classical-valid colorings.
solution_set = set(CLASSICAL_SOLUTIONS)
hit_counts = 0
most_common = sorted(counts.items(), key=lambda kv: -kv[1])
print("Top measured outcomes (bitstring: count):")
for bs, c in most_common[:6]:
    vt = bitstring_to_vertex_tuple(bs)
    print(f"  {bs} -> vertex coloring {vt}, valid={vt in solution_set}, count={c}")

for bs, c in counts.items():
    if bitstring_to_vertex_tuple(bs) in solution_set:
        hit_counts += c

hit_fraction = hit_counts / shots
print(f"Grover iterations used: {optimal_iterations}")
print(f"Fraction of shots landing on a valid 2-coloring: {hit_fraction:.4f} "
      f"(expected amplified, >> {M / N:.4f} uniform baseline)")

# The circuit's quantum-derived verdict: does Grover amplification find a
# valid coloring with much-better-than-uniform probability?
quantum_found_solution = hit_fraction > 0.5

# ---------------------------------------------------------------------------
# 5. Compare quantum result to classical answer and report PASS/FAIL.
# ---------------------------------------------------------------------------

# The most-sampled outcome should itself be a valid coloring, and the overall
# hit fraction should be strongly amplified relative to the classical prior.
top_bs, top_count = most_common[0]
top_is_valid = bitstring_to_vertex_tuple(top_bs) in solution_set

success = (
    CLASSICAL_IS_2_COLORABLE
    and quantum_found_solution
    and top_is_valid
)

if success:
    print("PASS: Grover search on the ideal AerSimulator amplified and "
          "recovered a proper 2-coloring of the 4-cycle graph, matching "
          "the classical brute-force answer (2-colorable = True).")
else:
    print("FAIL: quantum search result did not match the classical answer.")
