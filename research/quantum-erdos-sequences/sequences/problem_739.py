"""
Erdos problem #739 (from https://github.com/manman4/erdosproblems,
data/problems.yaml entry `number: "739"`) has metadata:

    oeis: ["N/A"]
    tags: ["graph theory", "chromatic number"]

There is NO OEIS sequence attached to this problem (oeis is literally the
placeholder "N/A"), so the task of "identify a small, finite, computable
property of the sequence" as originally specified does not apply here:
there is no sequence to test membership/terms of. This is an honest
limitation, not a bug to route around.

Rather than fabricate an OEIS-backed property, this script instead builds a
genuine, honestly-scoped quantum circuit around the one piece of real
mathematical content problem #739 does carry in its metadata: it is a
graph-theory / chromatic-number problem. We pick a small, fully concrete
instance of the generic decision problem that "chromatic number" work of
this kind revolves around -- k-colorability of a small graph -- and use
Grover's algorithm to search for a proper 2-coloring (equivalently: a valid
assignment of 2 "colors" to vertices such that no edge is monochromatic).

Concretely:
  - Graph: the 4-cycle C4 (vertices 0,1,2,3; edges (0,1),(1,2),(2,3),(3,0)).
  - Property being tested: "C4 is 2-colorable", i.e. there exists a
    2-coloring x in {0,1}^4 such that for every edge (u,v), x_u != x_v.
  - Classical answer: computed in this script by brute-force enumeration of
    all 2^4 = 16 colorings (first principles, no external data), which
    finds exactly 2 valid colorings: 0101 and 1010 (bipartite class swap).
  - Quantum side: a genuine Grover search circuit over 4 qubits (one per
    vertex) is built. The oracle phase-flips exactly the valid-coloring
    basis states (computed via the same classical edge-check logic,
    compiled into a multi-controlled-Z oracle over auxiliary XOR/ancilla
    qubits), and Grover diffusion amplifies them. The circuit is run on the
    ideal AerSimulator and the most frequent measured bitstrings are
    compared against the classically-known set of valid colorings.

PASS/FAIL: PASS if Grover's algorithm's most-sampled outcomes are exactly
the classically verified valid 2-colorings of C4 (up to the expected
sampling noise threshold).

Honesty note (required by task spec): this script's classical property
("C4 is 2-colorable, and its valid colorings are {0101,1010}") is NOT taken
from an OEIS sequence -- none exists for problem #739. It is a real,
independently-verifiable computable property in the same tag space
("graph theory", "chromatic number") as the problem's metadata, checked
classically in this script and then verified with a real Grover circuit.
ran_ok / verified_against_classical below are reported accurately.
"""

import itertools

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# 1. Classical ground truth (first principles, brute force).
# ---------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # C4


def is_proper_2coloring(bits):
    """bits: tuple of 0/1 of length N_VERTICES. True iff no edge monochromatic."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


classical_valid = []
for bits in itertools.product([0, 1], repeat=N_VERTICES):
    if is_proper_2coloring(bits):
        classical_valid.append(bits)

# Expected: exactly 2 valid colorings for C4 (bipartition and its complement).
assert len(classical_valid) == 2, f"unexpected classical result: {classical_valid}"

# Bitstrings as Qiskit prints them: qubit 0 is the rightmost character.
classical_valid_strs = set(
    "".join(str(b) for b in reversed(bits)) for bits in classical_valid
)

print("Classical brute-force search over all 2^4 colorings of C4:")
for bits in itertools.product([0, 1], repeat=N_VERTICES):
    tag = "VALID" if is_proper_2coloring(bits) else ""
    print(f"  {bits} {tag}")
print(f"Classically valid 2-colorings (vertex order q3q2q1q0): {sorted(classical_valid_strs)}")

# ---------------------------------------------------------------------
# 2. Grover oracle construction.
#
# For each edge (u, v) we need "x_u != x_v", i.e. XOR(x_u, x_v) == 1.
# We compute that XOR into an ancilla qubit per edge using CNOTs, then
# flip the phase of the state only when ALL edge-ancillas are 1
# (multi-controlled Z), then uncompute the ancillas (to disentangle them
# so the diffusion operator acts cleanly on the vertex register alone).
# ---------------------------------------------------------------------

n_edges = len(EDGES)
vertex = QuantumRegister(N_VERTICES, "v")
anc = QuantumRegister(n_edges, "e")  # one ancilla per edge, holds XOR(x_u, x_v)


def build_oracle():
    qc = QuantumCircuit(vertex, anc, name="oracle")
    # compute edge XORs into ancillas
    for i, (u, v) in enumerate(EDGES):
        qc.cx(vertex[u], anc[i])
        qc.cx(vertex[v], anc[i])
    # multi-controlled Z on all ancillas being 1 (all edges properly colored)
    qc.h(anc[n_edges - 1])
    qc.mcx(list(anc[: n_edges - 1]), anc[n_edges - 1])
    qc.h(anc[n_edges - 1])
    # uncompute
    for i, (u, v) in enumerate(EDGES):
        qc.cx(vertex[v], anc[i])
        qc.cx(vertex[u], anc[i])
    return qc


def build_diffuser():
    qc = QuantumCircuit(vertex, name="diffuser")
    qc.h(vertex)
    qc.x(vertex)
    qc.h(vertex[N_VERTICES - 1])
    qc.mcx(list(vertex[: N_VERTICES - 1]), vertex[N_VERTICES - 1])
    qc.h(vertex[N_VERTICES - 1])
    qc.x(vertex)
    qc.h(vertex)
    return qc


# Number of "good" states M = 2 out of N = 16 search space -> optimal
# Grover iterations r ~ floor(pi/4 * sqrt(N/M)).
N_STATES = 2 ** N_VERTICES
M_GOOD = len(classical_valid)
iterations = max(1, round((np.pi / 4) * np.sqrt(N_STATES / M_GOOD)))
print(f"\nGrover: N={N_STATES} states, M={M_GOOD} valid colorings, iterations={iterations}")

creg = ClassicalRegister(N_VERTICES, "c")
qc = QuantumCircuit(vertex, anc, creg)
qc.h(vertex)
oracle = build_oracle()
diffuser = build_diffuser()
for _ in range(iterations):
    qc.append(oracle.to_instruction(), list(vertex) + list(anc))
    qc.append(diffuser.to_instruction(), list(vertex))
qc.measure(vertex, creg)

# ---------------------------------------------------------------------
# 3. Run on ideal AerSimulator.
# ---------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print("\nTop measured outcomes (bitstring: count):")
for bs, c in sorted_counts[:6]:
    print(f"  {bs}: {c}")

# The two most frequent outcomes should be exactly the classically valid
# colorings, each amplified to roughly half the shots.
top2 = set(bs for bs, _ in sorted_counts[:2])
top2_total = sum(c for bs, c in sorted_counts[:2])

verified = (top2 == classical_valid_strs) and (top2_total / shots > 0.8)

print(f"\nExpected valid colorings: {sorted(classical_valid_strs)}")
print(f"Grover top-2 outcomes:    {sorted(top2)}")
print(f"Top-2 share of shots:     {top2_total / shots:.3f}")

if verified:
    print("\nPASS: Grover search recovered exactly the classically verified "
          "proper 2-colorings of C4 with high probability.")
else:
    print("\nFAIL: Grover search did not match the classical ground truth.")

ran_ok = True
verified_against_classical = verified
