"""
Erdos problem #745 -- quantum-testable-sequence lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 745"):
    prize: no
    status: proved (as of 2025-08-31)
    oeis: ["N/A"]
    tags: ["graph theory"]

HONEST LIMITATION, stated up front: problem #745's metadata record carries no
OEIS sequence id ("N/A"). There is therefore no OEIS sequence for this lane to
derive a finite, checkable numeric property from -- the requested "identify a
small, finite, computable property of the [OEIS] sequence" step has no real
sequence to anchor to. Per instructions for this case, rather than fabricate
an OEIS-derived property or copy a value with no real content, this script
instead builds a genuine, self-contained finite decision problem from the
one piece of real content the metadata *does* give us: the tag "graph theory".

Chosen property (real, independently checkable, nothing invented about #745
itself): does the 4-cycle graph C4 (vertices 0,1,2,3; edges 0-1,1-2,2-3,3-0)
admit a proper 2-coloring? A proper 2-coloring assigns a color in {0,1} to
each vertex such that every edge joins differently-colored vertices. This is
a small, finite, exactly computable search problem (2^4 = 16 candidate
colorings), well matched to a genuine Grover search circuit, and its answer
is checked here classically by direct brute force before the quantum run.

Classical fact checked in this script: C4 is bipartite, so it has exactly 2
proper 2-colorings among the 16 possible bit-strings: 0101 and 1010 (vertex
order v0 v1 v2 v3, bit = color).

Circuit: exact Grover search (4 data qubits + 1 phase-kickback ancilla) with
the optimal integer number of Grover iterations for N=16, M=2 marked items,
run on the ideal AerSimulator. The oracle is built directly from the C4 edge
list (XOR-based inequality checks per edge, multi-controlled phase flip),
not from a lookup table of the answer -- it recomputes "is this coloring
proper" in-circuit.

PASS criterion: the two most frequent measurement outcomes after Grover
search are exactly the two classically-verified proper 2-colorings.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

ERDOS_PROBLEM = 745
OEIS_IDS = ["N/A"]  # no OEIS id present in the source metadata for this problem

# ---------------------------------------------------------------------------
# Graph definition: C4 (4-cycle)
# ---------------------------------------------------------------------------
N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]

# ---------------------------------------------------------------------------
# Step 1: classical ground truth, computed from first principles by brute force
# ---------------------------------------------------------------------------


def is_proper_2coloring(bits):
    """bits: tuple of 0/1, length N_VERTICES. True iff every edge is bichromatic."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


def classical_solutions():
    sols = []
    for bits in itertools.product([0, 1], repeat=N_VERTICES):
        if is_proper_2coloring(bits):
            sols.append(bits)
    return sols


CLASSICAL_SOLUTIONS = classical_solutions()
N = 2 ** N_VERTICES
M = len(CLASSICAL_SOLUTIONS)

print(f"Erdos problem #{ERDOS_PROBLEM}, OEIS ids used: {OEIS_IDS} (none available)")
print(f"Graph: C4 on vertices {list(range(N_VERTICES))}, edges {EDGES}")
print(f"Classical brute force over all {N} colorings found {M} proper 2-colorings:")
for s in CLASSICAL_SOLUTIONS:
    print(f"  {''.join(map(str, s))}")

assert M == 2, "Sanity check on classical brute force failed unexpectedly."
assert set(CLASSICAL_SOLUTIONS) == {(0, 1, 0, 1), (1, 0, 1, 0)}

# ---------------------------------------------------------------------------
# Step 2: Grover search circuit whose oracle recomputes properness in-circuit
# ---------------------------------------------------------------------------
# Qubit layout: q0..q3 = vertex colors v0..v3 ; q4 = phase-kickback ancilla
# (kept in |-> throughout, so a multi-controlled-X onto it implements a
# controlled phase flip -Z on the "all conditions true" state).


def build_oracle(qc, data, edge_anc, out):
    """Marks (phase-flips) computational basis states where every edge is
    bichromatic, i.e. proper 2-colorings of C4.

    edge_anc: one ancilla per edge, set to 1 iff that edge's endpoints differ.
    out: the phase-kickback ancilla (assumed already in |->).
    """
    # compute per-edge "different colors" flags via CNOT-based XOR into fresh ancillas
    for i, (u, v) in enumerate(EDGES):
        qc.cx(data[u], edge_anc[i])
        qc.cx(data[v], edge_anc[i])  # edge_anc[i] = data[u] XOR data[v]

    # phase-flip when ALL edge flags are 1 (all edges bichromatic)
    qc.mcx(edge_anc, out)

    # uncompute the ancillas (must undo in reverse to restore |0> cleanly)
    for i, (u, v) in reversed(list(enumerate(EDGES))):
        qc.cx(data[v], edge_anc[i])
        qc.cx(data[u], edge_anc[i])


def build_diffuser(qc, data):
    n = len(data)
    qc.h(data)
    qc.x(data)
    qc.h(data[-1])
    qc.mcx(data[:-1], data[-1])
    qc.h(data[-1])
    qc.x(data)
    qc.h(data)


def grover_iterations(n_items, n_marked):
    theta = np.arcsin(np.sqrt(n_marked / n_items))
    return max(1, round((np.pi / (4 * theta)) - 0.5))


data = QuantumRegister(N_VERTICES, "v")
edge_anc = QuantumRegister(len(EDGES), "e")
out = QuantumRegister(1, "out")
creg = ClassicalRegister(N_VERTICES, "c")

qc = QuantumCircuit(data, edge_anc, out, creg)

# init: uniform superposition over colorings, ancilla in |->
qc.h(data)
qc.x(out)
qc.h(out)

iterations = grover_iterations(N, M)
print(f"Grover iterations used: {iterations} (N={N}, M={M})")

for _ in range(iterations):
    build_oracle(qc, data, edge_anc, out)
    build_diffuser(qc, data)

qc.measure(data, creg)

# ---------------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator
# ---------------------------------------------------------------------------
sim = AerSimulator()
tqc = transpile(qc, sim)
SHOTS = 4096
result = sim.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit order in the returned bitstring is q(n-1)...q0 for the classical
# register; our register was built [v0 v1 v2 v3], so reverse to get v0..v3 order.
def to_vertex_order(bitstring):
    return tuple(int(b) for b in reversed(bitstring))


counts_by_vertex_order = {}
for bitstring, freq in counts.items():
    v = to_vertex_order(bitstring)
    counts_by_vertex_order[v] = counts_by_vertex_order.get(v, 0) + freq

top2 = sorted(counts_by_vertex_order.items(), key=lambda kv: -kv[1])[:2]
print("Top measurement outcomes (vertex order v0v1v2v3): counts")
for v, freq in sorted(counts_by_vertex_order.items(), key=lambda kv: -kv[1])[:6]:
    print(f"  {''.join(map(str, v))}: {freq}")

top2_states = set(v for v, _ in top2)
classical_set = set(CLASSICAL_SOLUTIONS)

verified = top2_states == classical_set

print()
print(f"Classical proper 2-colorings: {sorted(classical_set)}")
print(f"Quantum top-2 measured states: {sorted(top2_states)}")

if verified:
    print("PASS")
    sys.exit(0)
else:
    print("FAIL")
    sys.exit(1)
