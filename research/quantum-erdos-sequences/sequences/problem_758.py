"""
Erdos problem #758 (data/problems.yaml, entry "number: '758'"):
  tags: ["graph theory", "chromatic number"]
  oeis: ["possible"]   <-- NOT a real OEIS sequence id. The upstream YAML uses
  the literal placeholder string "possible" in the oeis field for this
  problem (and for #759, its neighbor), instead of an A-number. There is no
  concrete integer sequence attached to problem #758 in the source data, so
  this script cannot honestly build a circuit "from the OEIS sequence for
  #758" the way most other lanes in this library can.

LIMITATION (stated honestly, per instructions): with no OEIS id to derive a
sequence-membership or sequence-term property from, this script instead
targets the one piece of real mathematical content actually present in the
problem's metadata: its tag "chromatic number" / graph coloring. It builds a
genuine, small, computable decision problem in that area -- k-colorability of
a small graph -- and solves it two ways:

  1. Classically, by brute-force search over all colorings (ground truth).
  2. Quantum-mechanically, with a real Grover search circuit built from a
     phase oracle over the graph's edge constraints, run on the ideal
     AerSimulator.

Concrete instance: the 4-cycle graph C4 (vertices 0-1-2-3-0), 2 colors per
vertex (1 qubit per vertex, 4 qubits total). A coloring is valid iff every
edge connects differently-colored vertices, i.e. XOR(color_u, color_v) = 1
for each edge (u, v). C4 is bipartite, so its 2-colorings are exactly the two
alternating colorings: 0101 and 1010 (out of 16 possible assignments) -- this
is verified in-script by brute force, not asserted.

The Grover oracle computes, for each of the 4 edges, the XOR of its two
endpoint qubits into a dedicated "edge satisfied" ancilla via a CNOT-CNOT
pair, then phase-flips the state only when all 4 edge ancillas are 1
(multi-controlled Z into a phase kickback qubit), then uncomputes the edge
ancillas. This is a genuine oracle over the actual constraint structure of
the graph, not a hard-coded marked-state oracle.

With N = 16 basis states and M = 2 marked (valid) colorings, the optimal
number of Grover iterations is round(pi/4 * sqrt(N/M)) = round(pi/4*sqrt(8))
= 2, which the script also derives itself rather than hard-coding.

PASS criterion: after running the Grover circuit on AerSimulator and taking
the most frequently measured 4-qubit string, that string must classically
satisfy the C4 proper-2-coloring constraint (i.e. be one of the two colorings
found by the brute-force ground truth), and the two valid colorings together
must account for a large majority of the measured shots (amplitude
amplification concentrated onto the marked subspace).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. The graph instance and the classical ground truth (brute force).
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # C4, 4-cycle


def is_valid_coloring(bits):
    """bits: tuple of 0/1, one per vertex. Valid iff every edge's endpoints differ."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


def classical_brute_force():
    valid = []
    for bits in itertools.product([0, 1], repeat=N_VERTICES):
        if is_valid_coloring(bits):
            valid.append(bits)
    return valid


VALID_COLORINGS = classical_brute_force()
N_STATES = 2 ** N_VERTICES
N_MARKED = len(VALID_COLORINGS)

assert N_MARKED == 2, f"expected exactly 2 valid 2-colorings of C4, found {N_MARKED}"
assert set(VALID_COLORINGS) == {(0, 1, 0, 1), (1, 0, 1, 0)}, VALID_COLORINGS

print("Classical ground truth:")
print(f"  graph: C4 on vertices {list(range(N_VERTICES))}, edges {EDGES}")
print(f"  search space size N = {N_STATES}")
print(f"  valid 2-colorings (brute force) = {VALID_COLORINGS}  (M = {N_MARKED})")


# ---------------------------------------------------------------------------
# 2. Grover search circuit built from a real oracle over the edge constraints.
# ---------------------------------------------------------------------------
# Qubit layout:
#   q0..q3   : the 4 vertex-color qubits (the search register)
#   q4..q7   : one ancilla per edge, holds XOR(color_u, color_v) for that edge
#   q8       : phase-kickback target qubit (prepared in |-> once, outside loop)

N_VERT_Q = N_VERTICES
N_EDGE_Q = len(EDGES)
PHASE_Q = N_VERT_Q + N_EDGE_Q  # index of the single phase-kick qubit
TOTAL_Q = N_VERT_Q + N_EDGE_Q + 1


def apply_oracle(qc):
    """Phase-flips basis states whose 4 vertex bits form a valid C4 2-coloring."""
    edge_qubits = list(range(N_VERT_Q, N_VERT_Q + N_EDGE_Q))

    # Compute edge-satisfied ancillas: edge_q[i] = color_u XOR color_v
    for i, (u, v) in enumerate(EDGES):
        eq = edge_qubits[i]
        qc.cx(u, eq)
        qc.cx(v, eq)

    # Multi-controlled Z (via phase kickback on PHASE_Q, prepared in |->)
    # flips the phase exactly when ALL edge ancillas are 1, i.e. every edge
    # constraint is satisfied.
    qc.mcx(edge_qubits, PHASE_Q)

    # Uncompute the edge ancillas so they can be reused next iteration.
    for i, (u, v) in enumerate(EDGES):
        eq = edge_qubits[i]
        qc.cx(v, eq)
        qc.cx(u, eq)


def apply_diffuser(qc):
    """Standard Grover diffuser over the 4 vertex-color qubits."""
    verts = list(range(N_VERT_Q))
    qc.h(verts)
    qc.x(verts)
    qc.h(verts[-1])
    qc.mcx(verts[:-1], verts[-1])
    qc.h(verts[-1])
    qc.x(verts)
    qc.h(verts)


n_iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / N_MARKED)))
print(f"Grover iterations (derived from N={N_STATES}, M={N_MARKED}): {n_iterations}")

qc = QuantumCircuit(TOTAL_Q, N_VERT_Q)

# Uniform superposition over the 4 vertex-color qubits.
qc.h(range(N_VERT_Q))

# Phase-kickback ancilla prepared in |->.
qc.x(PHASE_Q)
qc.h(PHASE_Q)

for _ in range(n_iterations):
    apply_oracle(qc)
    apply_diffuser(qc)

qc.measure(range(N_VERT_Q), range(N_VERT_Q))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

SHOTS = 4096
sim = AerSimulator()
result = sim.run(qc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit order is little-endian in the classical register string
# (rightmost char = qubit 0), so reverse to get (q0, q1, q2, q3) order.
def bitstring_to_tuple(bs):
    rev = bs[::-1]
    return tuple(int(c) for c in rev)

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_bitstring, top_count = sorted_counts[0]
top_bits = bitstring_to_tuple(top_bitstring)

marked_shot_total = sum(
    c for bs, c in counts.items() if bitstring_to_tuple(bs) in VALID_COLORINGS
)
marked_fraction = marked_shot_total / SHOTS

print()
print("Quantum result:")
print(f"  measurement counts (top 5): {sorted_counts[:5]}")
print(f"  most frequent outcome: {top_bitstring} -> vertex colors {top_bits} (count {top_count}/{SHOTS})")
print(f"  fraction of shots landing on a valid coloring: {marked_fraction:.3f}")

quantum_found_valid = top_bits in VALID_COLORINGS
amplification_worked = marked_fraction > 0.8  # uniform baseline would be 2/16 = 0.125

ok = quantum_found_valid and amplification_worked

print()
if ok:
    print("PASS")
else:
    print("FAIL")
    if not quantum_found_valid:
        print(f"  top measured coloring {top_bits} is not a valid C4 2-coloring")
    if not amplification_worked:
        print(f"  marked-state fraction {marked_fraction:.3f} too low (expected concentration via Grover)")
