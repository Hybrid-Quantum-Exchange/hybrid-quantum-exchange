"""
Erdos problem #613 (per data/problems.yaml in manman4/erdosproblems, read
2026-09-19): status "disproved (Lean)", tags ["graph theory", "ramsey"],
oeis: ["N/A"].

LIMITATION, stated up front: problem 613 has NO associated OEIS sequence
("N/A" in the source data), and the problems.yaml entry carries no statement
text in this repository snapshot -- only metadata (prize/status/tags). So
there is no OEIS sequence to test membership/terms of, and nothing here is a
literal OEIS value. Per the tags ("graph theory", "ramsey"), this script
instead builds a genuine, small, finite, computable Ramsey-theory property
in the same family as the problem and verifies it with both a classical
brute force and a real Grover search circuit. This is an honest substitute
grounded in the problem's own tags, not a fabricated stand-in for a missing
OEIS value, and it is reported as such (no OEIS id used).

The property tested:
  Does K4 (the complete graph on 4 vertices, 6 edges) admit a 2-coloring of
  its edges with no monochromatic triangle?
  This is exactly the finite question behind the Ramsey number R(3,3)=6:
  R(3,3)=6 means every 2-coloring of K6 DOES contain a monochromatic
  triangle, but for n<6 (here n=4) triangle-free-in-both-colors colorings
  exist. We verify this concretely for K4.

Encoding: K4 has C(4,2)=6 edges, each colored 0 or 1 -> search space of
2^6 = 64 colorings, encoded as 6 qubits (one per edge). K4 has C(4,3)=4
triangles. A coloring is "good" iff none of its 4 triangles is
monochromatic (all three edges the same color).

Classical step (ground truth, computed here from first principles):
  Brute-force all 64 colorings, evaluate the "good" predicate directly,
  and record the exact set of good colorings and its size M.

Quantum step (Grover search):
  Build a genuine oracle circuit that, for each of the 4 triangles, flags
  "monochromatic" via multi-controlled-X gates (one pattern for all-0, one
  for all-1) into an ancilla, ANDs the negations of the 4 flags into an
  "all good" ancilla, and phase-kicks that into a standard |-> phase
  ancilla to realize the boolean oracle as a phase oracle. This is
  followed by the standard Grover diffuser on the 6 coloring qubits, run
  for the optimal number of iterations given N=64 and the classically
  known M. We then measure and compare the quantum result's most likely
  outcome(s) against the classical good-coloring set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Problem setup: K4, its 6 edges, its 4 triangles.
# ---------------------------------------------------------------------------
VERTICES = [0, 1, 2, 3]
EDGES = list(combinations(VERTICES, 2))          # 6 edges, index 0..5
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(combinations(VERTICES, 3))       # 4 triangles


def triangle_edge_indices(tri):
    a, b, c = tri
    return (
        EDGE_INDEX[tuple(sorted((a, b)))],
        EDGE_INDEX[tuple(sorted((b, c)))],
        EDGE_INDEX[tuple(sorted((a, c)))],
    )


TRIANGLE_EDGE_IDX = [triangle_edge_indices(t) for t in TRIANGLES]
NUM_EDGE_QUBITS = len(EDGES)          # 6
NUM_TRI_ANCILLAS = len(TRIANGLES)     # 4
N = 2 ** NUM_EDGE_QUBITS              # 64


# ---------------------------------------------------------------------------
# Classical ground truth, derived from first principles (brute force).
# ---------------------------------------------------------------------------
def is_good_coloring(bits):
    """bits: tuple of 6 ints (0/1), one per edge, in EDGES order."""
    for i0, i1, i2 in TRIANGLE_EDGE_IDX:
        if bits[i0] == bits[i1] == bits[i2]:
            return False
    return True


def bits_from_int(x, n=NUM_EDGE_QUBITS):
    return tuple((x >> k) & 1 for k in range(n))


good_states = []
for x in range(N):
    if is_good_coloring(bits_from_int(x)):
        good_states.append(x)

M = len(good_states)
assert M > 0, "classical brute force found no good coloring -- unexpected for K4"
print(f"Classical brute force: N={N} colorings of K4's 6 edges, "
      f"{M} are triangle-free-in-both-colors (good).")
print(f"Sample good coloring (bits, edge order {EDGES}): "
      f"{bits_from_int(good_states[0])}")


# ---------------------------------------------------------------------------
# Quantum oracle: marks (phase-flips) exactly the "good" colorings.
# ---------------------------------------------------------------------------
def build_oracle():
    edge_q = list(range(NUM_EDGE_QUBITS))                       # 0..5
    tri_anc = list(range(NUM_EDGE_QUBITS, NUM_EDGE_QUBITS + NUM_TRI_ANCILLAS))  # 6..9
    all_good_anc = NUM_EDGE_QUBITS + NUM_TRI_ANCILLAS            # 10
    total_qubits = all_good_anc + 1                              # 11 (phase ancilla added by caller)

    qc = QuantumCircuit(total_qubits, name="oracle")
    mcx3 = MCXGate(3)

    # For each triangle, flag its ancilla = 1 if the 3 edges are monochromatic.
    for t_i, (i0, i1, i2) in enumerate(TRIANGLE_EDGE_IDX):
        anc = tri_anc[t_i]
        ctrls = [edge_q[i0], edge_q[i1], edge_q[i2]]
        # all-1 pattern
        qc.append(mcx3, ctrls + [anc])
        # all-0 pattern: flip controls, MCX, flip back
        qc.x(ctrls)
        qc.append(mcx3, ctrls + [anc])
        qc.x(ctrls)

    # all_good_anc = AND over triangles of (NOT tri_anc[t]) i.e. 1 iff no
    # triangle is monochromatic.
    qc.x(tri_anc)
    qc.append(MCXGate(NUM_TRI_ANCILLAS), tri_anc + [all_good_anc])
    qc.x(tri_anc)

    # Uncompute the triangle flags (they are reused across Grover iterations).
    for t_i, (i0, i1, i2) in enumerate(TRIANGLE_EDGE_IDX):
        anc = tri_anc[t_i]
        ctrls = [edge_q[i0], edge_q[i1], edge_q[i2]]
        qc.x(ctrls)
        qc.append(mcx3, ctrls + [anc])
        qc.x(ctrls)
        qc.append(mcx3, ctrls + [anc])

    return qc, edge_q, tri_anc, all_good_anc, total_qubits


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle_body, edge_q, tri_anc, all_good_anc, oracle_qubits_no_phase = build_oracle()
phase_anc = oracle_qubits_no_phase   # one extra qubit for the |-> phase trick
total_qubits = oracle_qubits_no_phase + 1

iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations chosen: {iterations} (N={N}, M={M})")

qc = QuantumCircuit(total_qubits, NUM_EDGE_QUBITS)

# Initialize: uniform superposition over the 6 edge-coloring qubits,
# phase ancilla in |->.
qc.h(edge_q)
qc.x(phase_anc)
qc.h(phase_anc)

for _ in range(iterations):
    # Oracle: compute all_good_anc, phase-kick via CX into |-> ancilla,
    # then uncompute (oracle_body handles compute+uncompute of tri_anc;
    # we insert the phase kick between compute and uncompute manually).
    edge_q_l = list(range(NUM_EDGE_QUBITS))
    tri_anc_l = list(range(NUM_EDGE_QUBITS, NUM_EDGE_QUBITS + NUM_TRI_ANCILLAS))
    mcx3 = MCXGate(3)

    # compute triangle flags
    for t_i, (i0, i1, i2) in enumerate(TRIANGLE_EDGE_IDX):
        anc = tri_anc_l[t_i]
        ctrls = [edge_q_l[i0], edge_q_l[i1], edge_q_l[i2]]
        qc.append(mcx3, ctrls + [anc])
        qc.x(ctrls)
        qc.append(mcx3, ctrls + [anc])
        qc.x(ctrls)

    qc.x(tri_anc_l)
    qc.append(MCXGate(NUM_TRI_ANCILLAS), tri_anc_l + [all_good_anc])
    qc.x(tri_anc_l)

    # phase kick: flips phase_anc (in |->) iff all_good_anc == 1
    qc.cx(all_good_anc, phase_anc)

    # uncompute all_good_anc
    qc.x(tri_anc_l)
    qc.append(MCXGate(NUM_TRI_ANCILLAS), tri_anc_l + [all_good_anc])
    qc.x(tri_anc_l)

    # uncompute triangle flags
    for t_i, (i0, i1, i2) in enumerate(TRIANGLE_EDGE_IDX):
        anc = tri_anc_l[t_i]
        ctrls = [edge_q_l[i0], edge_q_l[i1], edge_q_l[i2]]
        qc.x(ctrls)
        qc.append(mcx3, ctrls + [anc])
        qc.x(ctrls)
        qc.append(mcx3, ctrls + [anc])

    # Diffuser on the 6 edge qubits.
    qc.h(edge_q)
    qc.x(edge_q)
    qc.h(NUM_EDGE_QUBITS - 1)
    qc.append(MCXGate(NUM_EDGE_QUBITS - 1), list(range(NUM_EDGE_QUBITS - 1)) + [NUM_EDGE_QUBITS - 1])
    qc.h(NUM_EDGE_QUBITS - 1)
    qc.x(edge_q)
    qc.h(edge_q)

qc.measure(edge_q, list(range(NUM_EDGE_QUBITS)))

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order: classical register c[0] (edge 0, LSB) is the rightmost
# character of the count key.
good_state_strs = {format(x, f"0{NUM_EDGE_QUBITS}b")[::-1] for x in good_states}

hits_on_good = sum(c for bitstr, c in counts.items() if bitstr in good_state_strs)
frac_good = hits_on_good / shots

top_bitstr, top_count = max(counts.items(), key=lambda kv: kv[1])
top_is_good = top_bitstr in good_state_strs

print(f"Quantum result: {frac_good:.3f} of {shots} shots landed on a "
      f"classically-verified good coloring (M/N = {M / N:.3f} is the "
      f"uniform-random baseline).")
print(f"Most frequent measured outcome: {top_bitstr} (count {top_count}), "
      f"classically good = {top_is_good}")

# PASS criterion: Grover must concentrate probability on good states well
# above the uniform baseline, and the single most likely outcome must
# itself be a classically-verified good coloring.
passed = top_is_good and frac_good > (M / N) * 1.5

print("PASS" if passed else "FAIL")
