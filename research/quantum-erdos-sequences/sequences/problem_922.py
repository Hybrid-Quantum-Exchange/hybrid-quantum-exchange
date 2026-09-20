"""
Erdos problem #922 (from erdosproblems.com / the manman4/erdosproblems data
dump) is tagged ["graph theory", "chromatic number"] and its `oeis` field in
data/problems.yaml is literally ["N/A"] -- there is no OEIS sequence attached
to this problem. That means the assignment "identify a small computable
property of the OEIS sequence for problem #922" has no sequence to draw on.

Rather than fabricate an OEIS id or bolt on an unrelated sequence, this
script stays honest about that gap and instead builds a genuine quantum
circuit around the one real piece of mathematical content the metadata does
give us: the "chromatic number" tag. The chosen finite, computable property is

    Does the path graph P4 (vertices 0-1-2-3, edges (0,1),(1,2),(2,3))
    have a proper 2-coloring?

This is a legitimate small instance of the general graph-coloring question
that chromatic-number problems are about. It is classically computed here
from first principles (brute-force enumeration of all 2^4 = 16 colorings)
and then verified with a real Grover search circuit run on AerSimulator.

Classical ground truth (computed below, not copied from anywhere):
  - P4 has exactly 2 proper 2-colorings out of 16 total assignments:
    0101 and 1010 (reading qubit/vertex order 0,1,2,3).

Quantum approach:
  - 4 "color" qubits (one bit per vertex, 2 colors).
  - 3 ancilla qubits compute the XOR (color-difference) of each edge's two
    endpoints via CNOTs (an edge is "properly colored" iff its endpoint
    colors differ, i.e. XOR = 1).
  - A multi-controlled Z (via phase kickback on a flag qubit) marks the
    states where all three edge-ancillas are 1, i.e. all edges are properly
    colored -- this is the Grover oracle for "valid 2-coloring of P4".
  - Ancillas are uncomputed, a standard Grover diffuser is applied on the 4
    color qubits, and the circuit is run for the classically-optimal number
    of Grover iterations for N=16, M=2 marked states.
  - PASS/FAIL: the two most frequently measured 4-bit strings from the
    quantum run must equal exactly the classically-computed valid-coloring
    set {0101, 1010} (bit order matches vertex order 0,1,2,3, using Qiskit's
    little-endian bit ordering handled explicitly below).

Limitation, stated plainly: this is NOT a circuit over an OEIS sequence for
problem #922, because no such sequence exists in the source data (oeis:
["N/A"]). It is the most genuine, non-fabricated quantum computation that
could be built from what problem #922's metadata actually contains (its
chromatic-number tag), verified against a first-principles classical
computation of the same finite instance.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

ERDOS_PROBLEM_NUMBER = 922
OEIS_IDS_USED = []  # none exist for this problem; see docstring

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2), (2, 3)]  # path graph P4 on vertices 0,1,2,3
NUM_VERTICES = 4


def is_proper_2_coloring(coloring):
    """coloring: tuple of 0/1, one color per vertex, index = vertex id."""
    return all(coloring[u] != coloring[v] for (u, v) in EDGES)


def classical_valid_colorings():
    valid = []
    for bits in itertools.product([0, 1], repeat=NUM_VERTICES):
        if is_proper_2_coloring(bits):
            valid.append(bits)
    return valid


CLASSICAL_VALID = classical_valid_colorings()
# Bitstrings as they will be compared to quantum measurement output,
# vertex 0 is the most-significant character when printed left-to-right
# in "vertex 0,1,2,3" order.
CLASSICAL_VALID_STRINGS = {"".join(str(b) for b in c) for c in CLASSICAL_VALID}

assert CLASSICAL_VALID_STRINGS == {"0101", "1010"}, CLASSICAL_VALID_STRINGS

N_STATES = 2 ** NUM_VERTICES  # 16
M_MARKED = len(CLASSICAL_VALID)  # 2

# ---------------------------------------------------------------------------
# 2. Grover circuit: color qubits q0..q3 (vertex 0..3), edge ancillas a0..a2,
#    one phase-kickback flag qubit f.
# ---------------------------------------------------------------------------

NUM_COLOR_QUBITS = NUM_VERTICES
NUM_EDGE_ANCILLAS = len(EDGES)

COLOR = list(range(NUM_COLOR_QUBITS))               # 0,1,2,3
ANCILLA = list(range(NUM_COLOR_QUBITS, NUM_COLOR_QUBITS + NUM_EDGE_ANCILLAS))  # 4,5,6
FLAG = NUM_COLOR_QUBITS + NUM_EDGE_ANCILLAS          # 7
NUM_QUBITS = FLAG + 1


def build_oracle(qc):
    """Marks (phase-flips) computational basis states that are proper
    2-colorings of P4, by computing per-edge XOR into ancillas and doing a
    multi-controlled Z (via phase kickback on FLAG) when all ancillas are 1.
    """
    # compute edge XORs into ancillas
    for anc_idx, (u, v) in zip(ANCILLA, EDGES):
        qc.cx(COLOR[u], anc_idx)
        qc.cx(COLOR[v], anc_idx)

    # multi-controlled Z on ANCILLA (all must be 1) via phase kickback on FLAG
    qc.mcx(ANCILLA, FLAG)

    # uncompute edge XORs
    for anc_idx, (u, v) in zip(ANCILLA, EDGES):
        qc.cx(COLOR[v], anc_idx)
        qc.cx(COLOR[u], anc_idx)


def build_diffuser(qc):
    qc.h(COLOR)
    qc.x(COLOR)
    qc.h(COLOR[-1])
    qc.mcx(COLOR[:-1], COLOR[-1])
    qc.h(COLOR[-1])
    qc.x(COLOR)
    qc.h(COLOR)


def build_grover_circuit(num_iterations):
    qc = QuantumCircuit(NUM_QUBITS, NUM_COLOR_QUBITS)

    # FLAG starts in |1>, then H -> |-> so mcx on it gives a phase kickback
    qc.x(FLAG)
    qc.h(FLAG)

    # uniform superposition over color assignments
    qc.h(COLOR)

    for _ in range(num_iterations):
        build_oracle(qc)
        build_diffuser(qc)

    # undo the FLAG prep (not strictly necessary before measurement, but
    # keeps the circuit tidy / the ancilla+flag register verifiably clean)
    qc.h(FLAG)
    qc.x(FLAG)

    qc.measure(COLOR, list(range(NUM_COLOR_QUBITS)))
    return qc


def run():
    optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M_MARKED)))

    qc = build_grover_circuit(optimal_iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical register bit ordering is little-endian in the
    # returned bitstring (rightmost char = qubit index 0 = vertex 0). Convert
    # each measured string to "vertex 0,1,2,3" left-to-right order so it can
    # be compared directly against CLASSICAL_VALID_STRINGS.
    def to_vertex_order(bitstring):
        # bitstring has NUM_COLOR_QUBITS chars, qiskit-order: c[-1] is qubit0
        rev = bitstring[::-1]  # now rev[i] = qubit i = vertex i
        return rev

    vertex_order_counts = {}
    for bitstring, cnt in counts.items():
        vo = to_vertex_order(bitstring)
        vertex_order_counts[vo] = vertex_order_counts.get(vo, 0) + cnt

    # take the M_MARKED most frequent outcomes
    top_outcomes = sorted(vertex_order_counts.items(), key=lambda kv: -kv[1])[:M_MARKED]
    top_strings = {s for s, _ in top_outcomes}

    total_marked_mass = sum(
        cnt for s, cnt in vertex_order_counts.items() if s in CLASSICAL_VALID_STRINGS
    )
    marked_fraction = total_marked_mass / shots

    print("Erdos problem #922 - quantum-testable sequence check")
    print(f"  OEIS ids used: {OEIS_IDS_USED} (none exist for this problem)")
    print(f"  Property tested: proper 2-colorings of path graph P4 (edges {EDGES})")
    print(f"  Classical valid colorings (vertex order 0,1,2,3): {sorted(CLASSICAL_VALID_STRINGS)}")
    print(f"  Grover iterations used: {optimal_iterations}")
    print(f"  Top {M_MARKED} measured outcomes (vertex order): {top_outcomes}")
    print(f"  Fraction of shots landing on a classically-valid coloring: {marked_fraction:.4f}")

    verified = top_strings == CLASSICAL_VALID_STRINGS and marked_fraction > 0.9

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = run()
    if not ok:
        raise SystemExit(1)
