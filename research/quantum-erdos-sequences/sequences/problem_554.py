"""
Erdos problem #554 (erdosproblems.com), quantum-testable lane.

Source metadata (data/problems.yaml in the manman4/erdosproblems clone),
verified 2026-09-19:

    - number: "554"
      prize: "no"
      informal_status: {state: "open", last_update: "2025-08-31"}
      formal_status: {state: "unformalized"}
      status: {state: "open", last_update: "2025-08-31"}
      oeis: ["possible"]
      tags: ["graph theory", "ramsey theory"]

HONEST LIMITATION: the "oeis" field for problem 554 is the literal
placeholder string "possible", not a real OEIS sequence id. There is no
associated OEIS sequence to build a quantum test around, and the yaml entry
carries no numeric data or formula beyond the tags "graph theory" /
"ramsey theory". Per the task instructions, this script does not fabricate
an OEIS id or invent a sequence value; instead it builds the closest
genuine, small, finite, computable property that the "ramsey theory" tag
actually supports and that a real quantum circuit can search:

    PROPERTY TESTED: existence of a 2-colouring of the edges of the
    complete graph K5 with no monochromatic triangle.

This is exactly the classical fact underlying the Ramsey number R(3,3) = 6
(Erdos's own area): every 2-colouring of K6 has a monochromatic triangle,
but K5 admits colourings that avoid one (e.g. colour edges by whether the
vertices, placed on a pentagon 0..4, are graph-distance 1 or 2 apart). This
is a small, finite, fully computable search problem (2^10 = 1024
candidate colourings of the 10 edges of K5), well-suited to Grover search,
and is checked classically from first principles in `classical_good_colorings()`
below (brute force over all 1024 colourings, testing all 10 triangles).

CIRCUIT: a genuine Grover search circuit over 10 "edge" qubits (one qubit
per edge of K5, qubit = colour of that edge). The oracle computes, entirely
by controlled-NOT/Toffoli/multi-controlled-X logic (no lookup table, no
classical shortcut baked into the circuit), whether a given 10-bit colouring
contains a monochromatic triangle, for all 10 triangles of K5, and flips the
phase of the state iff none does. The diffuser is the standard Grover
inversion-about-the-mean operator on the 10 edge qubits. The circuit is run
on the ideal AerSimulator (statevector method) with the classically-computed
optimal number of Grover iterations. PASS means the most probable measured
outcomes are triangle-free colourings, i.e. quantum search finds true
positives that agree with the brute-force classical answer.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles (brute force).
# ---------------------------------------------------------------------------

VERTICES = range(5)
EDGES = list(itertools.combinations(VERTICES, 2))          # 10 edges of K5
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))       # 10 triangles


def is_triangle_free_coloring(bits):
    """bits: tuple of 10 ints (0/1), one per edge in EDGES order.
    Returns True iff no triangle of K5 is monochromatic under this colouring."""
    for (a, b, c) in TRIANGLES:
        e1 = bits[EDGE_INDEX[(a, b)]]
        e2 = bits[EDGE_INDEX[(a, c)]]
        e3 = bits[EDGE_INDEX[(b, c)]]
        if e1 == e2 == e3:
            return False
    return True


def classical_good_colorings():
    """Brute-force all 2^10 edge colourings of K5; return the triangle-free ones,
    as bitstrings (edge 0 = least significant qubit, matching the circuit's
    qubit ordering)."""
    good = []
    for bits in itertools.product([0, 1], repeat=10):
        if is_triangle_free_coloring(bits):
            good.append(bits)
    return good


GOOD_COLORINGS = classical_good_colorings()
N_STATES = 2 ** 10
M_GOOD = len(GOOD_COLORINGS)

# Sanity check against the known mathematical fact: exactly 12 of the 1024
# edge-colourings of K5 avoid a monochromatic triangle (this is the standard
# witness that R(3,3) > 5, i.e. R(3,3) = 6).
assert M_GOOD == 12, f"expected 12 triangle-free K5 colourings, got {M_GOOD}"

GOOD_BITSTRINGS = {
    "".join(str(b) for b in reversed(bits)) for bits in GOOD_COLORINGS
}  # qiskit bit-order: rightmost char = qubit 0 = edge 0


# ---------------------------------------------------------------------------
# 2. Grover oracle: a real reversible circuit computing "triangle-free".
# ---------------------------------------------------------------------------

N_EDGES = 10
edges_q = QuantumRegister(N_EDGES, "edge")
xor_q = QuantumRegister(2, "xor")            # reusable scratch pair
bad_q = QuantumRegister(len(TRIANGLES), "bad")  # one flag per triangle
good_q = QuantumRegister(1, "good")          # 1 iff no triangle is monochromatic


def build_oracle():
    qc = QuantumCircuit(edges_q, xor_q, bad_q, good_q, name="triangle_free_oracle")

    # For each triangle (a,b,c), its three edges are monochromatic iff
    # e_ab == e_ac and e_ac == e_bc, i.e. (e_ab XOR e_ac) == 0 and
    # (e_ac XOR e_bc) == 0. Compute those two XORs into scratch qubits,
    # then Toffoli them into bad[t] iff both XORs are 0 -> flip xor
    # qubits with X first so "control on 0" becomes "control on 1".
    for t, (a, b, c) in enumerate(TRIANGLES):
        i_ab = EDGE_INDEX[(a, b)]
        i_ac = EDGE_INDEX[(a, c)]
        i_bc = EDGE_INDEX[(b, c)]

        qc.cx(edges_q[i_ab], xor_q[0])
        qc.cx(edges_q[i_ac], xor_q[0])   # xor_q[0] = e_ab XOR e_ac
        qc.cx(edges_q[i_ac], xor_q[1])
        qc.cx(edges_q[i_bc], xor_q[1])   # xor_q[1] = e_ac XOR e_bc

        qc.x(xor_q[0])
        qc.x(xor_q[1])                   # now both ==1 iff triangle t monochromatic
        qc.ccx(xor_q[0], xor_q[1], bad_q[t])
        qc.x(xor_q[1])
        qc.x(xor_q[0])

        # uncompute the scratch XORs for reuse by the next triangle
        qc.cx(edges_q[i_ac], xor_q[1])
        qc.cx(edges_q[i_bc], xor_q[1])
        qc.cx(edges_q[i_ab], xor_q[0])
        qc.cx(edges_q[i_ac], xor_q[0])

    # good = 1 iff ALL bad[t] == 0: a multi-controlled X on good_q, with
    # every bad[t] control active on the |0> state.
    mcx_all_zero = MCXGate(num_ctrl_qubits=len(TRIANGLES), ctrl_state="0" * len(TRIANGLES))
    qc.append(mcx_all_zero, list(bad_q) + [good_q[0]])

    # phase-flip the marked (triangle-free) states
    qc.z(good_q[0])

    # uncompute good_q and all bad[t] flags (mirror image) to leave ancillas
    # back in |0> so the oracle can be reused unentangled with edges_q.
    qc.append(mcx_all_zero, list(bad_q) + [good_q[0]])

    for t, (a, b, c) in reversed(list(enumerate(TRIANGLES))):
        i_ab = EDGE_INDEX[(a, b)]
        i_ac = EDGE_INDEX[(a, c)]
        i_bc = EDGE_INDEX[(b, c)]

        qc.cx(edges_q[i_ac], xor_q[0])
        qc.cx(edges_q[i_ab], xor_q[0])
        qc.cx(edges_q[i_bc], xor_q[1])
        qc.cx(edges_q[i_ac], xor_q[1])

        qc.x(xor_q[0])
        qc.x(xor_q[1])
        qc.ccx(xor_q[0], xor_q[1], bad_q[t])
        qc.x(xor_q[1])
        qc.x(xor_q[0])

        qc.cx(edges_q[i_ac], xor_q[1])
        qc.cx(edges_q[i_bc], xor_q[1])
        qc.cx(edges_q[i_ab], xor_q[0])
        qc.cx(edges_q[i_ac], xor_q[0])

    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(iterations):
    qc = QuantumCircuit(edges_q, xor_q, bad_q, good_q)
    qc.h(edges_q)

    oracle = build_oracle()
    diffuser = build_diffuser(N_EDGES)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), list(edges_q) + list(xor_q) + list(bad_q) + list(good_q))
        qc.append(diffuser.to_instruction(), edges_q)

    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator, with the classically-optimal iteration count.
# ---------------------------------------------------------------------------

def optimal_grover_iterations(n_states, n_good):
    theta = math.asin(math.sqrt(n_good / n_states))
    return max(1, round((math.pi / (4 * theta)) - 0.5))


def main():
    iterations = optimal_grover_iterations(N_STATES, M_GOOD)
    print(f"K5 triangle-free 2-colourings: {M_GOOD} good out of {N_STATES} "
          f"(classical brute force). Grover iterations: {iterations}.")

    qc = build_grover_circuit(iterations)
    qc = qc.decompose().decompose()

    sim = AerSimulator(method="statevector")
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Only look at the edge-qubit bits (the first N_EDGES classical bits in
    # the measured bitstring, since edges_q was added first and Qiskit
    # concatenates measured registers with the last-added register's bits
    # leftmost). We measured with measure_all(), which appends a classical
    # register matching creg order == qreg order (edges, xor, bad, good),
    # printed most-significant (last qubit) first.
    total_qubits = N_EDGES + 2 + len(TRIANGLES) + 1

    def edge_bits_from_key(key):
        key = key.replace(" ", "")
        assert len(key) == total_qubits
        # rightmost `total_qubits` chars correspond to qubits 0..total_qubits-1
        # in increasing order right-to-left; edge qubits are indices 0..9.
        return key[-N_EDGES:]  # last N_EDGES chars = edge qubits (right-aligned)

    hit_counts = {}
    total_hits = 0
    for key, c in counts.items():
        edge_bits = edge_bits_from_key(key)
        if edge_bits in GOOD_BITSTRINGS:
            hit_counts[edge_bits] = hit_counts.get(edge_bits, 0) + c
            total_hits += c

    success_prob = total_hits / shots
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    print("Top measured outcomes (full bitstring: good|bad|xor|edge):")
    for key, c in top:
        eb = edge_bits_from_key(key)
        print(f"  {key}  count={c}  edge_bits={eb}  triangle_free={eb in GOOD_BITSTRINGS}")

    print(f"Fraction of shots landing on a genuinely triangle-free K5 "
          f"colouring: {success_prob:.3f} (uniform-random baseline would be "
          f"{M_GOOD / N_STATES:.3f}).")

    # Independent classical re-check of a few of the found bitstrings, from
    # first principles (not just membership in the precomputed set).
    for eb in list(hit_counts.keys())[:3]:
        bits = tuple(int(ch) for ch in reversed(eb))
        assert is_triangle_free_coloring(bits), "quantum result failed classical re-verification"

    passed = success_prob > 3 * (M_GOOD / N_STATES) and len(hit_counts) > 0
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
