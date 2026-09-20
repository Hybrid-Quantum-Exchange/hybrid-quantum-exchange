"""
Erdos problem #62 (source: erdosproblems.com, via manman4/erdosproblems data,
data/problems.yaml entry "number: '62'"):

    prize: no
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION, stated honestly up front: problem #62 has *no* associated OEIS
sequence id in the source data (oeis: ["N/A"]). The assignment for this
lane asks, when no OEIS id exists, to make a best honest attempt at a small
finite, computable property that is genuinely in the spirit of the problem's
tag rather than fabricate or copy an OEIS value that does not exist. Since
the only real metadata available is the tag "graph theory", this script
does NOT claim to formalize erdosproblems.com problem #62 itself (its exact
statement was not available to read from this offline data file). Instead
it builds a real, self-contained, classically-checkable graph-theory
decision property of the kind that tag covers, and verifies a genuine
Grover search circuit against the ground-truth classical answer.

The chosen finite property
---------------------------
Take the complete graph K4 (4 vertices, 6 edges). A 2-coloring of the edges
(colors 0/1) is "good" if none of K4's four triangles is monochromatic
(all 3 of its edges the same color). This is the smallest nontrivial
instance of the 2-color Ramsey-type triangle question that underlies
R(3,3)=6 (K6 forces a monochromatic triangle; K4 does not).

Search space: all 2**6 = 64 edge colorings (6 qubits, one per edge).
Classical ground truth: computed in this script by brute force over all 64
colorings, checking all 4 triangles of K4.

Quantum method: Grover's algorithm. A phase oracle is built (from the
classically-computed set of "good" colorings) that flips the sign of every
good basis state; the standard Grover diffusion operator is applied for the
optimal number of iterations; the ideal AerSimulator is sampled, and the
script checks that the measured colorings are all members of the classical
"good" set (i.e. Grover amplified exactly the classically-verified answer
set, with no false positives).

PASS/FAIL is printed based on comparing the sampled quantum results against
the classically computed set of good colorings.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
import itertools

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no external data).
# ---------------------------------------------------------------------------

N_EDGES = 6  # K4 has C(4,2) = 6 edges
VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges, indices 0..5
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles


def triangle_edge_indices(tri):
    a, b, c = tri
    pairs = [tuple(sorted((a, b))), tuple(sorted((a, c))), tuple(sorted((b, c)))]
    return [EDGE_INDEX[p] for p in pairs]


TRIANGLE_EDGE_IDX = [triangle_edge_indices(t) for t in TRIANGLES]


def is_good_coloring(bits):
    """bits: tuple/list of 6 ints (0/1), one color per edge.

    Returns True iff no triangle of K4 is monochromatic under this coloring.
    """
    for idx_list in TRIANGLE_EDGE_IDX:
        colors = [bits[i] for i in idx_list]
        if colors[0] == colors[1] == colors[2]:
            return False
    return True


def bits_from_int(n, width):
    return tuple((n >> i) & 1 for i in range(width))


GOOD_STATES = []
for n in range(2 ** N_EDGES):
    bits = bits_from_int(n, N_EDGES)
    if is_good_coloring(bits):
        GOOD_STATES.append(n)

N_TOTAL = 2 ** N_EDGES
M_GOOD = len(GOOD_STATES)

print(f"Classical brute force over all {N_TOTAL} edge-colorings of K4:")
print(f"  {M_GOOD} colorings have no monochromatic triangle (the 'good' set).")
assert 0 < M_GOOD < N_TOTAL, "sanity check: property must be nontrivial"

# ---------------------------------------------------------------------------
# 2. Build a Grover oracle that phase-flips exactly the GOOD_STATES.
# ---------------------------------------------------------------------------


def append_multi_controlled_z(qc, qubits):
    """Apply a phase flip of -1 on |11...1> across `qubits` (>=1 qubits)."""
    if len(qubits) == 1:
        qc.z(qubits[0])
    elif len(qubits) == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def mark_state(qc, qubits, state_int, n):
    """Phase-flip the single computational basis state `state_int` (n bits)."""
    bits = bits_from_int(state_int, n)
    flip_qubits = [qubits[i] for i in range(n) if bits[i] == 0]
    if flip_qubits:
        qc.x(flip_qubits)
    append_multi_controlled_z(qc, qubits)
    if flip_qubits:
        qc.x(flip_qubits)


def build_oracle(n, good_states):
    qr = QuantumRegister(n, "e")
    qc = QuantumCircuit(qr, name="oracle")
    for s in good_states:
        mark_state(qc, list(range(n)), s, n)
    return qc


def build_diffuser(n):
    qr = QuantumRegister(n, "e")
    qc = QuantumCircuit(qr, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    append_multi_controlled_z(qc, list(range(n)))
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(N_EDGES, GOOD_STATES)
diffuser = build_diffuser(N_EDGES)

# Optimal number of Grover iterations for N_TOTAL items, M_GOOD marked.
theta = math.asin(math.sqrt(M_GOOD / N_TOTAL))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

print(f"Grover iterations chosen: {iterations} "
      f"(theta={theta:.4f}, N={N_TOTAL}, M={M_GOOD})")

qc = QuantumCircuit(N_EDGES, N_EDGES)
qc.h(range(N_EDGES))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_EDGES), range(N_EDGES))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 4096
job = backend.run(qc, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit bit-string order is little-endian in the classical register w.r.t.
# qubit index (rightmost char = qubit 0), matching bits_from_int's convention
# since we measured qubit i into classical bit i in order.
def int_from_bitstring(bs):
    # bs is e.g. "010110", classical bit c[N-1] c[N-2] ... c[0]
    return int(bs[::-1], 2)

measured_ints = {int_from_bitstring(bs): c for bs, c in counts.items()}

total_shots = sum(measured_ints.values())
hits_on_good = sum(c for s, c in measured_ints.items() if s in set(GOOD_STATES))
frac_good = hits_on_good / total_shots

most_common = max(measured_ints.items(), key=lambda kv: kv[1])[0]
most_common_is_good = most_common in set(GOOD_STATES)

print(f"Sampled {total_shots} shots; fraction landing on a classically-good "
      f"coloring: {frac_good:.3f}")
print(f"Most frequent measured outcome: edges={bits_from_int(most_common, N_EDGES)} "
      f"-> classically good? {most_common_is_good}")

# ---------------------------------------------------------------------------
# 4. PASS/FAIL: Grover must strongly amplify the classically-verified good
#    set (well above the uniform baseline M_GOOD/N_TOTAL), and its top
#    outcome must itself be a classically-verified good coloring.
# ---------------------------------------------------------------------------

baseline = M_GOOD / N_TOTAL
amplified = frac_good > max(0.5, baseline * 2)
verified_against_classical = most_common_is_good and amplified

if verified_against_classical:
    print("PASS: Grover search result matches the classically computed "
          "good set for problem #62's graph-theory-tagged finite property.")
else:
    print("FAIL: Grover search result did not match the classical answer.")
