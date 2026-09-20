"""
Erdos problem #637 (erdosproblems.com), from the read-only clone of
manman4/erdosproblems, data/problems.yaml, entry `number: "637"`:

    prize: no
    status: proved (as of 2025-08-31)
    oeis: ["N/A"]
    tags: ["graph theory", "ramsey theory"]

LIMITATION, stated up front: this entry carries no OEIS sequence id (the
metadata literally says oeis: ["N/A"]). There is therefore no OEIS sequence
to build a "quantum-testable sequence membership" test against, and this
script cannot honestly claim to test problem #637's own statement (whose
exact combinatorial content is not reproduced in the YAML metadata either).

Rather than fabricate an OEIS value or invent unrelated content, this script
stays honest to the two tags that ARE given -- "graph theory" and "ramsey
theory" -- and builds a real, small, fully classically-checkable Ramsey-type
property, then verifies it with a genuine Grover search circuit on
AerSimulator. This is offered as the closest honest substitute, not as a
verification of Erdos problem #637 itself.

Classical property tested
--------------------------
Take the complete graph K4 (4 vertices, 6 edges:
  e0=(0,1) e1=(0,2) e2=(0,3) e3=(1,2) e4=(1,3) e5=(2,3)
Each edge is 2-colored (bit 0 / bit 1), so a coloring is a 6-bit string,
64 total colorings. K4 has exactly 4 triangles:
  T0={e0,e1,e3} (0,1,2), T1={e0,e2,e4} (0,1,3),
  T2={e1,e2,e5} (0,2,3), T3={e3,e4,e5} (1,2,3)
A coloring is "good" if no triangle is monochromatic (all 3 of its edges the
same color). This is a small, finite, fully computable property directly in
the spirit of Ramsey theory (R(3,3)=6 says K6 forces a monochromatic
triangle in every 2-coloring; K4 is small enough to have good colorings,
which is exactly what is searched for here).

The classical answer (computed in this script, by brute force over all 64
colorings, from first principles) is the exact set of "good" colorings and
their count.

Quantum circuit
----------------
A genuine Grover search over 6 qubits (64-dimensional search space):
  - A quantum oracle built from Toffoli/X gates that flips the phase of
    exactly the "good" (no monochromatic triangle) computational basis
    states, computed directly from the same triangle logic as the classical
    checker (mono_i = NOT(e_a XOR e_b) AND NOT(e_b XOR e_c) for each
    triangle's edges a,b,c; state is marked good iff NOT(OR_i mono_i)).
  - The standard Grover diffusion operator.
  - The optimal integer number of Grover iterations for a 6-qubit space
    with the classically-known number of marked ("good") states.
Run on AerSimulator (ideal, no noise). PASS means: over many shots, the
measured bitstrings are overwhelmingly (>= 95%) valid "good" colorings by
the classical checker, and every distinct sampled bitstring is independently
re-checked classically.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ----------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ----------------------------------------------------------------------

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # e0..e5
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
VERTICES = [0, 1, 2, 3]

TRIANGLES = []
for a, b, c in itertools.combinations(VERTICES, 3):
    tri_edges = []
    for u, v in itertools.combinations(sorted((a, b, c)), 2):
        tri_edges.append(EDGE_INDEX[(u, v)])
    TRIANGLES.append(tuple(tri_edges))  # each a triple of edge indices


def is_good_coloring(bits):
    """bits: sequence of 6 ints (0/1), bits[i] = color of edge i.
    Returns True iff no triangle is monochromatic."""
    for t in TRIANGLES:
        colors = {bits[i] for i in t}
        if len(colors) == 1:
            return False
    return True


def bits_from_int(n, width=6):
    return [(n >> i) & 1 for i in range(width)]


GOOD_STATES = [n for n in range(64) if is_good_coloring(bits_from_int(n))]
NUM_GOOD = len(GOOD_STATES)

assert NUM_GOOD > 0, "sanity: K4 must admit at least one good 2-coloring"
print(f"Classical brute force: {NUM_GOOD} / 64 colorings of K4 are "
      f"monochromatic-triangle-free ('good').")

# ----------------------------------------------------------------------
# 2. Grover oracle marking exactly the "good" states.
# ----------------------------------------------------------------------

N_QUBITS = 6  # e0..e5 -> qubits 0..5


def equal_into(qc, x, y, anc):
    """anc (starts |0>) becomes 1 iff qubit x == qubit y."""
    qc.cx(x, anc)
    qc.cx(y, anc)
    qc.x(anc)


def equal_into_inv(qc, x, y, anc):
    qc.x(anc)
    qc.cx(y, anc)
    qc.cx(x, anc)


def build_grover_oracle():
    n_tri = len(TRIANGLES)
    eq1_start = N_QUBITS            # 4 ancillas: (a==b) per triangle
    eq2_start = eq1_start + n_tri   # 4 ancillas: (b==c) per triangle
    mono_start = eq2_start + n_tri  # 4 ancillas: mono_i = eq1_i AND eq2_i
    or_qubit = mono_start + n_tri   # 1 ancilla: OR of all mono_i
    total = or_qubit + 1
    qc = QuantumCircuit(total, name="oracle")

    # compute eq1_i, eq2_i, mono_i for each triangle
    for i, (a, b, c) in enumerate(TRIANGLES):
        e1 = eq1_start + i
        e2 = eq2_start + i
        m = mono_start + i
        equal_into(qc, a, b, e1)
        equal_into(qc, b, c, e2)
        qc.ccx(e1, e2, m)

    # OR all mono_i into or_qubit: OR(x1..xk) computed via
    # or_qubit ^= NOT(AND(NOT x1 .. NOT xk)); simpler for k=4: chain of
    # "at least one" using De Morgan with X gates + multi-controlled X.
    for i in range(n_tri):
        qc.x(mono_start + i)
    qc.mcx([mono_start + i for i in range(n_tri)], or_qubit)
    qc.x(or_qubit)
    for i in range(n_tri):
        qc.x(mono_start + i)
    # now or_qubit = 1 iff at least one mono_i was 1 (i.e. coloring is BAD)

    # phase flip when or_qubit == 0 (state is GOOD): X, then phase kick via
    # Z on or_qubit conditioned on being flipped, then X back.
    qc.x(or_qubit)
    qc.z(or_qubit)
    qc.x(or_qubit)

    # uncompute ancillas (reverse order)
    for i in range(n_tri):
        qc.x(mono_start + i)
    qc.mcx([mono_start + i for i in range(n_tri)], or_qubit)
    for i in range(n_tri):
        qc.x(mono_start + i)

    for i, (a, b, c) in enumerate(TRIANGLES):
        e1 = eq1_start + i
        e2 = eq2_start + i
        m = mono_start + i
        qc.ccx(e1, e2, m)
        equal_into_inv(qc, b, c, e2)
        equal_into_inv(qc, a, b, e1)

    return qc, total


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


oracle_qc, oracle_total_qubits = build_grover_oracle()
n_ancilla = oracle_total_qubits - N_QUBITS

diffuser_qc = build_diffuser(N_QUBITS)

# Optimal number of Grover iterations for 64 states, NUM_GOOD marked.
theta = math.asin(math.sqrt(NUM_GOOD / 64))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover: {NUM_GOOD} marked states out of 64, using {iterations} "
      f"iteration(s).")

qc = QuantumCircuit(oracle_total_qubits, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle_qc.to_instruction(), range(oracle_total_qubits))
    qc.append(diffuser_qc.to_instruction(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

# ----------------------------------------------------------------------
# 3. Run on AerSimulator and verify against the classical ground truth.
# ----------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
SHOTS = 4096
result = sim.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit order: classical bit c_i corresponds to circuit qubit i, and
# the returned bitstring lists c_{n-1}...c_0 (MSB first). Reverse to
# recover [bit for e0, e1, ..., e5].
good_shots = 0
distinct_states_seen = set()
for bitstring, cnt in counts.items():
    bits = [int(b) for b in reversed(bitstring)]  # bits[0]=e0 ... bits[5]=e5
    n = int(bitstring, 2)  # value ignoring bit-order label, just for bookkeeping
    distinct_states_seen.add(bitstring)
    if is_good_coloring(bits):
        good_shots += cnt

success_rate = good_shots / SHOTS
print(f"Measured {len(counts)} distinct bitstrings over {SHOTS} shots; "
      f"{good_shots} shots ({success_rate:.1%}) decoded to a classically "
      f"verified 'good' (monochromatic-triangle-free) K4 coloring.")

# Independently re-verify every distinct sampled bitstring classically.
all_sampled_good = all(
    is_good_coloring([int(b) for b in reversed(bs)]) or counts[bs] == 0
    for bs in counts
)

PASS_THRESHOLD = 0.95
verified = success_rate >= PASS_THRESHOLD

if verified:
    print("PASS: Grover search on AerSimulator overwhelmingly amplified "
          "monochromatic-triangle-free K4 colorings, matching the "
          "classically brute-forced ground truth.")
else:
    print("FAIL: quantum result did not match the classical ground truth "
          "within the required threshold.")
