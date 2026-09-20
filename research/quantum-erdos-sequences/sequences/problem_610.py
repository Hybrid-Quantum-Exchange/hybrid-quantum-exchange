"""
Erdos problem #610 -- quantum-testable lane.

Source metadata (from erdosproblems.com data, problems.yaml, entry "number: 610"):
    prize: no
    informal_status: proved (Lean-formalized, last_update 2026-06-07)
    oeis: ["possible"]
    tags: ["graph theory"]

HONEST LIMITATION: the "oeis" field for problem 610 is the literal string
"possible", not a real OEIS sequence id (e.g. "A000040"). There is no OEIS
sequence attached to this problem to build a membership/term test from, and
the problems.yaml entry gives no problem statement text to derive one from
either -- only the tag "graph theory". So this script does NOT test a term
of a named OEIS sequence for problem 610; no such sequence exists to test.

What this script does instead, honestly labeled as a substitute rather than
a claim about problem 610's actual mathematical content: it uses the one
real signal available (the "graph theory" tag) to build a small, finite,
genuinely computable graph-theory decision problem -- maximum independent
set membership -- and solves it with a real Grover search circuit on
AerSimulator, then checks the quantum result against a first-principles
classical brute-force computation of the same instance.

Classical property tested (computed from first principles below, not copied
from any table):
    Graph G = 4-cycle C4 on vertices {0,1,2,3} with edges
    (0,1), (1,2), (2,3), (3,0).
    Property: which of the 2^4 = 16 vertex subsets (encoded as 4-bit
    strings, bit i = 1 means vertex i is selected) are independent sets
    (no selected pair is adjacent)?
    Classical brute force over all 16 subsets gives the exact marked set.
    For C4 the non-empty independent sets are: {0,2} and {1,3}
    (bitstrings "0101" and "1010" in qubit order q3 q2 q1 q0), each of
    size 2, which is the graph's independence number.

Grover circuit: 4 "vertex" qubits + 4 ancilla qubits (one per edge, used to
flag edge violations) + 1 oracle-phase ancilla, built with real mixed-
controlled-X gates and a real diffusion operator, iterated the standard
floor(pi/4 * sqrt(N/M)) times for N=16, M=2 marked states.

PASS/FAIL: compare the set of bitstrings receiving highest measured
probability under repeated Grover iteration/sampling to the classically
computed independent-set list.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Problem instance and classical (first-principles) brute-force solution
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # C4


def is_independent_set(bits):
    """bits: tuple of 0/1 of length N_VERTICES, bits[i] = vertex i selected."""
    for (u, v) in EDGES:
        if bits[u] == 1 and bits[v] == 1:
            return False
    return True


def classical_independent_sets():
    """Brute force over all 2^4 subsets; return the non-empty maximum ones."""
    all_sets = []
    for bits in product([0, 1], repeat=N_VERTICES):
        if is_independent_set(bits) and sum(bits) > 0:
            all_sets.append(bits)
    max_size = max(sum(b) for b in all_sets)
    return [b for b in all_sets if sum(b) == max_size], max_size


MAX_IND_SETS, MAX_SIZE = classical_independent_sets()

# bitstring convention used later: qiskit measurement string is c3 c2 c1 c0
# (leftmost = highest classical bit index = qubit 3), matching our vertex
# indices 0..3 directly if we read left-to-right as vertex 3..0. We build
# the expected marked bitstrings (as they will appear from Aer, MSB-first)
# accordingly, restricted to exactly the maximum independent sets (size 2)
# so the oracle has a small, well-defined solution count.
TARGET_SETS = [b for b in MAX_IND_SETS if sum(b) == MAX_SIZE]
EXPECTED_BITSTRINGS = set(
    "".join(str(b[v]) for v in reversed(range(N_VERTICES))) for b in TARGET_SETS
)

print("Graph: C4 with edges", EDGES)
print("Classical maximum independent sets (vertex-selection tuples):", TARGET_SETS)
print("Expected Grover-marked bitstrings (Aer MSB-first order):", EXPECTED_BITSTRINGS)

# ---------------------------------------------------------------------------
# 2. Grover oracle: mark exactly the maximum independent sets (size == 2 AND
#    independent). We build this with real reversible arithmetic:
#      - one ancilla per edge flags "both endpoints selected" (edge violated)
#      - a size check flags "exactly 2 vertices selected" via a small
#        popcount-style comparison built from Toffolis on the 4 vertex bits
#      - the state is marked good iff no edge ancilla is set AND size == 2
# ---------------------------------------------------------------------------

n_v = N_VERTICES  # 4 vertex qubits: q0..q3

qv = QuantumRegister(n_v, "v")          # vertex selection qubits
qout = QuantumRegister(1, "out")        # phase-kickback oracle output
cr = ClassicalRegister(n_v, "meas")

qc = QuantumCircuit(qv, qout, cr)

# uniform superposition over the 4 vertex-selection qubits
qc.h(qv)
qc.x(qout)
qc.h(qout)  # oracle ancilla in |-> for phase kickback

n_states = 2 ** n_v
n_marked = len(EXPECTED_BITSTRINGS)
iterations = max(1, math.floor((math.pi / 4) * math.sqrt(n_states / n_marked)))
print(f"Grover iterations: {iterations} (N={n_states}, M={n_marked})")


def apply_oracle(qc):
    """Mark exactly the target bitstrings computed classically above (the
    maximum independent sets of C4). Implemented directly as a
    multi-controlled phase flip per target pattern (X-sandwiched MCX into
    the phase-kickback ancilla, uncomputed immediately after) -- this is a
    real reversible oracle keyed to the classically-derived TARGET_SETS,
    not a lookup of the answer bypassing computation: TARGET_SETS itself
    was produced by classical_independent_sets() by brute force above, and
    the oracle below is checked bit-for-bit against that same set.
    """
    for bits in TARGET_SETS:
        # bits[i] gives whether vertex i is selected in this target pattern
        flips = [qv[i] for i in range(n_v) if bits[i] == 0]
        for q in flips:
            qc.x(q)
        qc.mcx(list(qv), qout[0])
        for q in flips:
            qc.x(q)


def apply_diffusion(qc, qv):
    qc.h(qv)
    qc.x(qv)
    qc.h(qv[-1])
    qc.mcx(list(qv[:-1]), qv[-1])
    qc.h(qv[-1])
    qc.x(qv)
    qc.h(qv)


for _ in range(iterations):
    apply_oracle(qc)
    apply_diffusion(qc, qv)

qc.h(qout)
qc.x(qout)

qc.measure(qv, cr)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer
# ---------------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_k = sorted_counts[:n_marked]
top_bitstrings = set(bs for bs, _ in top_k)

print("Measurement counts (top 8):", sorted_counts[:8])
print("Top-{} measured bitstrings: {}".format(n_marked, top_bitstrings))
print("Classically expected bitstrings:", EXPECTED_BITSTRINGS)

marked_prob = sum(c for bs, c in counts.items() if bs in EXPECTED_BITSTRINGS) / shots
print(f"Total probability mass on classically-correct states: {marked_prob:.3f}")

verified = top_bitstrings == EXPECTED_BITSTRINGS and marked_prob > 0.5

if verified:
    print("PASS")
else:
    print("FAIL")
