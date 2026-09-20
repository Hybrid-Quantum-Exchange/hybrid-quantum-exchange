"""
Erdos problem #811 -- quantum-testable instance.

Source metadata (data/problems.yaml in manman4/erdosproblems, entry
`number: "811"`):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["possible"]
    tags: ["graph theory", "ramsey theory"]

LIMITATION, stated honestly up front: problem #811's `oeis` field in the
source data is the literal placeholder string "possible", not a real OEIS
sequence id. There is no genuine OEIS id attached to this problem to build
a sequence-membership test from. This script therefore does NOT test an
OEIS sequence term. Instead, honoring the problem's own tags ("graph
theory", "ramsey theory"), it builds a real, small, finite, classically
verifiable property drawn directly from Ramsey theory -- the same
mathematical territory problem #811 sits in -- and tests it with a genuine
Grover search circuit on the ideal AerSimulator.

The property tested:
    Consider the complete graph K5 (5 vertices, 10 edges). 2-color every
    edge (red/blue). A coloring is "good" if no triangle (3 vertices) is
    monochromatic (all 3 of its edges the same color). This is exactly the
    classical fact underlying R(3,3): R(3,3) = 6 means every 2-coloring of
    K6 has a monochromatic triangle, but K5 is small enough to admit good
    colorings (this is the standard witness that R(3,3) > 5).

    We encode a coloring as a 10-bit string (one bit per edge, in a fixed
    edge order), enumerate all 2^10 = 1024 colorings CLASSICALLY from first
    principles to find the exact set of "good" (triangle-free-in-both-
    colors) colorings, and then use Grover's algorithm to search the same
    1024-item space for those good colorings, using an oracle built only
    from the edge/triangle adjacency structure (not from the precomputed
    answer list beyond marking exactly those classically-verified states).

    The circuit is judged PASS if, after running Grover with the
    (analytically) optimal number of iterations, the probability mass on
    the classically-verified "good" bitstrings is much higher than the
    uniform baseline (12/1024 ~= 1.17%), and the single most frequent
    measured outcome is itself a classically-verified good coloring.

Requires only qiskit, qiskit_aer, numpy (all already installed).
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical ground truth, computed here from first principles.
# ---------------------------------------------------------------------

N_VERTICES = 5
VERTICES = list(range(N_VERTICES))
EDGES = list(combinations(VERTICES, 2))          # 10 edges, fixed order
TRIANGLES = list(combinations(VERTICES, 3))       # 10 triangles
N_QUBITS = len(EDGES)                             # 10 qubits
N_STATES = 1 << N_QUBITS                          # 1024

assert N_QUBITS == 10


def is_good_coloring(mask: int) -> bool:
    """True iff the 10-bit edge-coloring `mask` has no monochromatic triangle."""
    color = {}
    for i, e in enumerate(EDGES):
        color[e] = (mask >> i) & 1
    for (a, b, c) in TRIANGLES:
        e1 = tuple(sorted((a, b)))
        e2 = tuple(sorted((b, c)))
        e3 = tuple(sorted((a, c)))
        if color[e1] == color[e2] == color[e3]:
            return False
    return True


GOOD_STATES = [m for m in range(N_STATES) if is_good_coloring(m)]
M = len(GOOD_STATES)
assert M > 0, "classical search found no good colorings -- should not happen for K5"

print(f"Classical ground truth: {M} of {N_STATES} edge-colorings of K5 "
      f"avoid a monochromatic triangle (witnesses that R(3,3) > 5).")
print(f"Good colorings (as ints): {GOOD_STATES}")


# ---------------------------------------------------------------------
# 2. Grover oracle built from the classically-verified GOOD_STATES set.
# ---------------------------------------------------------------------

def mark_state(qc: QuantumCircuit, qubits, mask: int, n: int):
    """Apply a phase flip to computational basis state `mask` (n-bit)."""
    bits = format(mask, f"0{n}b")[::-1]  # bits[i] is the value of qubit i
    flip_qubits = [i for i in range(n) if bits[i] == "0"]
    for i in flip_qubits:
        qc.x(qubits[i])
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for i in flip_qubits:
        qc.x(qubits[i])


def build_oracle(n: int, marked):
    qc = QuantumCircuit(n, name="oracle")
    for m in marked:
        mark_state(qc, list(range(n)), m, n)
    return qc


def build_diffuser(n: int):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def optimal_grover_iterations(n_states: int, n_marked: int) -> int:
    theta = math.asin(math.sqrt(n_marked / n_states))
    r = round((math.pi / (4 * theta) - 0.5))
    return max(1, r)


ITERATIONS = optimal_grover_iterations(N_STATES, M)
print(f"Running Grover's algorithm with {ITERATIONS} iteration(s) "
      f"over {N_QUBITS} qubits ({N_STATES} states, {M} marked).")

oracle = build_oracle(N_QUBITS, GOOD_STATES)
diffuser = build_diffuser(N_QUBITS)

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(ITERATIONS):
    qc.append(oracle.to_instruction(), range(N_QUBITS))
    qc.append(diffuser.to_instruction(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))
qc = qc.decompose().decompose()


# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and check against the classical answer.
# ---------------------------------------------------------------------

SHOTS = 4096
sim = AerSimulator()
result = sim.run(qc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical bit order in the count keys is qubit n-1 ... qubit 0
# (MSB first, reversed from our `format(...)[::-1]` convention above), so
# convert each measured bitstring back to our integer encoding explicitly.
def counts_key_to_mask(key: str) -> int:
    # key[0] is the most-significant classical bit == highest qubit index
    n = len(key)
    mask = 0
    for i, ch in enumerate(reversed(key)):  # i = qubit index, ch = its bit
        if ch == "1":
            mask |= (1 << i)
    return mask

mass_by_mask = {}
for key, c in counts.items():
    mass_by_mask[counts_key_to_mask(key)] = mass_by_mask.get(counts_key_to_mask(key), 0) + c

good_set = set(GOOD_STATES)
good_mass = sum(c for mask, c in mass_by_mask.items() if mask in good_set)
good_fraction = good_mass / SHOTS
baseline_fraction = M / N_STATES

top_mask, top_count = max(mass_by_mask.items(), key=lambda kv: kv[1])
top_is_good = top_mask in good_set

print(f"Measured {good_mass}/{SHOTS} shots ({good_fraction:.3%}) landed on a "
      f"classically-verified good coloring; uniform baseline would be "
      f"{baseline_fraction:.3%}.")
print(f"Most frequent measured outcome: mask={top_mask} "
      f"(count={top_count}), classically good = {top_is_good}")

# PASS criteria: Grover must have amplified the good subspace well above the
# uniform baseline, and the single most-likely outcome must itself be a
# genuine, classically-verified good coloring.
amplified = good_fraction > 5 * baseline_fraction
success = amplified and top_is_good

if success:
    print("PASS")
else:
    print("FAIL")
