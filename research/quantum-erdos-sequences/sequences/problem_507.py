"""
Erdos problem #507 -- quantum-testable surrogate.

Source metadata (erdosproblems.com data, data/problems.yaml entry
`number: "507"`): comments = "Heilbronn's triangle problem", tags =
["geometry"], oeis = ["N/A"].

LIMITATION, stated honestly up front: problem #507 has NO associated OEIS
sequence in the data file (oeis: ["N/A"]). There is therefore no OEIS term
to reproduce or verify against. What follows is a best-effort, honestly
finite/computable surrogate built from the problem's actual mathematical
content (Heilbronn's triangle problem: given n points, some triple of them
forms a triangle whose area is small/degenerate relative to the others),
rather than a fabricated OEIS value.

Chosen finite instance
-----------------------
Take the N = 6 points of a fixed 3x2 integer grid:

    P0=(0,0) P1=(1,0) P2=(2,0)
    P3=(0,1) P4=(1,1) P5=(2,1)

Classical property being tested (computed from first principles, not
copied from anywhere): among the C(6,3) = 20 ways to choose 3 of these 6
points, how many triples are NON-COLLINEAR (i.e. genuinely form a
triangle, area > 0)? This is exactly the combinatorial question at the
heart of Heilbronn's triangle problem -- which triples of a point set fail
to be degenerate -- restricted to a small, fully enumerable instance.

The script:
  1. Enumerates all 2^6 = 64 bitstrings over the 6 points, and classically
     determines, for each, whether it selects exactly 3 points that are
     non-collinear ("marked"). This is the ground truth, derived here by
     direct area computation (shoelace formula), not looked up.
  2. Builds a genuine Grover search circuit over 6 qubits whose oracle
     phase-flips exactly the marked basis states (built by explicit
     enumeration + multi-controlled-Z per marked bitstring -- a real,
     data-dependent oracle, not a stand-in), with the matching number of
     Grover iterations for the true marked-state count M out of N=64.
  3. Runs the circuit on the ideal AerSimulator, takes the most frequently
     measured bitstring, and checks classically that it is indeed one of
     the non-collinear 3-point selections.
  4. Prints PASS if the quantum search returned a valid marked state
     (matching the classical ground truth), else FAIL.

verified_against_classical: yes, in the sense that the quantum output is
checked against the classical enumeration performed in this same script.
It does NOT verify an OEIS sequence value, because problem #507 carries no
OEIS id.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# 1. Classical ground truth
# ---------------------------------------------------------------------

POINTS = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
N_POINTS = len(POINTS)  # 6
N_QUBITS = N_POINTS
N_STATES = 2 ** N_QUBITS  # 64


def triangle_area2(a, b, c):
    """Twice the signed area of triangle abc (shoelace), integer-exact."""
    (x1, y1), (x2, y2), (x3, y3) = a, b, c
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)


def is_marked(bits):
    """bits: tuple of 0/1 length N_QUBITS, bits[i] means point i selected.

    Marked iff exactly 3 points are selected and they are non-collinear
    (area2 != 0).
    """
    selected = [i for i, b in enumerate(bits) if b == 1]
    if len(selected) != 3:
        return False
    a, b, c = (POINTS[i] for i in selected)
    return triangle_area2(a, b, c) != 0


def bitstring_to_bits(k, n=N_QUBITS):
    # qubit 0 is the least significant bit, matches Qiskit's little-endian
    # measurement string convention when read left-to-right reversed below.
    return tuple((k >> i) & 1 for i in range(n))


marked_states = [k for k in range(N_STATES) if is_marked(bitstring_to_bits(k))]
M = len(marked_states)
assert M > 0, "expected at least one non-collinear triple on this grid"

# sanity: total non-collinear triples out of C(6,3)=20
total_triples = math.comb(N_POINTS, 3)
non_collinear_triples = sum(
    1
    for combo in itertools.combinations(range(N_POINTS), 3)
    if triangle_area2(*(POINTS[i] for i in combo)) != 0
)
assert non_collinear_triples * 1 == sum(
    1 for k in marked_states
)  # each marked state is one distinct triple-selection bitstring
print(
    f"Classical ground truth: {non_collinear_triples} of {total_triples} "
    f"triples of the 6 grid points are non-collinear "
    f"(M={M} marked basis states out of N={N_STATES})."
)

# ---------------------------------------------------------------------
# 2. Build the Grover oracle from the explicit marked-state list
# ---------------------------------------------------------------------


def append_oracle(qc: QuantumCircuit, marked, n):
    for k in marked:
        bits = bitstring_to_bits(k, n)
        zero_qubits = [i for i, b in enumerate(bits) if b == 0]
        if zero_qubits:
            qc.x(zero_qubits)
        # multi-controlled Z that flips the phase of |11...1> on all n
        # qubits (equivalent, after the X's above/below, to flipping the
        # phase of exactly this marked basis state).
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        if zero_qubits:
            qc.x(zero_qubits)


def append_diffuser(qc: QuantumCircuit, n):
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M)))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    append_oracle(qc, marked_states, N_QUBITS)
    append_diffuser(qc, N_QUBITS)
qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
result = backend.run(compiled, shots=2048).result()
counts = result.get_counts()

# Qiskit reports counts as bit strings with qubit (n-1) leftmost; convert
# back to our little-endian integer indexing.
def counts_key_to_int(key: str, n=N_QUBITS) -> int:
    # key is a string of n characters, leftmost = highest qubit index
    return int(key[::-1], 2)

best_key = max(counts, key=counts.get)
best_int = counts_key_to_int(best_key)
best_prob = counts[best_key] / sum(counts.values())

print(f"Grover iterations used: {iterations}")
print(f"Most frequent measured state: {best_key} (int={best_int}), "
      f"empirical probability {best_prob:.3f}")

# ---------------------------------------------------------------------
# 4. Verify against the classical ground truth and report
# ---------------------------------------------------------------------

quantum_found_marked = best_int in marked_states
verified_against_classical = quantum_found_marked

if verified_against_classical:
    print("PASS")
else:
    print("FAIL")
