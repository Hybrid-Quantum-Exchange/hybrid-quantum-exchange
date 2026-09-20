"""
Quantum-testable sequence entry for Erdos problem #566.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 566"):
    tags: ["graph theory", "ramsey theory"]
    oeis: ["N/A"]
    informal_status: open

LIMITATION (read before trusting "PASS" as evidence about problem 566 itself):
Problem #566 has no associated OEIS sequence id in the source data (oeis is
literally "N/A"), so there is no integer sequence to build a membership /
term-defining quantum test around, as the task otherwise calls for. Rather
than fabricate an OEIS value, this script falls back to the problem's own
tags ("graph theory", "ramsey theory") and tests a small, genuinely finite,
classically-checkable Ramsey-theory fact that is representative of the kind
of statement problem 566's area concerns -- it is NOT a derivation of, or
proof related to, problem 566's actual (open) statement. Treat this as an
honest best-effort quantum-testable stand-in, not a solution to problem 566.

The classical property tested:
    R(3,3) = 6, so K4 (only 4 vertices) admits at least one 2-coloring of its
    edges with no monochromatic triangle. K4 has C(4,2) = 6 edges and
    C(4,3) = 4 triangles. Enumerating all 2^6 = 64 edge-colorings classically
    (in this script, from first principles, no external data) finds the exact
    set of "good" colorings (no monochromatic triangle). This is a small,
    finite, computable search problem: exactly the shape Grover's algorithm
    is built for.

Quantum approach:
    Grover search over 6 qubits (one per edge of K4, N = 64 basis states).
    The oracle is a diagonal phase-flip built directly from the classically
    computed set of "good" colorings (marked states), applied via
    multi-controlled-Z gates (X-conjugated per bitstring) -- a legitimate
    Grover oracle for a target set specified as an explicit list of marked
    computational basis states. Number of marked states M is computed
    classically first, which fixes the (classically well known) optimal
    number of Grover iterations ~ floor(pi/4 * sqrt(N/M)).

    Run on the ideal AerSimulator (statevector + sampling), then compare the
    quantum result (the most frequently measured bitstring, and the total
    probability mass landing on the classically-verified "good" colorings)
    against the classical answer. PASS if the top measured outcome is a true
    "no monochromatic triangle" coloring and the marked-state probability
    mass is enriched far above the uniform 18/64 baseline.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

from itertools import combinations, product
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = list(combinations(range(N_VERTICES), 2))  # 6 edges of K4
TRIANGLES = list(combinations(range(N_VERTICES), 3))  # 4 triangles of K4
N_EDGES = len(EDGES)
assert N_EDGES == 6
assert len(TRIANGLES) == 4

EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}


def triangle_edge_indices(triangle):
    a, b, c = triangle
    return [
        EDGE_INDEX[tuple(sorted((a, b)))],
        EDGE_INDEX[tuple(sorted((b, c)))],
        EDGE_INDEX[tuple(sorted((a, c)))],
    ]


TRIANGLE_EDGE_IDXS = [triangle_edge_indices(t) for t in TRIANGLES]


def has_monochromatic_triangle(coloring_bits):
    """coloring_bits: tuple of 6 bits (0/1), one per edge in EDGES order."""
    for idxs in TRIANGLE_EDGE_IDXS:
        vals = [coloring_bits[i] for i in idxs]
        if vals[0] == vals[1] == vals[2]:
            return True
    return False


def bits_to_qiskit_bitstring(bits):
    """Qiskit orders classical bitstrings with qubit 0 as the rightmost char."""
    return "".join(str(b) for b in reversed(bits))


ALL_COLORINGS = list(product([0, 1], repeat=N_EDGES))
GOOD_COLORINGS = [c for c in ALL_COLORINGS if not has_monochromatic_triangle(c)]

CLASSICAL_NUM_GOOD = len(GOOD_COLORINGS)
CLASSICAL_TOTAL = len(ALL_COLORINGS)

print(f"Classical brute force over K4 edge-colorings: "
      f"{CLASSICAL_NUM_GOOD} / {CLASSICAL_TOTAL} colorings avoid a "
      f"monochromatic triangle (R(3,3)=6 predicts this set is nonempty).")
assert CLASSICAL_NUM_GOOD > 0, "R(3,3)=6 guarantees K4 has such a coloring"

GOOD_BITSTRINGS = {bits_to_qiskit_bitstring(c) for c in GOOD_COLORINGS}


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the classically computed GOOD_COLORINGS.
# ---------------------------------------------------------------------------

def append_marking_for_state(qc, bits, n):
    """Flip the phase of the single basis state `bits` (tuple of 0/1, length n,
    bits[i] corresponds to qubit i) using X-conjugated multi-controlled-Z."""
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)


def build_oracle(n, marked_states):
    qc = QuantumCircuit(n, name="oracle")
    for bits in marked_states:
        append_marking_for_state(qc, bits, n)
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


N_QUBITS = N_EDGES  # 6
oracle = build_oracle(N_QUBITS, GOOD_COLORINGS)
diffuser = build_diffuser(N_QUBITS)

M = CLASSICAL_NUM_GOOD
N = 2 ** N_QUBITS
n_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover: N={N} states, M={M} marked, using {n_iterations} iteration(s).")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(n_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 8192
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

top_bitstring, top_count = max(counts.items(), key=lambda kv: kv[1])
marked_mass = sum(c for bs, c in counts.items() if bs in GOOD_BITSTRINGS) / shots
uniform_baseline = M / N

print(f"Most frequent measured outcome: {top_bitstring} "
      f"({top_count}/{shots} shots)")
print(f"Probability mass on classically-verified good colorings: "
      f"{marked_mass:.3f} (uniform baseline would be {uniform_baseline:.3f})")

top_is_good = top_bitstring in GOOD_BITSTRINGS
enriched = marked_mass > 3 * uniform_baseline  # Grover should strongly amplify

verified_against_classical = top_is_good and enriched

if verified_against_classical:
    print("PASS")
else:
    print("FAIL")
