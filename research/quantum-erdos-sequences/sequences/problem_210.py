"""
Erdos problem #210 (erdosproblems.com/210) -- Sylvester-Gallai / orchard problem:
the minimum number of "ordinary lines" (lines through exactly 2 of n given,
non-collinear points) that any configuration of n points in the plane must
contain. OEIS: A003034 (minimum number of ordinary lines determined by n
points, not all on a line, for n = 3, 4, 5, ...).

Classical property tested here (derived and checked from first principles in
this script, not copied from OEIS):

    For the 4-point configuration A=(0,0), B=(1,0), C=(2,0), D=(0,1)
    (A, B, C collinear on the x-axis; D off that line -- the standard
    "near-pencil" extremal configuration for n=4), the number of ORDINARY
    lines (lines through exactly 2 of the 4 points) is 3, matching
    A003034(4) = 3, the known minimum for n=4 points not all collinear.

    There are C(4,2) = 6 point-pairs / candidate lines total. Grouping pairs
    that lie on the same line (using exact integer cross-product collinearity
    tests, no floating point), exactly one line (A,B,C) is non-ordinary
    (3 points on it) and the other 3 distinct lines (AD, BD, CD) are each
    ordinary (exactly 2 points on them). So among the 6 pairs, the pairs
    {A,B},{B,C},{A,C} all sit on the SAME non-ordinary line, and {A,D},{B,D},
    {C,D} are each their own ordinary line -- 3 ordinary lines total.

Quantum circuit: a Grover search over the 6 pair-indices (encoded in 3
qubits, states 0..5; states 6,7 unused/never marked) that marks exactly the
3 pairs lying on ordinary lines ({A,D},{B,D},{C,D} -> indices 3,4,5 below).
The circuit is run on the ideal AerSimulator; we check that measurement
amplifies exactly the marked (ordinary-line) indices, and that the number of
distinct marked indices found equals the classically-computed count of
ordinary lines, which equals A003034(4) = 3.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth
# ---------------------------------------------------------------------------

POINTS = [(0, 0), (1, 0), (2, 0), (0, 1)]  # A, B, C, D
PAIRS = list(combinations(range(4), 2))  # 6 pairs, index 0..5


def collinear(p, q, r):
    """Exact integer collinearity test via cross product (no floats)."""
    (x1, y1), (x2, y2), (x3, y3) = p, q, r
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0


def line_multiplicity(pair_idx):
    """How many of the 4 points lie on the line through PAIRS[pair_idx]."""
    i, j = PAIRS[pair_idx]
    p, q = POINTS[i], POINTS[j]
    count = 0
    for k, r in enumerate(POINTS):
        if k == i or k == j:
            count += 1
            continue
        if collinear(p, q, r):
            count += 1
    return count


multiplicities = [line_multiplicity(idx) for idx in range(6)]
ordinary_indices = [idx for idx, m in enumerate(multiplicities) if m == 2]
classical_ordinary_count = len(ordinary_indices)

# Sanity: this must match A003034(4) = 3, derived here, not assumed.
assert classical_ordinary_count == 3, (
    f"expected 3 ordinary lines for this 4-point configuration, "
    f"got {classical_ordinary_count} (multiplicities={multiplicities})"
)

print("Points:", POINTS)
print("Pairs (index -> (i,j), line multiplicity):")
for idx, (i, j) in enumerate(PAIRS):
    print(f"  {idx}: ({i},{j}) mult={multiplicities[idx]}"
          f"{'  <-- ordinary' if idx in ordinary_indices else ''}")
print("Classical ordinary-line pair indices:", ordinary_indices)
print("Classical ordinary-line count (should equal A003034(4)=3):",
      classical_ordinary_count)


# ---------------------------------------------------------------------------
# 2. Grover search over the 6 pair-indices (3 qubits, states 0..7) marking
#    exactly the ordinary-line indices computed above.
# ---------------------------------------------------------------------------

N_QUBITS = 3
N_STATES = 2 ** N_QUBITS  # 8
MARKED = set(ordinary_indices)  # subset of {0,...,5}


def oracle_circuit():
    """Phase-flip the marked basis states (multi-controlled Z per state)."""
    qc = QuantumCircuit(N_QUBITS, name="Oracle")
    for m in MARKED:
        bits = format(m, f"0{N_QUBITS}b")
        # flip qubits that should be 0 so the marked state looks like |11..1>
        for q, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(q)
        if N_QUBITS == 1:
            qc.z(0)
        else:
            qc.h(N_QUBITS - 1)
            qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
            qc.h(N_QUBITS - 1)
        for q, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(q)
    return qc


def diffuser_circuit():
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(N_QUBITS, name="Diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


n_marked = len(MARKED)
iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / n_marked)))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
oracle = oracle_circuit()
diffuser = diffuser_circuit()
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

print(f"\nGrover: {N_STATES} states, {n_marked} marked, {iterations} iteration(s)")

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit prints the classical bitstring as c[n-1]...c[0] (leftmost char is
# the highest-index qubit/clbit), which already matches "qubit q contributes
# weight 2**q" when read as a plain binary number -- no reversal needed.
def bitstring_to_index(bs):
    return int(bs, 2)

index_counts = {}
for bitstring, c in counts.items():
    idx = bitstring_to_index(bitstring)
    index_counts[idx] = index_counts.get(idx, 0) + c

print("Measured index -> counts (out of", shots, "shots):")
for idx in sorted(index_counts):
    flag = " <-- marked" if idx in MARKED else ""
    print(f"  {idx}: {index_counts[idx]}{flag}")

# The top-n_marked most frequent measured indices should be exactly the
# marked (ordinary-line) indices.
top_indices = sorted(index_counts, key=lambda k: -index_counts[k])[:n_marked]
quantum_found = set(top_indices)

marked_shots = sum(index_counts.get(i, 0) for i in MARKED)
marked_fraction = marked_shots / shots

print("\nTop", n_marked, "measured indices:", sorted(quantum_found))
print("Classical marked (ordinary-line) indices:", sorted(MARKED))
print(f"Fraction of shots landing on a marked (ordinary-line) index: "
      f"{marked_fraction:.3f}")

verified = (quantum_found == MARKED) and (marked_fraction > 0.5) and (
    classical_ordinary_count == 3
)

print("\nRESULT:", "PASS" if verified else "FAIL")
if not verified:
    raise SystemExit(1)
