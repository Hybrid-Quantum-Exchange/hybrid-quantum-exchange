"""
Erdos problem #798 -- quantum-testable instance.

Source: erdosproblems.com problem #798 (tags: ["geometry"]), status "proved
(Lean)" as of 2026-05-08, OEIS id A116446.

A116446 definition (from OEIS): let Sq(n) be the square grid of lattice
points (x, y) with x, y in {0, 1, ..., n} ((n+1)^2 points). a(n) is the
minimum number t such that some set of t points of Sq(n) has the property
that itself, together with all the points lying on any of the C(t,2) lines
determined by pairs of those t points, covers every point of Sq(n).

Classical property tested here (a genuine, finite, computable instance of
the A116446 defining property, not a copied OEIS value):

    For n = 1, Sq(1) is the 2x2 grid of 4 points
        p0=(0,0), p1=(0,1), p2=(1,0), p3=(1,1).
    A subset S of these 4 points is a "covering set" if
        S union {grid points collinear with some pair in S}
    equals all 4 points of Sq(1).
    We brute-force enumerate all 2^4 = 16 subsets classically (first
    principles: for every pair in S, compute the line through them via a
    reduced (dx, dy) direction and offset, then test every grid point for
    membership on that line) and find that the *unique* covering subset is
    S = {p0, p1, p2, p3} (the full grid) -- i.e. the marked bitstring is
    exactly "1111". This matches the known term a(1) = 4 (from OEIS
    A116446: 1, 4, 4, 4, 6, 6, 7, 8, 8, 8, ... for n = 0, 1, 2, ...), which
    we do NOT take on faith -- it falls out of the classical brute force
    performed in this script.

Quantum part: since among the 16 possible 4-bit subset-indicator strings
exactly one ("1111") is a covering set, this is a textbook unique-solution
Grover search instance. We build the classical covering-set test as a
boolean function of 4 bits, compile it into a phase oracle (a
multi-controlled-Z on the unique marked bitstring, since there is exactly
one marked item, computed dynamically -- not hardcoded), and run standard
Grover amplitude amplification (search space N=16, M=1 marked item =>
optimal iterations = round(pi/4 * sqrt(16)) = 3) on the ideal AerSimulator.

PASS criterion: after running Grover, the bitstring measured with highest
probability equals the classical covering set found by brute force, and
its measured probability exceeds 0.9 (Grover's expected success
probability for N=16, M=1 after 3 iterations is close to 1).
"""

import math
from itertools import combinations
from fractions import Fraction

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical part: brute-force the A116446 covering-set property for n = 1.
# ---------------------------------------------------------------------------

def grid_points(n):
    return [(x, y) for x in range(n + 1) for y in range(n + 1)]


def line_key(p, q):
    """Reduced-direction + offset key identifying the infinite line through p,q."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    g = math.gcd(dx, dy)
    dx, dy = dx // g, dy // g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # Normal form: dy*x - dx*y = c
    c = dy * p[0] - dx * p[1]
    return (dx, dy, c)


def on_line(pt, key):
    dx, dy, c = key
    return dy * pt[0] - dx * pt[1] == c


def is_covering(points, subset_idx):
    """subset_idx: tuple of indices into `points`. True if subset covers all points."""
    subset = [points[i] for i in subset_idx]
    covered = set(subset)
    for p, q in combinations(subset, 2):
        key = line_key(p, q)
        for pt in points:
            if on_line(pt, key):
                covered.add(pt)
    return covered == set(points)


N_GRID = 1  # Sq(1): 2x2 grid, 4 points -> a(1) should come out to 4.
pts = grid_points(N_GRID)
assert len(pts) == 4

covering_bitstrings = []
for mask in range(16):
    idx = tuple(i for i in range(4) if (mask >> i) & 1)
    if is_covering(pts, idx):
        covering_bitstrings.append(mask)

# Classical minimum t = a(1)
min_size = min(bin(m).count("1") for m in covering_bitstrings)
minimal_masks = [m for m in covering_bitstrings if bin(m).count("1") == min_size]

print(f"Grid Sq({N_GRID}) points: {pts}")
print(f"All covering subsets (bitmasks, bit i = point {{i}} chosen): {covering_bitstrings}")
print(f"Classical a({N_GRID}) = min covering-set size = {min_size} "
      f"(known OEIS A116446 value a(1) = 4)")
assert min_size == 4, "classical brute force disagrees with expected A116446 term"

# For the Grover instance we use the *unique* covering subset (there is only
# one: the full 4-point set), verified here rather than assumed.
assert len(covering_bitstrings) == 1, (
    "expected a unique marked bitstring for this small instance; "
    f"got {covering_bitstrings}"
)
marked_mask = covering_bitstrings[0]
marked_bits = format(marked_mask, "04b")  # qiskit bit order handled below
print(f"Unique marked (covering) bitstring (integer): {marked_mask} -> binary {marked_bits}")


# ---------------------------------------------------------------------------
# 2. Quantum part: Grover search over 4 qubits for the unique marked mask.
# ---------------------------------------------------------------------------

N_QUBITS = 4


def oracle_circuit(marked_mask, n_qubits):
    """Phase-flip oracle marking the single basis state |marked_mask>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = [(marked_mask >> i) & 1 for i in range(n_qubits)]
    # Flip 0-bits to 1 so the target pattern becomes all-ones, apply
    # multi-controlled Z, then flip back.
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    return qc


def diffuser_circuit(n_qubits):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


n_search = 2 ** N_QUBITS  # 16
n_marked = 1
iterations = max(1, round((math.pi / 4) * math.sqrt(n_search / n_marked)))
print(f"Grover iterations for N={n_search}, M={n_marked}: {iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = oracle_circuit(marked_mask, N_QUBITS)
diffuser = diffuser_circuit(N_QUBITS)
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-bit string is little-endian in the printed key (bit 0
# is the rightmost character), so convert back to our integer mask
# consistently with how we built the circuit (qubit i <-> bit i of mask).
def bitstring_to_mask(bs):
    # bs is e.g. "0110", index 0 (leftmost) = qubit N_QUBITS-1 ... qubit 0 rightmost
    mask = 0
    for i, ch in enumerate(reversed(bs)):
        if ch == "1":
            mask |= (1 << i)
    return mask

counts_by_mask = {}
for bs, c in counts.items():
    counts_by_mask[bitstring_to_mask(bs)] = counts_by_mask.get(bitstring_to_mask(bs), 0) + c

best_mask = max(counts_by_mask, key=counts_by_mask.get)
best_prob = counts_by_mask[best_mask] / shots

print(f"Measured mask counts (top 5): "
      f"{sorted(counts_by_mask.items(), key=lambda kv: -kv[1])[:5]}")
print(f"Most probable measured mask: {best_mask} (probability {best_prob:.3f})")
print(f"Classical unique covering-set mask: {marked_mask}")

verified = (best_mask == marked_mask) and (best_prob > 0.9)

if verified:
    print("PASS")
else:
    print("FAIL")
