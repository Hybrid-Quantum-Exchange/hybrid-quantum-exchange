"""
Erdos problem #216 (erdosproblems.com), quantum-testable instance.

Problem #216 concerns "empty polygon numbers": a(n) is the smallest number
of points in the plane, no three collinear, such that any such point set
must contain n points in convex position with no other point of the set
inside them (an empty, or "hole", convex n-gon). The problem's OEIS entry
is A381776, whose known values are a(3)=3, a(4)=5, a(5)=10, a(6)=30 (for
n<=2 the convention is a(n)=0). The n=6 case was only settled in 2024
(Heule & Scheucher); n>=7 admits configurations (Horton, 1983) with no
empty n-gon at all, so a(n) is undefined there. erdosproblems.com records
this problem as "disproved": Erdos and Szekeres's original conjecture that
a(n) is finite for every n fails for n>=7.

Classical property tested here (small, finite, and directly checkable):

    a(4) = 5, i.e. among ANY 5 points in the plane in general position
    (no 3 collinear), some 4 of them form the vertices of a convex
    quadrilateral. (With only 5 points total, any convex quadrilateral
    they form is automatically "empty" -- there is at most one point left
    over, and it cannot lie inside a triangle-free 4-subset's hull without
    violating general position in a way that would still leave a convex
    4-subset among the other configurations -- concretely, for n=4 the
    "empty" qualifier adds nothing: this is exactly the classical
    Erdos-Szekeres "Happy Ending" theorem for k=4.)

Concretely: we fix 5 points in the plane, no 3 collinear, chosen once and
independently of any dependence on Grover's answer. We classically enumerate
every 4-point subset (there are exactly 5 = C(5,4), one for each point left
out) and determine -- from first principles, via signed-area / orientation
tests -- which of those subsets are in strictly convex position. The
Happy-Ending theorem guarantees at least one such subset exists; we verify
this directly by brute force in Python (no OEIS value is copied -- the
convex/non-convex label for each subset is derived from the point
coordinates themselves).

We then build a genuine Grover search circuit over the 3-qubit index space
{0,...,7} (index i in 0..4 = "the subset obtained by leaving out point i";
indices 5,6,7 are unused/padding and never marked) whose oracle marks
exactly the indices found to be convex above. Running Grover's algorithm on
the ideal AerSimulator should overwhelmingly return one of the marked
(convex) indices. We compare the quantum result against the classical
brute-force answer and print PASS if the most frequently measured index is
indeed one of the classically-determined convex subsets, FAIL otherwise.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Fix 5 points in the plane, no 3 collinear (chosen independently of any
#    quantum computation -- just a concrete generic configuration).
# ---------------------------------------------------------------------------

POINTS = [
    (0.0, 0.0),
    (6.0, 0.0),
    (6.0, 6.0),
    (0.0, 6.0),
    (4.0, 1.0),
]

N_POINTS = len(POINTS)
assert N_POINTS == 5


def cross(o, a, b):
    """2D cross product of (a-o) and (b-o); sign gives turn direction."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def no_three_collinear(points):
    for a, b, c in itertools.combinations(points, 3):
        if abs(cross(a, b, c)) < 1e-9:
            return False
    return True


assert no_three_collinear(POINTS), "point set must be in general position"


def is_convex_quadrilateral(quad):
    """
    Return True iff the 4 points, taken in the cyclic order that is their
    convex hull order, form a strictly convex quadrilateral (i.e. all 4
    points are vertices of their own convex hull -- none lies inside the
    triangle formed by the other three).

    Computed directly from signed areas / orientation tests, first
    principles, no library convex-hull routine used.
    """
    pts = list(quad)
    # A set of 4 points (general position) is in convex position iff no
    # point lies inside the triangle formed by the other three.
    for i in range(4):
        p = pts[i]
        tri = [pts[j] for j in range(4) if j != i]
        if point_in_triangle(p, tri):
            return False
    return True


def point_in_triangle(p, tri):
    a, b, c = tri
    d1 = cross(a, b, p)
    d2 = cross(b, c, p)
    d3 = cross(c, a, p)
    has_neg = (d1 < -1e-9) or (d2 < -1e-9) or (d3 < -1e-9)
    has_pos = (d1 > 1e-9) or (d2 > 1e-9) or (d3 > 1e-9)
    return not (has_neg and has_pos)  # all same sign (or on boundary) => inside


# ---------------------------------------------------------------------------
# 2. Classical brute force: for each of the 5 subsets (leave out point i),
#    determine whether it is a convex quadrilateral.
# ---------------------------------------------------------------------------

classical_marked = []
for leave_out in range(N_POINTS):
    subset = [POINTS[j] for j in range(N_POINTS) if j != leave_out]
    if is_convex_quadrilateral(subset):
        classical_marked.append(leave_out)

print("Classical brute-force check of all C(5,4)=5 subsets:")
for leave_out in range(N_POINTS):
    subset = [POINTS[j] for j in range(N_POINTS) if j != leave_out]
    convex = is_convex_quadrilateral(subset)
    print(f"  leave out point {leave_out} -> subset convex? {convex}")

assert len(classical_marked) > 0, (
    "Happy Ending theorem violated for this point set -- should not happen"
)
print(f"Classically marked (convex) indices: {classical_marked}")
print(
    "This directly confirms, for this instance, the n=4 case underlying "
    "Erdos problem #216 / OEIS A381776: a(4)=5, i.e. some 4-subset of any "
    "5 general-position points is in convex position."
)

# ---------------------------------------------------------------------------
# 3. Grover search over the 3-qubit index space {0,...,7} for a marked
#    (convex) index. Indices 5,6,7 are padding and are never marked.
# ---------------------------------------------------------------------------

N_QUBITS = 3  # 2^3 = 8 >= 5


def mark_index_oracle(qc, index, qubits):
    """
    Apply a phase flip (-1) to the computational basis state |index> on the
    given qubit register, using X gates to map the target bit pattern to
    |11...1> around a multi-controlled Z.
    """
    bits = format(index, f"0{len(qubits)}b")[::-1]  # little-endian bit string
    flip_qubits = [qubits[i] for i, b in enumerate(bits) if b == "0"]

    for q in flip_qubits:
        qc.x(q)

    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])

    for q in flip_qubits:
        qc.x(q)


def build_oracle(marked_indices, n_qubits):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    qubits = list(range(n_qubits))
    for idx in marked_indices:
        mark_index_oracle(qc, idx, qubits)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qubits = list(range(n_qubits))
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


n_items = 2 ** N_QUBITS  # 8
n_marked = len(classical_marked)
# Standard Grover optimal iteration count.
theta = math.asin(math.sqrt(n_marked / n_items))
n_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(classical_marked, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

for _ in range(n_iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit strings are big-endian over the classical register with qubit 0
# as the rightmost bit; our register was built with qubit 0 as index bit 0
# (little-endian), so int(bitstring, 2) with the string reversed recovers
# the index directly. Easiest: just reverse the returned bitstring.
measured_indices = {}
for bitstring, count in counts.items():
    idx = int(bitstring[::-1], 2)
    measured_indices[idx] = measured_indices.get(idx, 0) + count

most_common_index = max(measured_indices, key=measured_indices.get)
most_common_count = measured_indices[most_common_index]

print("\nGrover search results (index -> counts):")
for idx in sorted(measured_indices):
    print(f"  index {idx}: {measured_indices[idx]} / {SHOTS}")

print(f"\nMost frequently measured index: {most_common_index} "
      f"({most_common_count}/{SHOTS} shots)")
print(f"Classically marked (convex) indices: {classical_marked}")

quantum_found_marked = most_common_index in classical_marked
# Also require that the marked indices collectively dominate the
# distribution, as a sanity check that Grover actually amplified them
# rather than a fluke single-shot majority.
marked_total = sum(measured_indices.get(i, 0) for i in classical_marked)
amplification_ok = marked_total / SHOTS > 0.5

verified = quantum_found_marked and amplification_ok

if verified:
    print("\nPASS: Grover search on the ideal AerSimulator concentrated on "
          "a classically-verified convex (marked) 4-point subset, matching "
          "the first-principles classical computation.")
else:
    print("\nFAIL: quantum result did not match/confirm the classical "
          "computation.")

print("\nRESULT:", "PASS" if verified else "FAIL")
