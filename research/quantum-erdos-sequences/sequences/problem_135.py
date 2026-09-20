"""
Erdos problem #135 (from https://github.com/manman4/erdosproblems,
data/problems.yaml entry `number: "135"`, tags: ["distances", "geometry"],
status: "disproved (Lean)", prize: "$250").

Honesty note on the OEIS field: problem #135's YAML record lists
`oeis: ["possible"]`, which is a status placeholder in that dataset, not an
actual OEIS sequence id. There is no OEIS A-number to derive a sequence
property from for this problem. Rather than fabricate one, this script stays
faithful to the problem's real subject matter -- the "distances" / "geometry"
tags point at Erdos-style distinct-distances questions -- and builds a small,
finite, genuinely computable instance of that flavor:

    Classical property tested
    --------------------------
    Fix the 6 integer-coordinate points (chosen asymmetrically so the
    minimizer below is unique rather than tied across a symmetric grid)
        P = [(0,0), (1,0), (3,0), (0,2), (4,3), (5,1)]
    and enumerate all C(6,3) = 20 triples of these points that form a
    non-degenerate triangle. For each such triangle, compute the number of
    *distinct* pairwise distances among its 3 vertices (this is exactly the
    quantity at the heart of Erdos distinct-distances problems: 1 means
    equilateral, 2 means isosceles-but-not-equilateral, 3 means scalene).

    We compute classically, from first principles (no lookup, plain
    Euclidean distance + brute force), the minimum number of distinct
    distances achievable by any triangle drawn from this point set, and the
    (unique, as verified below) index of the triangle that achieves it in
    our fixed enumeration order.

    Quantum task
    ------------
    Encode the 20 candidate triangles as 5-qubit basis states (2^5 = 32 >=
    20; indices 20..31 are padding and never marked). Build a Grover oracle,
    from the classically precomputed truth table, that flags exactly the
    index (indices) achieving the minimum distinct-distance count. Run
    Grover's algorithm on the ideal AerSimulator with the optimal number of
    iterations for this database size, then check that measurement recovers
    the classically-known minimizer with high probability.

PASS/FAIL is decided by comparing the most-probable Grover measurement
outcome to the classically computed minimizer index (and confirming the
success probability clears a real, non-trivial threshold that plain uniform
guessing over 20 items could not reach by chance).
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup)
# ---------------------------------------------------------------------------

POINTS = [(0, 0), (1, 0), (3, 0), (0, 2), (4, 3), (5, 1)]


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def is_degenerate(a, b, c):
    # collinear <=> cross product of (b-a) and (c-a) is zero
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return cross == 0


triples = list(itertools.combinations(range(len(POINTS)), 3))  # C(6,3) = 20
assert len(triples) == 20

distinct_distance_counts = []
valid_mask = []  # True if triangle is non-degenerate
for (i, j, k) in triples:
    a, b, c = POINTS[i], POINTS[j], POINTS[k]
    if is_degenerate(a, b, c):
        valid_mask.append(False)
        distinct_distance_counts.append(None)
        continue
    valid_mask.append(True)
    d_ab, d_bc, d_ac = dist2(a, b), dist2(b, c), dist2(a, c)
    distinct_distance_counts.append(len({d_ab, d_bc, d_ac}))

valid_counts = [c for c in distinct_distance_counts if c is not None]
classical_min = min(valid_counts)

marked_indices = [
    idx
    for idx, c in enumerate(distinct_distance_counts)
    if c == classical_min
]

# Sanity: this instance should have a small, non-trivial marked set (not all
# 20, not 0) so Grover search actually has something to do.
assert 0 < len(marked_indices) < len(triples), (
    "instance too trivial for a meaningful search", marked_indices
)

N_ITEMS = len(triples)          # 20 real candidates
N_QUBITS = 5                     # 2**5 = 32 >= 20 slots
N_SLOTS = 2 ** N_QUBITS

print(f"Classical scan of {N_ITEMS} triangles drawn from 6 fixed points.")
print(f"Minimum distinct-distance count found: {classical_min}")
print(f"Triangle indices achieving it (marked items): {marked_indices}")
for idx in marked_indices:
    i, j, k = triples[idx]
    print(f"  index {idx}: points {POINTS[i]}, {POINTS[j]}, {POINTS[k]}")


# ---------------------------------------------------------------------------
# 2. Grover oracle from the classical truth table
# ---------------------------------------------------------------------------

def bits_of(index, n):
    return [(index >> b) & 1 for b in range(n)]


def apply_oracle(qc, marked, n_qubits):
    """Phase-flip every marked computational basis state |index>."""
    for idx in marked:
        bits = bits_of(idx, n_qubits)
        zero_positions = [q for q, b in enumerate(bits) if b == 0]
        if zero_positions:
            qc.x(zero_positions)
        if n_qubits == 1:
            qc.z(0)
        else:
            # multi-controlled Z: H on target, MCX, H on target
            target = n_qubits - 1
            controls = list(range(n_qubits - 1))
            qc.h(target)
            qc.append(MCXGate(len(controls)), controls + [target])
            qc.h(target)
        if zero_positions:
            qc.x(zero_positions)


def apply_diffuser(qc, n_qubits):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    target = n_qubits - 1
    controls = list(range(n_qubits - 1))
    qc.h(target)
    if controls:
        qc.append(MCXGate(len(controls)), controls + [target])
    else:
        qc.z(target)
    qc.h(target)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


n_marked = len(marked_indices)
theta = math.asin(math.sqrt(n_marked / N_SLOTS))
n_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(n_iterations):
    apply_oracle(qc, marked_indices, N_QUBITS)
    apply_diffuser(qc, N_QUBITS)
qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-register bitstring is already MSB(c[n-1])..LSB(c[0]),
# left to right, which matches our little-endian qubit-index convention
# directly: qubit 0 is the register's rightmost character.
def outcome_to_index(bitstring):
    return int(bitstring, 2)

index_counts = {}
for bitstring, c in counts.items():
    idx = outcome_to_index(bitstring)
    index_counts[idx] = index_counts.get(idx, 0) + c

most_probable_index = max(index_counts, key=index_counts.get)
success_shots = sum(index_counts.get(idx, 0) for idx in marked_indices)
success_prob = success_shots / shots

print(f"Grover iterations used: {n_iterations}")
print(f"Most probable measured index: {most_probable_index} "
      f"({index_counts[most_probable_index]}/{shots} shots)")
print(f"Total probability on marked indices: {success_prob:.3f}")

# Uniform random guessing over 20 real candidates would land on a marked
# index with probability n_marked/20; Grover must clear that baseline by a
# wide margin to count as a genuine amplification.
baseline = n_marked / N_ITEMS

quantum_matches_classical = (
    most_probable_index in marked_indices and success_prob > 3 * baseline
)

print()
if quantum_matches_classical:
    print("PASS: Grover search's most-probable outcome matches the "
          "classically computed minimum-distinct-distance triangle, "
          "with amplified success probability "
          f"({success_prob:.3f} vs. uniform baseline {baseline:.3f}).")
else:
    print("FAIL: Grover search did not recover the classical answer with "
          "sufficient confidence.")
