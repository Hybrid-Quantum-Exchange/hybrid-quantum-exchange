"""
Erdos problem #657 (erdosproblems.com), a "geometry"/"distances"-tagged problem
(the distinct-distances family: how few distinct pairwise distances can N
points in the plane determine). As of the source clone read for this script,
problem 657's `oeis` field in data/problems.yaml is the literal string
"possible" -- i.e. there is NO real OEIS sequence id attached to this problem
in the data. This is therefore NOT a case where an OEIS sequence's terms can
be looked up or verified; that part of the task cannot be done honestly for
this problem.

LIMITATION, stated plainly: because no OEIS id exists for #657, this script
does not test membership/growth of an OEIS sequence. Instead, in the spirit
of the problem's own subject (minimum number of distinct pairwise distances
determined by a finite point set) it tests a small, fully self-contained,
classically-checkable combinatorial instance drawn directly from that
subject, and uses a genuine Grover search circuit to find it.

Classical property tested
--------------------------
Fix the 3x3 integer grid {0,1,2} x {0,1,2} (9 points). Over all 4-point
subsets of that grid (C(9,4) = 126 subsets), let m* be the minimum number of
distinct pairwise (squared) Euclidean distances determined by any 4-point
subset. This minimum, and the set of subsets that achieve it, are computed
classically in this script by brute force (first principles: enumerate all
126 subsets, compute the 6 pairwise squared distances for each, count
distinct values, take the min).

Those minimizing subsets are then encoded as marked indices (7 qubits index
the 126 <= 128 subsets) and Grover's algorithm is run on the ideal
AerSimulator to search for a minimizer. The script PASSes if the
highest-probability measured index is one of the classically-verified
minimizing subsets.

This is a real amplitude-amplification search (oracle + diffuser, iterated
the standard floor(pi/4 * sqrt(N/k)) times) over a search space whose correct
answer is independently computed classically in this same script -- not a
copied literal value.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (first principles): find the minimum number of
#    distinct pairwise squared distances among 4-point subsets of the 3x3
#    integer grid, and which subsets achieve it.
# ---------------------------------------------------------------------------

grid_points = [(x, y) for x in range(3) for y in range(3)]  # 9 points
assert len(grid_points) == 9

subsets = list(itertools.combinations(range(9), 4))  # C(9,4) = 126
assert len(subsets) == 126


def distinct_distance_count(subset_indices):
    pts = [grid_points[i] for i in subset_indices]
    dists = set()
    for (x1, y1), (x2, y2) in itertools.combinations(pts, 2):
        dists.add((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return len(dists)


counts = [distinct_distance_count(s) for s in subsets]
m_star = min(counts)
minimizers = [i for i, c in enumerate(counts) if c == m_star]

print(f"Classical brute force over all {len(subsets)} 4-point subsets of the 3x3 grid:")
print(f"  minimum number of distinct pairwise squared distances m* = {m_star}")
print(f"  number of minimizing subsets = {len(minimizers)}")
print(f"  minimizing subset indices (first 5 shown) = {minimizers[:5]}")

# ---------------------------------------------------------------------------
# 2. Grover search over the 126 subset indices (padded to 128 = 2^7), marking
#    exactly the classically-computed minimizer indices.
# ---------------------------------------------------------------------------

n_qubits = 7  # 2^7 = 128 >= 126
N = 2 ** n_qubits
marked = set(minimizers)  # indices 126,127 are padding and never marked


def bits_of(i, n):
    return [(i >> b) & 1 for b in range(n)]


def oracle(qc, marked_indices, n):
    """Phase-flip exactly the marked computational basis states."""
    for idx in marked_indices:
        b = bits_of(idx, n)
        # flip qubits that are 0 in this index so the multi-controlled Z
        # triggers exactly on this bit pattern
        for q, bit in enumerate(b):
            if bit == 0:
                qc.x(q)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for q, bit in enumerate(b):
            if bit == 0:
                qc.x(q)


def diffuser(qc, n):
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


k = len(marked)
iterations = max(1, round((math.pi / 4) * math.sqrt(N / k)))
print(f"Grover: N={N} states, k={k} marked, iterations={iterations}")

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    oracle(qc, marked, n_qubits)
    diffuser(qc, n_qubits)
qc.measure(range(n_qubits), range(n_qubits))

sim = AerSimulator()
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=2048).result()
counts_out = result.get_counts()

# Qiskit counts keys are written MSB-first with c[0] as the rightmost
# character, which is exactly standard big-endian int() parsing -- so no
# reversal is needed (verified against a known single-qubit-flip circuit).
def bitstring_to_index(bs):
    return int(bs, 2)

top_bitstring = max(counts_out, key=counts_out.get)
top_index = bitstring_to_index(top_bitstring)
top_prob = counts_out[top_bitstring] / sum(counts_out.values())

print(f"Grover most-likely measured index = {top_index} (prob {top_prob:.3f})")
print(f"Is it a classically-verified minimizer? {top_index in marked}")

quantum_result_is_correct = top_index in marked

if quantum_result_is_correct:
    print("PASS")
else:
    print("FAIL")
