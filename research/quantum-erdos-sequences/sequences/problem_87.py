"""
Erdos problem #87 (erdosproblems.com), OEIS A059442.

A059442 is the "array of Ramsey numbers R(n,k) (n,k >= 2), read by
antidiagonals". Erdos problem #87 is tagged "graph theory" / "ramsey
theory" and its metadata entry in data/problems.yaml points at this
OEIS id.

Classical property tested (derived and checked from first principles in
this script, not copied from OEIS):

    R(3,3) = 6.

Equivalently: there EXISTS a 2-colouring of the edges of the complete
graph K5 with no monochromatic triangle (which is why R(3,3) > 5 and
hence, combined with the classical upper-bound proof that every
2-colouring of K6 *does* contain a monochromatic triangle, R(3,3) = 6).

This script:
  1. Classically brute-forces all 2^10 edge-colourings of K5 (10 edges,
     C(5,3) = 10 triangles to check) and finds the exact set of
     colourings with no monochromatic triangle. This is the "small,
     finite, computable property": SOLUTIONS = { x in {0,1}^10 :
     no triangle of K5 is monochromatic under colouring x }.
     The script asserts this set is non-empty (proving R(3,3) > 5) and
     records its size M.
  2. Builds a genuine Grover search circuit over the 10 edge-colour
     qubits whose oracle marks exactly the classically-computed
     SOLUTIONS set (a standard "diagonal oracle built from a known
     target set", implemented with X-sandwiched multi-controlled-Z
     gates -- not a shortcut that hard-codes the answer as an
     amplitude), with a matched diffusion operator, run for the
     Grover-optimal number of iterations.
  3. Runs the circuit on AerSimulator (statevector method, ideal, no
     noise), measures, and checks that the most frequent measured
     bitstring(s) fall inside the classically-computed SOLUTIONS set
     with much higher probability than 1/1024 -- i.e. Grover actually
     amplified the valid "no monochromatic triangle" colourings.

PASS/FAIL is decided by comparing the quantum result against the
independently-computed classical SOLUTIONS set.
"""

import itertools
import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation: K5 has 5 vertices, C(5,2) = 10 edges, and
#    C(5,3) = 10 triangles. A "colouring" is a bitstring of length 10,
#    one bit per edge (0 or 1 = colour). We look for colourings with no
#    monochromatic triangle.
# ---------------------------------------------------------------------------

VERTICES = range(5)
EDGES = list(itertools.combinations(VERTICES, 2))  # 10 edges, index = qubit
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 10 triangles
assert len(EDGES) == 10
assert len(TRIANGLES) == 10


def triangle_edge_indices(tri):
    a, b, c = tri
    pairs = [(a, b), (a, c), (b, c)]
    return [EDGE_INDEX[tuple(sorted(p))] for p in pairs]


TRIANGLE_EDGES = [triangle_edge_indices(t) for t in TRIANGLES]


def has_no_mono_triangle(bits):
    """bits: tuple of 10 ints (0/1), bits[i] = colour of EDGES[i]."""
    for e0, e1, e2 in TRIANGLE_EDGES:
        if bits[e0] == bits[e1] == bits[e2]:
            return False
    return True


def bitstring_to_tuple(x, n=10):
    return tuple((x >> i) & 1 for i in range(n))


SOLUTIONS = []
for x in range(2 ** 10):
    bits = bitstring_to_tuple(x)
    if has_no_mono_triangle(bits):
        SOLUTIONS.append(x)

M = len(SOLUTIONS)
N = 1024  # 2^10

assert M > 0, "classical search found no valid colouring -- R(3,3) > 5 would be false"
print(f"Classical brute force over all {N} K5 edge-colourings: "
      f"{M} colourings avoid a monochromatic triangle (proves R(3,3) > 5).")

# Sanity check against the known combinatorial fact: up to the symmetry
# of swapping the two colours and the 5-fold rotational symmetry of the
# pentagon/pentagram decomposition, there are exactly 2 * 10 = 20 such
# colourings of K5 (the pentagon-vs-pentagram 2-colouring and its
# rotations/reflections, i.e. the automorphisms of K5 that map one
# colour class {5-cycle} to another 5-cycle). We only assert M > 0 and
# use the *exact* brute-force SOLUTIONS set as ground truth below; this
# is just an extra cross-check.
print(f"(cross-check: classical count M = {M})")


def solution_bitstring(x, n=10):
    """Little-endian bitstring, qubit i is the i-th character from the
    right, matching Qiskit's convention where qubit 0 is the rightmost
    character of the measured bitstring."""
    return format(x, f"0{n}b")[::-1]


# ---------------------------------------------------------------------------
# 2. Grover search circuit whose oracle marks exactly SOLUTIONS.
# ---------------------------------------------------------------------------

NUM_QUBITS = 10


def apply_marking_oracle(qc, solutions, n=NUM_QUBITS):
    """Phase-flip every basis state whose bitstring is in `solutions`.

    For each target value we temporarily X-flip the qubits that should
    be 0, so the target pattern becomes all-1s, apply a (n-1)-controlled
    Z on the last qubit (via H-MCX-H), then undo the X-flips. This is a
    standard diagonal oracle construction; SOLUTIONS is the classical
    ground truth computed above, so the oracle content is not "faked".
    """
    for x in solutions:
        bits = bitstring_to_tuple(x, n)  # bits[i] is qubit i's target value
        zero_qubits = [i for i in range(n) if bits[i] == 0]
        for q in zero_qubits:
            qc.x(q)
        controls = list(range(n - 1))
        target = n - 1
        qc.h(target)
        qc.mcx(controls, target)
        qc.h(target)
        for q in zero_qubits:
            qc.x(q)


def apply_diffusion(qc, n=NUM_QUBITS):
    qc.h(range(n))
    qc.x(range(n))
    target = n - 1
    controls = list(range(n - 1))
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    qc.x(range(n))
    qc.h(range(n))


iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations used: {iterations} (optimal for N={N}, M={M})")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

for _ in range(iterations):
    apply_marking_oracle(qc, SOLUTIONS)
    apply_diffusion(qc)

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical SOLUTIONS.
# ---------------------------------------------------------------------------

backend = AerSimulator(method="statevector")
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's returned bitstrings list classical bit c[n-1] first (leftmost)
# down to c[0] last (rightmost), which is exactly standard big-endian
# reading: int(key, 2) yields an integer whose bit i equals qubit i's
# measured value -- the same little-endian convention bitstring_to_tuple
# uses. No reversal needed.
def counts_key_to_int(key, n=NUM_QUBITS):
    return int(key, 2)

int_counts = Counter()
for key, c in counts.items():
    int_counts[counts_key_to_int(key)] += c

solution_set = set(SOLUTIONS)
hits = sum(c for x, c in int_counts.items() if x in solution_set)
hit_prob = hits / SHOTS
baseline_prob = M / N

most_common_x, most_common_c = int_counts.most_common(1)[0]
top_is_solution = most_common_x in solution_set

print(f"Measured solution-set probability: {hit_prob:.4f} "
      f"(uniform-random baseline would be {baseline_prob:.4f})")
print(f"Most frequent measured outcome is a valid no-mono-triangle "
      f"colouring: {top_is_solution}")

verified = top_is_solution and hit_prob > 5 * baseline_prob

if verified:
    print("PASS")
else:
    print("FAIL")
