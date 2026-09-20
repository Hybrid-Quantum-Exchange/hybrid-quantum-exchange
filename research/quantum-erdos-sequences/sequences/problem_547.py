"""
Erdos problem #547 -- quantum-testable lane (best-effort fallback).

Source metadata (data/problems.yaml, entry "number: '547'"):
    oeis: ["N/A"]
    tags: ["graph theory", "ramsey theory"]
    status: proved

LIMITATION, stated honestly up front: problem #547's entry carries no OEIS
sequence id ("N/A"). The task instructions for this lane say that when there
is no OEIS id, or no finite/small-instance property can be built, the script
should still make its best honest attempt, note the limitation clearly, and
report ran_ok/verified_against_classical accurately rather than faking a
pass. There is therefore no "term of sequence A123456" to test here. What
follows is NOT a property of an OEIS sequence attached to problem #547 -- it
is a small, finite, genuinely computable graph-theory/Ramsey-theory property
in the same subject area as the problem's tags, chosen because it is the
closest thing to a legitimate quantum-testable instance available without an
OEIS id to anchor to.

Classical property actually tested
-----------------------------------
Take the complete graph K4 (4 vertices, 6 edges). Consider all 2-colorings
of its 6 edges (2^6 = 64 colorings total, indexed by a 6-bit string, one bit
per edge). K4 contains exactly 4 triangles (each omitting one vertex). A
coloring is "triangle-free-in-both-colors" (a valid 2-coloring avoiding any
monochromatic triangle) iff none of the 4 triangles has all 3 of its edges
the same color.

Question: does such a coloring exist for K4?

This is computed from first principles below by brute force over all 64
colorings (well within Ramsey's theorem territory, R(3,3) = 6, so K4 -- with
only 4 vertices -- is far from forced to contain a monochromatic triangle,
and indeed valid colorings exist). The script:
  1. Computes the classical answer by exhaustive search over all 64
     6-bit colorings (this *is* the ground truth, derived here, not quoted
     from anywhere).
  2. Builds a real Grover search circuit over 6 qubits (search space size
     64 <= 64, satisfying the "N <= ~64" instance-size guidance) whose
     oracle marks exactly the "good" colorings (no monochromatic triangle),
     runs it on the ideal AerSimulator, and checks that the state(s) Grover
     amplifies are indeed valid colorings per the classical brute-force
     check.
  3. Prints PASS if the quantum search result matches the classical
     brute-force result (i.e., the most probable measured coloring is
     actually monochromatic-triangle-free), FAIL otherwise.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force search over all 2-colorings of K4.
# ---------------------------------------------------------------------------

# K4 vertices 0,1,2,3. Edges indexed 0..5 in a fixed order.
VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges
assert len(EDGES) == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

# The 4 triangles of K4 (each omits exactly one vertex), each given as the
# 3 edge-indices that form it.
TRIANGLES = []
for omit in VERTICES:
    tri_vertices = [v for v in VERTICES if v != omit]
    tri_edges = []
    for a, b in itertools.combinations(tri_vertices, 2):
        e = (a, b) if (a, b) in EDGE_INDEX else (b, a)
        tri_edges.append(EDGE_INDEX[e])
    TRIANGLES.append(tuple(sorted(tri_edges)))
assert len(TRIANGLES) == 4


def is_good_coloring(bits):
    """bits: length-6 sequence of 0/1, bit i = color of EDGES[i].

    Returns True iff no triangle among TRIANGLES is monochromatic
    (all three of its edges sharing the same color).
    """
    for tri in TRIANGLES:
        colors = {bits[i] for i in tri}
        if len(colors) == 1:
            return False
    return True


def bits_of(n, width=6):
    return [(n >> i) & 1 for i in range(width)]


good_colorings = []
for n in range(64):
    bits = bits_of(n)
    if is_good_coloring(bits):
        good_colorings.append(n)

classical_exists = len(good_colorings) > 0
classical_count = len(good_colorings)

print(f"Classical brute force over all {2**6} 2-colorings of K4's 6 edges:")
print(f"  triangle-free-in-both-colors colorings found: {classical_count}")
print(f"  exists such a coloring? {classical_exists}")
if good_colorings:
    example = good_colorings[0]
    print(f"  example witness (integer, LSB=edge0): {example} "
          f"= bits {bits_of(example)}")

# ---------------------------------------------------------------------------
# 2. Grover search circuit: search the 64 colorings for a "good" one.
# ---------------------------------------------------------------------------
#
# 6 search qubits (one per edge/bit). The oracle must flip the phase of
# exactly the basis states |b5 b4 b3 b2 b1 b0> (bit i = qubit i) for which
# is_good_coloring(bits) is True. We build the oracle directly from the
# classical predicate (computed above, honestly, from first principles) by
# marking each good coloring with a controlled-Z-style multi-controlled
# phase flip -- this is a legitimate (if unsophisticated) way to realize an
# arbitrary boolean oracle for a small instance, and it lets us cross-check
# Grover amplification against the classical brute-force set directly.

n_qubits = 6
N = 2 ** n_qubits
M = classical_count  # number of marked (good) states

if M == 0 or M == N:
    # Grover degenerates; nothing meaningful to amplify. Given the brute
    # force above we know this is not the case for K4 (some colorings are
    # good, some are not), but guard anyway.
    optimal_iterations = 0
else:
    theta = math.asin(math.sqrt(M / N))
    optimal_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))


def apply_oracle(qc, qubits):
    """Phase-flip every basis state in `good_colorings` (multi-controlled Z,
    with X-conjugation to match each marked bitstring)."""
    for marked in good_colorings:
        bits = bits_of(marked, n_qubits)
        # Flip qubits that should be 0 so the MCZ triggers on this pattern.
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(qubits[i])
        # Multi-controlled Z on all n_qubits-1 controls + 1 target, realized
        # as H - MCX - H on the last qubit to get a phase flip on |1...1>.
        qc.h(qubits[-1])
        qc.append(MCXGate(n_qubits - 1), qubits[:-1] + [qubits[-1]])
        qc.h(qubits[-1])
        for i in zero_positions:
            qc.x(qubits[i])


def apply_diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.append(MCXGate(n_qubits - 1), qubits[:-1] + [qubits[-1]])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


qc = QuantumCircuit(n_qubits, n_qubits)
qubits = list(range(n_qubits))
qc.h(qubits)

for _ in range(optimal_iterations):
    apply_oracle(qc, qubits)
    apply_diffuser(qc, qubits)

qc.measure(qubits, qubits)

print(f"\nGrover circuit: {n_qubits} qubits, {optimal_iterations} iteration(s), "
      f"{M} marked out of {N} states.")

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical bit order in the returned bitstring is c[n-1]...c[0],
# matching qubit index i -> bit i of our `bits_of` convention, so we can
# parse the returned key directly as our integer encoding.
most_common_bitstring = max(counts, key=counts.get)
measured_int = int(most_common_bitstring[::-1], 2)  # reverse to qubit-index order
measured_prob = counts[most_common_bitstring] / shots

quantum_found_good = is_good_coloring(bits_of(measured_int))

print(f"\nMost frequent measurement: {most_common_bitstring} "
      f"(int={measured_int}, probability={measured_prob:.3f})")
print(f"That measured coloring is triangle-free-in-both-colors "
      f"(classically checked): {quantum_found_good}")

# Also confirm the whole marked set is meaningfully amplified: sum of
# probability mass landing on any classically-good coloring should exceed
# uniform-random expectation (M/N) by a comfortable margin.
good_set = set(good_colorings)
good_mass = sum(c for k, c in counts.items()
                 if int(k[::-1], 2) in good_set) / shots
uniform_expectation = M / N

print(f"Probability mass on classically-good colorings: {good_mass:.3f} "
      f"(uniform-random baseline would be {uniform_expectation:.3f})")

verified = (
    classical_exists
    and quantum_found_good
    and good_mass > uniform_expectation
)

if verified:
    print("\nPASS")
else:
    print("\nFAIL")
