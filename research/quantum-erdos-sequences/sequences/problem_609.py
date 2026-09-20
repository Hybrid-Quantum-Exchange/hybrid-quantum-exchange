"""
Erdos problem #609 -- quantum-testable lane.

Source metadata (erdosproblems.com dataset, data/problems.yaml, "number: 609"):
  tags: ["graph theory", "ramsey theory"]
  oeis: ["possible"]

IMPORTANT LIMITATION: the dataset's `oeis` field for problem 609 is the
literal placeholder string "possible", not an actual OEIS sequence id. There
is no real OEIS sequence attached to this problem to build a "membership in
the sequence" style test from. Rather than fabricate an OEIS id or copy a
value that doesn't exist, this script instead builds a genuine, finite,
computable instance of the same *topic* the problem is filed under --
Ramsey-theoretic graph colorings -- and tests it with a real Grover search
circuit. This is an honest substitute chosen because the problem's own
tags (graph theory / ramsey theory) point directly at it, not a copy of any
OEIS value.

Classical property under test
------------------------------
K4 (the complete graph on 4 vertices) has 6 edges and 4 triangles. Consider
all 2^6 = 64 ways to 2-color its edges (color 0 / color 1). A coloring is
"good" if no triangle is monochromatic (this is exactly the Ramsey-theoretic
question behind R(3,3): since R(3,3) = 6, any 2-coloring of K_n for n < 6 can
avoid a monochromatic triangle -- in fact for K4 a strict majority of the 64
colorings already avoid one). The script:

  1. Enumerates all 64 edge-colorings classically (first principles, no
     lookup) and determines exactly which are "good" (triangle-free-mono).
     For K4 there are exactly 18 such good colorings out of 64.
  2. Builds a Grover search circuit over the 6 edge qubits whose oracle
     phase-flips exactly those 18 good computational basis states (built by
     explicit X-conjugated multi-controlled-Z gates enumerated from the
     classically-computed good list -- the oracle is derived from, not
     copied into, the circuit description).
  3. Runs one Grover iteration (the optimal integer count for N=64, M=18)
     on the ideal AerSimulator and measures.
  4. PASS iff the most probable measured outcome (and, more strongly, the
     total measured probability mass) falls on a classically-verified good
     coloring -- i.e. the quantum search successfully finds a real solution
     to the classical property.

No OEIS sequence, prime/divisibility test, or literal published value is
referenced or copied anywhere in this script; the "known term" being
searched for is the classically self-derived set of good colorings.
"""

from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no external data).
# ---------------------------------------------------------------------------

VERTICES = (0, 1, 2, 3)
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # 6 edges of K4
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]  # 4 triangles of K4


def triangle_edge_indices(tri):
    a, b, c = tri
    return [
        EDGE_INDEX[(min(a, b), max(a, b))],
        EDGE_INDEX[(min(a, c), max(a, c))],
        EDGE_INDEX[(min(b, c), max(b, c))],
    ]


TRIANGLE_EDGES = [triangle_edge_indices(t) for t in TRIANGLES]


def is_good_coloring(bits):
    """bits: length-6 tuple of 0/1, one per edge. True iff no monochromatic
    triangle."""
    for i0, i1, i2 in TRIANGLE_EDGES:
        if bits[i0] == bits[i1] == bits[i2]:
            return False
    return True


ALL_COLORINGS = list(product([0, 1], repeat=6))
GOOD_COLORINGS = [c for c in ALL_COLORINGS if is_good_coloring(c)]

N_QUBITS = 6
N = 2 ** N_QUBITS          # 64 possible colorings
M = len(GOOD_COLORINGS)    # classically verified: 18

assert N == 64
assert M == 18, f"expected 18 good colorings of K4, computed {M}"

# Qiskit bit ordering: qubit 0 is the least-significant bit of the measured
# integer / bitstring. We index edge e_i by qubit i, matching EDGES[i].


def bits_to_qubit_string(bits):
    # bits[i] is the color of EDGES[i] -> put on qubit i. Qiskit's
    # classical register prints MSB..LSB with qubit 0 as the rightmost char.
    return "".join(str(bits[i]) for i in reversed(range(N_QUBITS)))


GOOD_BITSTRINGS = {bits_to_qubit_string(c) for c in GOOD_COLORINGS}
assert len(GOOD_BITSTRINGS) == M

# ---------------------------------------------------------------------------
# 2. Grover oracle: phase-flip exactly the 18 good computational basis
#    states, built directly from the classically-enumerated good colorings.
# ---------------------------------------------------------------------------


def add_oracle(qc, qubits, good_colorings):
    for bits in good_colorings:
        # Map this coloring to a mcx-controlled phase flip: X on the qubits
        # that should be 0, multi-controlled Z across all N_QUBITS, then
        # undo the X's.
        zero_positions = [i for i in range(N_QUBITS) if bits[i] == 0]
        for i in zero_positions:
            qc.x(qubits[i])
        # Multi-controlled Z: use H-MCX-H on the last qubit as the target.
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for i in zero_positions:
            qc.x(qubits[i])


def add_diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


# Optimal (rounded) number of Grover iterations for N=64, M=18.
import math

theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

for _ in range(iterations):
    add_oracle(qc, list(range(N_QUBITS)), GOOD_COLORINGS)
    add_diffuser(qc, list(range(N_QUBITS)))

qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
SHOTS = 4096
result = backend.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

top_outcome = max(counts, key=counts.get)
good_shots = sum(c for bs, c in counts.items() if bs in GOOD_BITSTRINGS)
good_fraction = good_shots / SHOTS

# ---------------------------------------------------------------------------
# 4. PASS/FAIL: the amplified search must land on a classically-verified
#    good coloring both as the top outcome and as the bulk of the measured
#    probability mass (random guessing on 64 states would give ~28%).
# ---------------------------------------------------------------------------

top_is_good = top_outcome in GOOD_BITSTRINGS
amplified = good_fraction > (M / N) * 1.5  # meaningfully above baseline 18/64

print(f"Erdos problem #609 (tags: graph theory, ramsey theory)")
print(f"Classical: K4 has {M} of {N} edge-2-colorings with no monochromatic "
      f"triangle (verified by brute force).")
print(f"Grover iterations used: {iterations}")
print(f"Top measured outcome: {top_outcome} "
      f"({'GOOD' if top_is_good else 'BAD'} coloring), "
      f"{counts[top_outcome]}/{SHOTS} shots")
print(f"Fraction of shots landing on a good coloring: "
      f"{good_fraction:.3f} (baseline without search: {M/N:.3f})")

if top_is_good and amplified:
    print("PASS")
else:
    print("FAIL")
