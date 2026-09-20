"""
Erdos problem #544 (erdosproblems.com), OEIS A000791 (Ramsey numbers R(3,n)).

Tags on the problem entry: "graph theory", "ramsey theory". A000791's
n-th term is the diagonal-adjacent Ramsey number R(3,n): the smallest N such
that every 2-coloring of the edges of the complete graph K_N contains a
monochromatic triangle (color class 3) or a monochromatic K_n in the other
color. The first nontrivial, best-known small case is R(3,3) = 6.

The classical property tested here, derived and checked in this script
(not copied from OEIS):

    "R(3,3) > 5", equivalently: there exists a 2-coloring of the edges of
    K_5 with NO monochromatic triangle.

K_5 has C(5,2) = 10 edges, so there are 2^10 = 1024 possible red/blue edge
colorings. We first brute-force enumerate all of them classically and
collect the exact set of triangle-free-in-both-colors colorings (this is
the "small search space whose answer is a known term" -- the fact that
this set is nonempty is exactly the classical statement R(3,3) > 5, i.e.
the first term of A000791, R(3,3) = 6, is not smaller).

We then build a real Grover search circuit over the 10-qubit space of edge
colorings (one qubit per edge of K_5), with an oracle that phase-flips
exactly the classically-identified "good" (triangle-free) colorings, run it
on the ideal AerSimulator, and check that Grover amplifies the good
colorings far above the uniform 1/1024 baseline -- i.e. the quantum search
finds a genuine witness to R(3,3) > 5.

PASS criterion: the most frequently measured 10-bit string, when simulated
with enough Grover iterations, is one of the classically verified
triangle-free colorings, and its measured probability is much larger than
the uniform baseline 1/1024.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: enumerate all 2-colorings of K_5's edges and
#    find every one with no monochromatic triangle.
# ---------------------------------------------------------------------------

N_VERTICES = 5
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 10 edges
assert len(EDGES) == 10
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(range(N_VERTICES), 3))  # 10 triangles


def triangle_edge_indices(tri):
    a, b, c = tri
    return (EDGE_INDEX[(a, b)], EDGE_INDEX[(a, c)], EDGE_INDEX[(b, c)])


TRIANGLE_EDGE_IDX = [triangle_edge_indices(t) for t in TRIANGLES]


def is_triangle_free_coloring(bits):
    """bits: tuple of 10 ints (0/1), one per edge. True iff no triangle is
    monochromatic (all three of its edges the same color)."""
    for i0, i1, i2 in TRIANGLE_EDGE_IDX:
        if bits[i0] == bits[i1] == bits[i2]:
            return False
    return True


N = 1 << 10
good_bitstrings = []  # list of 10-char '0'/'1' strings, qubit 0 = LSB
for value in range(N):
    bits = tuple((value >> k) & 1 for k in range(10))
    if is_triangle_free_coloring(bits):
        good_bitstrings.append(format(value, "010b")[::-1])  # index k -> char k (LSB-first)

M = len(good_bitstrings)
assert M > 0, "classical search found no triangle-free coloring of K_5 -- would refute R(3,3)>5"
good_set = set(good_bitstrings)

print(f"Classical brute force over 2^10 = {N} edge colorings of K_5:")
print(f"  triangle-free colorings found: M = {M}")
print(f"  (nonzero M is exactly the classical fact R(3,3) > 5, "
      f"i.e. A000791(1) = R(3,3) = 6)")

# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 10-qubit space of edge colorings.
# ---------------------------------------------------------------------------

NUM_QUBITS = 10


def apply_multicontrol_z(qc, qubits):
    """Apply a phase flip of -1 to the |11...1> state of `qubits`
    (multi-controlled Z), using the last qubit as target via H-MCX-H."""
    controls = qubits[:-1]
    target = qubits[-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)


def oracle_circuit(marked_bitstrings):
    qc = QuantumCircuit(NUM_QUBITS, name="oracle")
    all_qubits = list(range(NUM_QUBITS))
    for bs in marked_bitstrings:
        # bs[k] is the value of qubit k (LSB-first, matches our encoding above)
        zero_positions = [k for k in range(NUM_QUBITS) if bs[k] == "0"]
        for q in zero_positions:
            qc.x(q)
        apply_multicontrol_z(qc, all_qubits)
        for q in zero_positions:
            qc.x(q)
    return qc


def diffusion_circuit():
    qc = QuantumCircuit(NUM_QUBITS, name="diffusion")
    all_qubits = list(range(NUM_QUBITS))
    qc.h(all_qubits)
    qc.x(all_qubits)
    apply_multicontrol_z(qc, all_qubits)
    qc.x(all_qubits)
    qc.h(all_qubits)
    return qc


iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations chosen: {iterations} (N={N}, M={M})")

grover = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
grover.h(range(NUM_QUBITS))

oracle = oracle_circuit(good_bitstrings)
diffusion = diffusion_circuit()

for _ in range(iterations):
    grover.compose(oracle, inplace=True)
    grover.compose(diffusion, inplace=True)

grover.measure(range(NUM_QUBITS), range(NUM_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(grover, backend)
SHOTS = 20000
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit returns bitstrings MSB-first (qubit N-1 ... qubit 0). Our encoding
# above used LSB-first strings (bs[k] = qubit k), so reverse to compare.
counts_lsb_first = {}
for bitstr, c in counts.items():
    lsb_first = bitstr[::-1]
    counts_lsb_first[lsb_first] = counts_lsb_first.get(lsb_first, 0) + c

best_bitstring, best_count = max(counts_lsb_first.items(), key=lambda kv: kv[1])
best_prob = best_count / SHOTS
uniform_baseline = 1.0 / N

good_prob_mass = sum(counts_lsb_first.get(bs, 0) for bs in good_bitstrings) / SHOTS

print(f"Most frequent measured coloring (LSB-first bitstring): {best_bitstring}")
print(f"  measured probability: {best_prob:.4f}  (uniform baseline: {uniform_baseline:.4f})")
print(f"  total probability mass on ALL classically-good colorings: {good_prob_mass:.4f}")

is_valid_witness = best_bitstring in good_set
amplified = good_prob_mass > 5 * uniform_baseline * M  # well above chance

if is_valid_witness and amplified:
    edges_used = [EDGES[k] for k in range(10) if best_bitstring[k] == "1"]
    print(f"Witness coloring (blue edges = '1'): {edges_used}")
    print("Verification: best.witness is triangle-free under brute-force check:",
          is_triangle_free_coloring(tuple(int(c) for c in best_bitstring)))
    print("PASS")
else:
    print("FAIL")
