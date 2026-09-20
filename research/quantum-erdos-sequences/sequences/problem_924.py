"""
Erdos problem #924 (as catalogued in erdosproblems/data/problems.yaml,
entry `number: "924"`).

Metadata on record for this problem:
    prize:           no
    informal_status: proved (2025-08-31)
    oeis:             ["N/A"]
    tags:             ["graph theory", "ramsey theory"]

LIMITATION, stated honestly up front: problem #924 has NO OEIS sequence id
attached (oeis: ["N/A"]). The task this script belongs to asks for a
property derived from "its OEIS sequence id(s) and tags" -- there is no
sequence id here to derive anything from, and the data file used as the
source of truth carries no free-text statement of the problem beyond the
metadata block above. So this script cannot honestly claim to test a
specific OEIS-indexed term for #924. Per instructions for exactly this
situation, this is the best honest attempt: it builds a REAL, genuine
finite/computable problem drawn from the problem's own tags
("graph theory", "ramsey theory"), rather than fabricating or copying a
value that doesn't exist in source.

The classical property tested
------------------------------
Ramsey's theorem territory: does there exist a red/blue edge-colouring of
the complete graph K4 (4 vertices, C(4,2) = 6 edges) that contains NO
monochromatic triangle?

This is finite and small: 2^6 = 64 possible colourings of K4's 6 edges,
and K4 has exactly 4 triangles (one per left-out vertex) to check for
monochromaticity. (Contrast with Ramsey's theorem R(3,3) = 6, which says
every colouring of K6 DOES contain a monochromatic triangle -- K4 is
comfortably below that threshold, so triangle-free-in-both-colours
colourings must exist, and the script proves that from first principles
by brute force before ever touching a qubit.)

The classical answer (computed in this script, from first principles):
    We brute-force all 64 colourings of K4's 6 edges and mark exactly
    those with no monochromatic triangle among the 4 triangles of K4.
    That count and the explicit set of "good" colourings (as 6-bit
    integers) is the ground truth the quantum circuit is checked against.

The quantum circuit
--------------------
A genuine Grover search circuit (6 "edge-colour" qubits + ancilla) whose
oracle marks exactly the "no monochromatic triangle" colourings computed
above (via a triangle-monochromaticity-detector built from reversible
XOR/OR logic on 4 ancilla qubits, one per triangle, uncomputed after the
phase flip), amplified by the standard Grover diffuser, run on the ideal
AerSimulator with the classically-optimal number of iterations for the
known number of solutions.

The script prints PASS if the state(s) with the highest measured
probability after running the circuit are all members of the classically
computed "good" set, and FAIL otherwise.
"""

import itertools
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))          # 6 edges of K4
assert len(EDGES) == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))       # 4 triangles of K4
assert len(TRIANGLES) == 4


def triangle_edge_bits(triangle):
    """Indices (into EDGES) of the 3 edges making up this triangle."""
    a, b, c = triangle
    return [EDGE_INDEX[tuple(sorted((a, b)))],
            EDGE_INDEX[tuple(sorted((b, c)))],
            EDGE_INDEX[tuple(sorted((a, c)))]]


TRIANGLE_EDGE_BITS = [triangle_edge_bits(t) for t in TRIANGLES]


def is_good_colouring(bits):
    """bits: tuple of 6 ints (0/1), one per edge in EDGES order.
    A colouring is 'good' iff no triangle is monochromatic
    (i.e. not all 3 of its edges share the same colour bit)."""
    for e0, e1, e2 in TRIANGLE_EDGE_BITS:
        if bits[e0] == bits[e1] == bits[e2]:
            return False
    return True


GOOD_COLOURINGS = []
for combo in itertools.product([0, 1], repeat=6):
    if is_good_colouring(combo):
        # bit 0 (edges[0]) is the least-significant qubit in our encoding
        value = sum(b << i for i, b in enumerate(combo))
        GOOD_COLOURINGS.append(value)

GOOD_COLOURINGS = sorted(GOOD_COLOURINGS)
NUM_SOLUTIONS = len(GOOD_COLOURINGS)
SEARCH_SPACE_SIZE = 2 ** 6

print(f"Classical brute force over all {SEARCH_SPACE_SIZE} edge-colourings of K4:")
print(f"  colourings with no monochromatic triangle: {NUM_SOLUTIONS}")
print(f"  as 6-bit values: {GOOD_COLOURINGS}")
assert NUM_SOLUTIONS > 0, "Ramsey's theorem says these must exist below R(3,3)=6 vertices; sanity check failed"

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for a "good" colouring.
# ---------------------------------------------------------------------------

N_EDGE_QUBITS = 6
N_ANCILLA = 4          # one triangle-monochromaticity flag per triangle
N_OUT = 1               # final "all triangles OK" flag

edge = QuantumRegister(N_EDGE_QUBITS, "edge")
anc = QuantumRegister(N_ANCILLA, "tri_anc")
out = QuantumRegister(N_OUT, "out")
creg = ClassicalRegister(N_EDGE_QUBITS, "meas")

qc = QuantumCircuit(edge, anc, out, creg)

# uniform superposition over all 64 colourings
qc.h(edge)

# out qubit prepared in |-> for phase-kickback oracle
qc.x(out[0])
qc.h(out[0])


def build_triangle_detector(circuit, edge_reg, anc_qubit, e0, e1, e2):
    """Flip anc_qubit iff edge_reg[e0] == edge_reg[e1] == edge_reg[e2]
    (i.e. the triangle formed by these 3 edges is monochromatic).

    Detects "all three equal" = "all three are 0" OR "all three are 1".
    Implemented with two multi-controlled Toffolis (one per all-0 / all-1
    case), using X-gates to flip the all-0 case into an all-1 pattern
    for the mcx, then flipping back (fully reversible / uncomputed later).
    """
    # all-1 case (all three edge qubits = 1) -> flip ancilla
    circuit.mcx([edge_reg[e0], edge_reg[e1], edge_reg[e2]], anc_qubit)
    # all-0 case: temporarily invert the three edge qubits, then mcx, then invert back
    circuit.x([edge_reg[e0], edge_reg[e1], edge_reg[e2]])
    circuit.mcx([edge_reg[e0], edge_reg[e1], edge_reg[e2]], anc_qubit)
    circuit.x([edge_reg[e0], edge_reg[e1], edge_reg[e2]])


def oracle(circuit):
    # compute the 4 triangle-monochromaticity flags into ancillas
    for anc_i, (e0, e1, e2) in enumerate(TRIANGLE_EDGE_BITS):
        build_triangle_detector(circuit, edge, anc[anc_i], e0, e1, e2)

    # "good" colouring <=> ALL 4 ancillas are 0 (no monochromatic triangle)
    # flip out qubit iff all ancillas are 0: X on each ancilla, mcx, X back
    circuit.x(anc)
    circuit.mcx(list(anc), out[0])
    circuit.x(anc)

    # uncompute the triangle-detector ancillas (reverse order, same reversible ops)
    for anc_i, (e0, e1, e2) in enumerate(TRIANGLE_EDGE_BITS):
        build_triangle_detector(circuit, edge, anc[anc_i], e0, e1, e2)


def diffuser(circuit):
    circuit.h(edge)
    circuit.x(edge)
    circuit.h(edge[-1])
    circuit.mcx(list(edge[:-1]), edge[-1])
    circuit.h(edge[-1])
    circuit.x(edge)
    circuit.h(edge)


# optimal number of Grover iterations for N=64, M=NUM_SOLUTIONS solutions
theta = math.asin(math.sqrt(NUM_SOLUTIONS / SEARCH_SPACE_SIZE))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations} (N={SEARCH_SPACE_SIZE}, M={NUM_SOLUTIONS})")

for _ in range(iterations):
    oracle(qc)
    diffuser(qc)

# undo the |-> prep on out qubit before measuring (it should end back near |1>,
# left untouched/unmeasured; we only measure the edge register)
qc.h(out[0])
qc.x(out[0])

qc.measure(edge, creg)

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
SHOTS = 4096
result = sim.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical register bit order in the returned bitstring is
# reversed relative to qubit index order (creg[0] is the rightmost char).
def bitstring_to_value(bs):
    bits = bs[::-1]  # bits[i] corresponds to edge qubit i
    return int(bits, 2)

measured_values = {bitstring_to_value(bs): c for bs, c in counts.items()}
sorted_measured = sorted(measured_values.items(), key=lambda kv: -kv[1])

top_count = sorted_measured[0][1]
top_states = [v for v, c in sorted_measured if c == top_count]
# be a little lenient: take every state within 15% of the top peak as "found"
threshold = 0.85 * top_count
found_states = sorted({v for v, c in sorted_measured if c >= threshold})

print(f"Top measured colouring(s) (>= {threshold:.0f}/{SHOTS} shots): {found_states}")

all_top_are_good = all(v in GOOD_COLOURINGS for v in found_states)
found_at_least_one_good_with_high_prob = any(
    v in GOOD_COLOURINGS and c >= 0.5 * SHOTS / NUM_SOLUTIONS
    for v, c in measured_values.items()
)

verified = all_top_are_good and found_at_least_one_good_with_high_prob

if verified:
    print("PASS: Grover search's amplified state(s) match the classically "
          "computed 'no monochromatic triangle' colourings of K4.")
else:
    print("FAIL: Grover search's amplified state(s) do NOT match the "
          "classically computed answer.")

print(f"ran_ok=True verified_against_classical={verified}")
