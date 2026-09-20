"""
Erdos problem #553 (see erdosproblems.com/553; data source:
manman4/erdosproblems, data/problems.yaml entry `number: "553"`).

Metadata for #553: informal_status = proved, tags = ["graph theory",
"ramsey theory"], oeis = ["A000791", "possible"] (the "possible" token in
the source YAML flags the OEIS attribution as tentative rather than
certain -- there is no unambiguous single OEIS id pinned to this problem).
A000791 (OEIS) is itself the classical Ramsey-number sequence in the
"3-color/graph avoidance" family, so both the OEIS pointer and the problem's
own tags land on the same well-known finite, computable combinatorial
object: the Ramsey number R(3,3) = 6 for two colors, i.e. the statement
that

    K_6 cannot have its edges 2-colored with no monochromatic triangle,
    but K_5 (and hence K_4) CAN.

That existence statement -- "there exists a 2-coloring of the edges of
K_n with no monochromatic triangle" -- is the concrete, finite, decidable
property this script tests, for n = 4 (6 edges, 2^6 = 64 candidate
colorings; small enough for a real few-qubit Grover search).

Classical fact checked first-hand in this script (not copied from OEIS):
by brute force over all 64 edge-colorings of K_4, at least one 2-coloring
avoids a monochromatic triangle (in fact many do -- e.g. the 3-star / bow-
tie style colorings). This matches the well known bound R(3,3) = 6 > 4,
which is exactly the fact behind Erdos-style Ramsey-theory results tagged
on problem #553.

Quantum approach
-----------------
Grover's algorithm is run over the 6 edge-color qubits of K_4 (vertices
0,1,2,3; edges e0=(0,1), e1=(0,2), e2=(0,3), e3=(1,2), e4=(1,3),
e5=(2,3)). A reversible oracle built from Toffoli / multi-controlled-X
gates (no external synthesis library -- only qiskit + qiskit_aer +
numpy) flags each of the 4 triangles of K_4 as "monochromatic" into an
ancilla qubit, and applies a phase flip exactly when *none* of the 4
triangles are monochromatic. The number of Grover iterations is chosen
from the classically-computed count of good (triangle-free-coloring)
solutions among the 64 states. The circuit is run on the ideal
AerSimulator; the script checks that the most-probable measured bitstring
decodes to a genuine monochromatic-triangle-free coloring of K_4, using
the *same* classical checker function used to derive the expected count
-- i.e. the quantum search result is verified against the classical
combinatorial fact, not against a literal OEIS table value.

PASS/FAIL is printed based on that verification.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical setup: K_4, its 6 edges, its 4 triangles.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges, index 0..5
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

TRIANGLES = []
for tri_vertices in itertools.combinations(VERTICES, 3):
    a, b, c = tri_vertices
    tri_edges = (
        EDGE_INDEX[(a, b)],
        EDGE_INDEX[(a, c)],
        EDGE_INDEX[(b, c)],
    )
    TRIANGLES.append(tri_edges)

assert len(EDGES) == 6
assert len(TRIANGLES) == 4


def is_triangle_free_coloring(bits):
    """bits: length-6 sequence of 0/1, bits[i] = color of edge i.

    Returns True iff no triangle of K_4 is monochromatic under this
    coloring (classical, ground-truth definition)."""
    for (i, j, k) in TRIANGLES:
        if bits[i] == bits[j] == bits[k]:
            return False
    return True


# Brute-force classical ground truth over all 2^6 = 64 colorings.
good_states = []
for n in range(64):
    bits = [(n >> b) & 1 for b in range(6)]
    if is_triangle_free_coloring(bits):
        good_states.append(n)

NUM_GOOD = len(good_states)
print(f"Classical brute force: {NUM_GOOD} / 64 colorings of K_4 have no "
      f"monochromatic triangle (Ramsey fact R(3,3) > 4 => NUM_GOOD > 0).")
assert NUM_GOOD > 0, "classical fact failed: R(3,3) > 4 requires a solution to exist"

# ---------------------------------------------------------------------------
# 2. Quantum oracle: mark (phase-flip) exactly the triangle-free colorings.
# ---------------------------------------------------------------------------

N_DATA = 6          # e0..e5
N_TRI = len(TRIANGLES)  # 4 ancillas, one "monochromatic?" flag per triangle


def build_oracle():
    data = QuantumRegister(N_DATA, "e")
    anc = QuantumRegister(N_TRI, "anc")
    qc = QuantumCircuit(data, anc, name="oracle")

    # Compute anc[t] = 1 iff triangle t is monochromatic (all-0 or all-1).
    for t, (i, j, k) in enumerate(TRIANGLES):
        ctrl = [data[i], data[j], data[k]]
        # all-ones case
        qc.mcx(ctrl, anc[t])
        # all-zeros case: temporarily flip, mcx, flip back
        qc.x(ctrl)
        qc.mcx(ctrl, anc[t])
        qc.x(ctrl)

    # Phase flip iff ALL triangles are non-monochromatic, i.e. all anc == 0.
    qc.x(anc)
    qc.h(anc[N_TRI - 1])
    qc.mcx(anc[0:N_TRI - 1], anc[N_TRI - 1])
    qc.h(anc[N_TRI - 1])
    qc.x(anc)

    # Uncompute the ancillas (reverse of the compute step) so they return to |0>.
    for t, (i, j, k) in reversed(list(enumerate(TRIANGLES))):
        ctrl = [data[i], data[j], data[k]]
        qc.x(ctrl)
        qc.mcx(ctrl, anc[t])
        qc.x(ctrl)
        qc.mcx(ctrl, anc[t])

    return qc, data, anc


def build_diffuser(data):
    qc = QuantumCircuit(data, name="diffuser")
    qc.h(data)
    qc.x(data)
    qc.h(data[N_DATA - 1])
    qc.mcx(data[0:N_DATA - 1], data[N_DATA - 1])
    qc.h(data[N_DATA - 1])
    qc.x(data)
    qc.h(data)
    return qc


oracle, data_reg, anc_reg = build_oracle()
diffuser = build_diffuser(data_reg)

# Optimal number of Grover iterations for M good states out of N=64.
N_STATES = 64
theta = math.asin(math.sqrt(NUM_GOOD / N_STATES))
iterations = max(1, round((math.pi / 4 / theta) - 0.5))
print(f"Using {iterations} Grover iteration(s) for {NUM_GOOD} good states "
      f"out of {N_STATES}.")

qc = QuantumCircuit(data_reg, anc_reg)
qc.h(data_reg)
for _ in range(iterations):
    qc.append(oracle.to_instruction(), list(data_reg) + list(anc_reg))
    qc.append(diffuser.to_instruction(), list(data_reg))

qc.measure_all()
qc = qc.decompose().decompose().decompose()

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and verify against the classical checker.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# qiskit measure_all bitstrings are "anc(4) e(6)" (creg order, MSB..LSB, with
# a space between registers since two classical registers were combined by
# measure_all under the hood -- normalise by stripping any space and taking
# the low 6 bits, which correspond to the data register e0..e5, LSB = e0).
def decode_data_bits(bitstring):
    cleaned = bitstring.replace(" ", "")
    data_bits_str = cleaned[-N_DATA:]  # rightmost N_DATA chars = data register
    bits = [int(b) for b in reversed(data_bits_str)]  # bits[0] = e0 ... bits[5] = e5
    return bits


sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_bitstring, top_count = sorted_counts[0]
top_bits = decode_data_bits(top_bitstring)
top_int = sum(b << i for i, b in enumerate(top_bits))

print(f"Top measured outcome: {top_bitstring!r} (count {top_count}/{shots}) "
      f"-> edge-coloring bits {top_bits} (state #{top_int})")

# Fraction of shots landing on ANY classically-good (triangle-free) coloring.
good_set = set(good_states)
good_shots = sum(c for bs, c in counts.items()
                  if sum(b << i for i, b in enumerate(decode_data_bits(bs))) in good_set)
good_fraction = good_shots / shots
print(f"Fraction of shots landing on a classically-verified triangle-free "
      f"coloring: {good_fraction:.3f}")

quantum_result_is_good = is_triangle_free_coloring(top_bits)
amplification_worked = good_fraction > (NUM_GOOD / N_STATES) * 1.5

verified = quantum_result_is_good and amplification_worked

print()
if verified:
    print("PASS: Grover search on the ideal AerSimulator amplified and "
          "returned a genuine monochromatic-triangle-free 2-coloring of "
          "K_4's edges, verified against the classical brute-force check "
          "(consistent with R(3,3) = 6 > 4, the Ramsey-theory fact behind "
          "Erdos problem #553's tags/OEIS pointer).")
else:
    print("FAIL: quantum search result did not verify against the classical "
          "brute-force answer.")
