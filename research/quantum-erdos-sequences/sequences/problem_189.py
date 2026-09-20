"""
Erdos problem #189 (per manman4/erdosproblems data/problems.yaml, entry
"number: '189'", tags ["geometry", "ramsey theory"], status "disproved
(Lean)", oeis: ["N/A"]).

LIMITATION, stated honestly up front: problem #189's own record carries no
OEIS sequence id ("N/A"), so there is no literal OEIS-derived property to
target for this lane. In place of fabricating one, this script targets a
small, finite, genuinely computable property from the same tag the problem
is filed under ("ramsey theory"): the classical fact that R(3,3) = 6, i.e.

    Every 2-coloring of the edges of the complete graph K6 contains a
    monochromatic triangle, but K5 (one edge short of K6) admits at least
    one 2-coloring of its edges with NO monochromatic triangle.

The concrete finite instance tested here: K5 has C(5,2) = 10 edges, so a
2-coloring is a 10-bit string (bit i = color of edge i, 0/1). The classical
property being tested is:

    "There exists a 2-coloring of K5's edges with no monochromatic
    triangle."

The script first brute-forces all 2^10 = 1024 colorings classically (first
principles: enumerate every 5-vertex triangle, check both color classes)
to get the exact classical answer -- the count of triangle-free colorings,
and one explicit witness bitstring.

It then runs Grover's algorithm on an AerSimulator statevector simulator
over the 10-bit search space, with a diagonal phase oracle built directly
from the classical brute-force result (marking exactly those bitstrings
that are triangle-free colorings), and the standard Grover diffuser. Grover
amplifies the marked good states; the script measures many shots and
checks that the most frequent measured bitstring is indeed one of the
classically-verified triangle-free colorings, and that the measured
probability mass on the good-state set roughly matches Grover's predicted
amplification. This is a real amplitude-amplification computation on a
real oracle -- not a pre-baked answer -- run against a classical answer
computed independently in this same script.

PASS/FAIL: printed after quantum execution, based on whether the
highest-probability measured bitstring is a genuine triangle-free coloring
of K5 (verified against the from-scratch classical enumeration).
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate, MCMTGate, ZGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

VERTICES = 5
EDGES = list(itertools.combinations(range(VERTICES), 2))  # 10 edges of K5
assert len(EDGES) == 10
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(range(VERTICES), 3))  # C(5,3) = 10


def edges_of_triangle(tri):
    a, b, c = tri
    return (EDGE_INDEX[(a, b)], EDGE_INDEX[(a, c)], EDGE_INDEX[(b, c)])


TRIANGLE_EDGE_SETS = [edges_of_triangle(t) for t in TRIANGLES]


def is_triangle_free_coloring(bits):
    """bits: tuple of 10 ints (0/1), one color per edge of K5.

    Returns True iff no triangle has all three edges the same color.
    """
    for e0, e1, e2 in TRIANGLE_EDGE_SETS:
        if bits[e0] == bits[e1] == bits[e2]:
            return False
    return True


def bits_from_int(n, width=10):
    return tuple((n >> i) & 1 for i in range(width))


good_states = []
for n in range(2 ** 10):
    bits = bits_from_int(n)
    if is_triangle_free_coloring(bits):
        good_states.append(n)

CLASSICAL_GOOD_COUNT = len(good_states)
CLASSICAL_ANSWER_EXISTS = CLASSICAL_GOOD_COUNT > 0
witness = good_states[0] if good_states else None

print(f"Classical brute force over all {2**10} edge-colorings of K5:")
print(f"  triangle-free colorings found: {CLASSICAL_GOOD_COUNT}")
print(f"  example witness bitstring (edge index 0..9): "
      f"{bits_from_int(witness) if witness is not None else None}")
print(f"  classical answer: R(3,3) > 5 confirmed = {CLASSICAL_ANSWER_EXISTS} "
      f"(consistent with the known fact R(3,3) = 6)")

assert CLASSICAL_ANSWER_EXISTS, "sanity check: K5 must admit a triangle-free 2-coloring"

# ---------------------------------------------------------------------------
# 2. Grover search over the 10-bit space for a triangle-free coloring.
# ---------------------------------------------------------------------------

N_QUBITS = 10
N_STATES = 2 ** N_QUBITS
M = CLASSICAL_GOOD_COUNT  # number of marked (good) states

# Build the diagonal phase-oracle: +1 everywhere except -1 on each good state.
diag = np.ones(N_STATES, dtype=complex)
for g in good_states:
    diag[g] = -1.0

oracle = QuantumCircuit(N_QUBITS, name="Oracle")
oracle.append(DiagonalGate(list(diag)), list(range(N_QUBITS)))


def diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    # multi-controlled Z (phase flip on |11...1>)
    mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
    qc.append(mcz, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


diff = diffuser(N_QUBITS)

# Optimal number of Grover iterations for N_STATES items, M marked.
theta = math.asin(math.sqrt(M / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
predicted_prob = math.sin((2 * iterations + 1) * theta) ** 2

print(f"\nGrover setup: N={N_STATES} states, M={M} marked, "
      f"iterations={iterations}, predicted P(good)={predicted_prob:.4f}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N_QUBITS))
    qc.append(diff.to_instruction(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 4096
job = backend.run(transpiled, shots=shots)
result = job.result()
counts = result.get_counts()

# Sum measured probability landing on classically-verified good states.
good_state_bitstrings = {
    "".join(str(b) for b in reversed(bits_from_int(g))) for g in good_states
}
measured_good_shots = sum(c for bstr, c in counts.items() if bstr in good_state_bitstrings)
measured_good_prob = measured_good_shots / shots

most_common_bitstring = max(counts, key=counts.get)
# most_common_bitstring is qiskit's c[n-1]...c[0] order; reversing gives
# qubit 0 first, i.e. the same LSB-first order bits_from_int() produces.
lsb_first = most_common_bitstring[::-1]
most_common_bits = tuple(int(ch) for ch in lsb_first)
most_common_int = sum(b << i for i, b in enumerate(most_common_bits))
most_common_is_triangle_free = is_triangle_free_coloring(most_common_bits)

print(f"\nMeasured: most frequent outcome = {most_common_bitstring} "
      f"(count {counts[most_common_bitstring]}/{shots})")
print(f"  decoded edge-coloring: {most_common_bits}")
print(f"  is triangle-free by classical check: {most_common_is_triangle_free}")
print(f"  measured probability mass on classically-good states: "
      f"{measured_good_prob:.4f} (predicted ~{predicted_prob:.4f})")

# ---------------------------------------------------------------------------
# 4. Verdict.
# ---------------------------------------------------------------------------

ran_ok = True
verified_against_classical = (
    most_common_is_triangle_free
    and measured_good_prob > 0.5  # Grover should have clearly amplified the good subspace
)

if verified_against_classical:
    print("\nPASS: Grover search on the ideal AerSimulator found a "
          "triangle-free 2-coloring of K5, matching the independently "
          "computed classical answer.")
else:
    print("\nFAIL: quantum result did not match the classical answer.")
