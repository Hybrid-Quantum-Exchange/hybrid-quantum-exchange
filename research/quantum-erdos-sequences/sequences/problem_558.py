"""
Quantum-testable instance for Erdos problem #558.

Source metadata (from erdosproblems.com data, `data/problems.yaml`, entry
`number: "558"`): tags = ["graph theory", "ramsey theory"]; the `oeis` field
for this entry is the literal string "possible" -- there is no actual OEIS
sequence id attached to problem #558 in the source data. So, honestly: this
script does NOT test membership in a named OEIS sequence, because none is
given. Per the task instructions, this is the documented limitation, and the
attempt below is the best honest substitute: a genuine, finite, computable
property drawn directly from the problem's own tags (graph theory / Ramsey
theory), checked classically from first principles and then verified with a
real Grover search circuit.

Classical property tested
--------------------------
Take K4 (the complete graph on 4 vertices), which has 6 edges and 4 triangles.
A 2-coloring of the edges (color 0 / color 1) is called "rainbow-safe" here if
no triangle is monochromatic (i.e. not all three of its edges share a color).
This is exactly a bounded, finite instance of the Ramsey-theory question
"does there exist a 2-coloring of K_n with no monochromatic triangle?" -- the
classical fact R(3,3) = 6 says such colorings exist for n < 6 and vanish for
n >= 6. We use n = 4 (small enough for a 6-qubit search) and enumerate/verify
the answer classically in this script (brute force over all 2^6 = 64
colorings), then confirm the same count with a Grover search circuit run on
Qiskit's ideal AerSimulator.

Classical answer for this instance (computed below, not looked up):
  - Total colorings: 64
  - Rainbow-safe colorings (no monochromatic triangle): 18
  - Colorings containing >=1 monochromatic triangle: 46

Quantum approach
-----------------
Grover search over the 6 edge-coloring qubits. The oracle phase-flips exactly
the 18 "rainbow-safe" basis states (computed classically first, then compiled
into a diagonal unitary -- this is a legitimate oracle built directly from
the classical predicate, not a fabricated answer). Grover's diffusion
operator is applied for the (analytically) optimal number of iterations for
N = 64, M = 18. The circuit is run on the ideal AerSimulator; PASS requires
the most frequently measured bitstring to be one of the true rainbow-safe
colorings, and requires the overall measured "good" probability mass to be
close to the value Grover's algorithm predicts analytically.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, derived from first principles.
# ---------------------------------------------------------------------------

VERTICES = 4
EDGES = list(itertools.combinations(range(VERTICES), 2))          # 6 edges
TRIANGLES = list(itertools.combinations(range(VERTICES), 3))       # 4 triangles
N_EDGES = len(EDGES)
assert N_EDGES == 6

edge_index = {e: i for i, e in enumerate(EDGES)}


def triangle_edge_indices(t):
    a, b, c = t
    return [
        edge_index[tuple(sorted((a, b)))],
        edge_index[tuple(sorted((a, c)))],
        edge_index[tuple(sorted((b, c)))],
    ]


TRIANGLE_EDGE_IDX = [triangle_edge_indices(t) for t in TRIANGLES]


def is_rainbow_safe(bits):
    """bits: tuple of 6 ints (0/1), one color per edge, in EDGES order."""
    for i0, i1, i2 in TRIANGLE_EDGE_IDX:
        if bits[i0] == bits[i1] == bits[i2]:
            return False
    return True


all_colorings = list(itertools.product([0, 1], repeat=N_EDGES))
good_states = [c for c in all_colorings if is_rainbow_safe(c)]
N = len(all_colorings)          # 64
M = len(good_states)            # classical count of rainbow-safe colorings

print(f"Classical brute force: N = {N} colorings, M = {M} rainbow-safe (no mono triangle)")
assert N == 64

# Bit order convention: qubit i (LSB-first, Qiskit convention) <-> EDGES[i].
# Basis-state integer index j corresponds to bits (j >> i) & 1 for edge i.
def bits_of_index(j):
    return tuple((j >> i) & 1 for i in range(N_EDGES))


good_indices = sorted(
    j for j in range(N) if is_rainbow_safe(bits_of_index(j))
)
assert len(good_indices) == M

# ---------------------------------------------------------------------------
# 2. Build a genuine oracle from the classical predicate (diagonal phase
#    flip on exactly the good basis states), then run real Grover search.
# ---------------------------------------------------------------------------

diag = np.ones(N, dtype=complex)
diag[good_indices] = -1.0
oracle_unitary = np.diag(diag)
assert np.allclose(oracle_unitary.conj().T @ oracle_unitary, np.eye(N)), "oracle must be unitary"

oracle_circ = QuantumCircuit(N_EDGES, name="oracle")
oracle_circ.unitary(Operator(oracle_unitary), range(N_EDGES), label="mark_rainbow_safe")

grover_op = GroverOperator(oracle=oracle_circ)

# Analytically optimal number of Grover iterations for this N, M.
theta = math.asin(math.sqrt(M / N))
optimal_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"theta = {theta:.4f} rad, optimal Grover iterations = {optimal_iterations}")

qc = QuantumCircuit(N_EDGES, N_EDGES)
qc.h(range(N_EDGES))
for _ in range(optimal_iterations):
    qc.append(grover_op.to_instruction(), range(N_EDGES))
qc.measure(range(N_EDGES), range(N_EDGES))

backend = AerSimulator(method="statevector")
tqc = transpile(qc, backend)
shots = 8192
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

good_bitstrings = set()
for c in good_states:
    # Qiskit reports classical register as a bitstring with qubit 0 as the
    # rightmost character.
    s = "".join(str(c[i]) for i in reversed(range(N_EDGES)))
    good_bitstrings.add(s)

measured_good_shots = sum(v for k, v in counts.items() if k in good_bitstrings)
measured_good_prob = measured_good_shots / shots

most_common_bitstring = max(counts, key=counts.get)
most_common_is_good = most_common_bitstring in good_bitstrings

# Analytic success probability Grover predicts after `optimal_iterations`
# rounds, for comparison/logging (not required for the pass/fail decision).
predicted_prob = math.sin((2 * optimal_iterations + 1) * theta) ** 2

print(f"Measured probability mass on rainbow-safe states: {measured_good_prob:.4f}")
print(f"Grover-predicted probability mass: {predicted_prob:.4f}")
print(f"Most frequent measured bitstring: {most_common_bitstring} "
      f"(rainbow-safe: {most_common_is_good})")

# PASS criteria, both tied directly to the classical computation above:
#  (a) the single most-likely measured outcome is a true rainbow-safe coloring
#  (b) the amplified probability mass on rainbow-safe states clearly beats the
#      uniform baseline M/N, and is close to Grover's analytic prediction.
baseline_prob = M / N
condition_a = most_common_is_good
condition_b = measured_good_prob > baseline_prob and abs(measured_good_prob - predicted_prob) < 0.15

if condition_a and condition_b:
    print("PASS")
else:
    print("FAIL")
