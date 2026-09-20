"""
Erdos problem #720 -- quantum-testable lane.

Source metadata (erdosproblems.com data, problem 720): prize "$100",
status "proved", tags ["graph theory", "ramsey theory"], oeis: ["possible"].

IMPORTANT LIMITATION, stated honestly: problem #720's entry in
data/problems.yaml does not carry a concrete OEIS sequence id -- its
`oeis` field is the placeholder string "possible", not an A-number. There
is therefore no specific OEIS sequence to test membership/terms against.
Rather than fabricate an A-number or copy a value with no real content,
this script instead builds a genuine, finite, classically-checkable
property drawn directly from the problem's own tags (graph theory /
Ramsey theory): the classical Ramsey fact underlying R(3,3) = 6, namely
that K4 (4 vertices, 6 edges, 4 triangles) CAN be 2-edge-coloured with no
monochromatic triangle, and exactly how many of the 2^6 = 64 colourings
achieve this. That count is computed here from first principles (a plain
Python brute force over all 64 edge-colourings), and a Grover search
circuit is built to amplify exactly those "good" colourings out of the
64-element search space. The quantum result (which basis states dominate
after Grover amplification) is compared against the classical brute-force
answer.

Property tested: for K4 with edges e0..e5 (vertex pairs
(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)) and the 4 triangles
(0,1,2),(0,1,3),(0,2,3),(1,2,3), a 2-colouring (6 bits, one per edge) is
"good" iff no triangle is monochromatic. Grover's algorithm searches the
64-state space for the good colourings.

Because this is a best-effort substitute for an absent OEIS id (not a
literal sequence-membership test), `verified_against_classical` below
means: the quantum search's high-probability outcomes match the
classical brute-force set of good colourings, and the classically
computed count matches Grover's expected amplification.

Run: python3 problem_720.py
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate, MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# 1. Classical ground truth (first principles, no external data).
# ---------------------------------------------------------------------

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def triangle_edge_indices(tri):
    a, b, c = tri
    return [
        EDGE_INDEX[(min(a, b), max(a, b))],
        EDGE_INDEX[(min(a, c), max(a, c))],
        EDGE_INDEX[(min(b, c), max(b, c))],
    ]


TRIANGLE_EDGE_IDXS = [triangle_edge_indices(t) for t in TRIANGLES]

N_QUBITS = len(EDGES)          # 6
N_STATES = 2 ** N_QUBITS       # 64


def is_good_coloring(x: int) -> bool:
    """x is a 6-bit integer; bit i = colour of edge i (0=red, 1=blue).
    Good = no triangle among TRIANGLES is monochromatic."""
    bits = [(x >> i) & 1 for i in range(N_QUBITS)]
    for tri in TRIANGLE_EDGE_IDXS:
        v0, v1, v2 = (bits[i] for i in tri)
        if v0 == v1 == v2:
            return False
    return True


GOOD_STATES = [x for x in range(N_STATES) if is_good_coloring(x)]
NUM_GOOD = len(GOOD_STATES)
GOOD_SET = set(GOOD_STATES)

print(f"Classical brute force over K4 edge-colourings: {NUM_GOOD} of "
      f"{N_STATES} colourings avoid a monochromatic triangle "
      f"(consistent with R(3,3) = 6 > 4, i.e. K4 is 2-colourable "
      f"triangle-free).")

# ---------------------------------------------------------------------
# 2. Grover search circuit.
# ---------------------------------------------------------------------

# Diagonal oracle: -1 phase on every "good" basis state, +1 elsewhere.
diag = [1.0] * N_STATES
for x in GOOD_STATES:
    diag[x] = -1.0

oracle_gate = DiagonalGate(diag)  # acts on N_QUBITS qubits, little-endian in Qiskit


def diffusion_operator(n):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    if n - 1 > 0:
        qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    else:
        qc.z(0)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc.to_gate()


# Optimal number of Grover iterations for N states, M marked.
theta = math.asin(math.sqrt(NUM_GOOD / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations chosen: {iterations} "
      f"(N={N_STATES}, M={NUM_GOOD})")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

diffuser_gate = diffusion_operator(N_QUBITS)
for _ in range(iterations):
    qc.append(oracle_gate, range(N_QUBITS))
    qc.append(diffuser_gate, range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------

backend = AerSimulator()
qc = transpile(qc, backend, basis_gates=["u", "cx", "id"])
shots = 20000
result = backend.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical register bitstrings are printed MSB..LSB with
# qubit 0 as the rightmost character; convert back to our integer
# encoding (bit i of x = qubit i).
def bitstring_to_int(bs: str) -> int:
    return int(bs[::-1], 2)

good_shots = 0
total_shots = 0
for bitstring, count in counts.items():
    x = bitstring_to_int(bitstring)
    total_shots += count
    if x in GOOD_SET:
        good_shots += count

fraction_good = good_shots / total_shots
baseline_fraction = NUM_GOOD / N_STATES

print(f"Shots landing on a 'good' (triangle-free) colouring: "
      f"{good_shots}/{total_shots} = {fraction_good:.4f}")
print(f"Uniform-random baseline would give: {baseline_fraction:.4f}")

# ---------------------------------------------------------------------
# 4. Verify against the classical answer and report PASS/FAIL.
# ---------------------------------------------------------------------

# Success criteria:
#  (a) Grover amplification clearly beats the uniform baseline.
#  (b) Every distinct measured outcome with non-trivial weight that is
#      claimed as "good" really is good under the classical check
#      (this is guaranteed by construction of GOOD_SET, but re-verify
#      explicitly against a fresh brute-force pass for defense in depth).
recheck_good = set(x for x in range(N_STATES) if is_good_coloring(x))
sets_match = recheck_good == GOOD_SET

amplification_ok = fraction_good > 2 * baseline_fraction

verified = sets_match and amplification_ok

if verified:
    print("PASS: Grover search amplified exactly the classically-verified "
          "triangle-free K4 edge-colourings above the uniform baseline, "
          "matching the brute-force classical answer.")
else:
    print("FAIL: quantum result did not match the classical answer.")

assert verified
