"""
Erdos problem #556 -- quantum-testable instance.

OEIS sequence used: A389335.
  a(n) = the smallest m such that every red/green/blue edge-coloring of the
  complete graph K_m contains a monochromatic n-cycle (the 3-color Ramsey
  number r(C_n, C_n, C_n) for cycles). Problem tags: graph theory, Ramsey
  theory. A389335(3) = 17: the 3-color Ramsey number for triangles is 17.

Why this script does NOT build a circuit on K_17 / 3 colors directly:
  K_17 with 3 colors has C(17,2) = 136 edges, each needing 2 qubits to encode
  one of 3 colors -> 272 qubits, and the "no monochromatic triangle" oracle
  would need to check C(17,3) = 680 triangles. That is not a small circuit
  (nowhere near N <= ~64 / "few qubits"), so it cannot honestly be simulated
  here. Instead this script tests a genuine, smaller instance of the exact
  same combinatorial property that A389335 generalizes: the classical
  (2-color) triangle Ramsey fact R(3,3) = 6, applied to K_5 (which is why
  K_6 forces a monochromatic triangle under 2 colors but K_5 does not).

Classical property tested (computed from first principles below, not looked
up): among all 2-colorings of the edges of K_5 (5 vertices, C(5,2) = 10
edges, so 2^10 = 1024 total colorings), how many avoid a monochromatic
triangle, and is at least one such coloring found by Grover search? This is
finite, exactly computable, and small enough for a real qubit budget
(10 "edge" qubits + ancillas for the diffusion step).

The script:
  1. Classically enumerates all 1024 edge-colorings of K5, checks each of the
     C(5,3) = 10 triangles, and determines the exact set of "good" colorings
     (no monochromatic triangle). This is the ground truth.
  2. Builds a Grover search circuit over 10 qubits (one per edge) on the
     ideal AerSimulator: a diagonal phase oracle built directly from the
     classically-computed good/bad labeling (so the oracle is provably
     correct by construction) plus the standard Grover diffusion operator,
     iterated the optimal number of times for the known number of marked
     states M.
  3. Measures the circuit and checks that the measured bitstrings are
     overwhelmingly good colorings (i.e. Grover actually amplified the
     valid, monochromatic-triangle-free colorings), and cross-checks this
     against the classical enumeration.
  4. Prints PASS if grover-search's top outcomes match the classically
     verified "no monochromatic triangle" colorings with high probability
     (far above the uniform baseline of M/1024), else FAIL.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Diagonal
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: 2-colorings of K5 with no monochromatic triangle
# ---------------------------------------------------------------------------

N_VERTICES = 5
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 10 edges
TRIANGLES = list(itertools.combinations(range(N_VERTICES), 3))  # 10 triangles
NUM_EDGES = len(EDGES)
assert NUM_EDGES == 10
N_STATES = 2 ** NUM_EDGES  # 1024


def is_good_coloring(bits):
    """bits[i] in {0,1} is the color of EDGES[i]. True iff no triangle is
    monochromatic (all three of its edges the same color)."""
    color = {EDGES[i]: bits[i] for i in range(NUM_EDGES)}
    for (a, b, c) in TRIANGLES:
        e_ab = color[tuple(sorted((a, b)))]
        e_bc = color[tuple(sorted((b, c)))]
        e_ac = color[tuple(sorted((a, c)))]
        if e_ab == e_bc == e_ac:
            return False
    return True


good_states = []
for x in range(N_STATES):
    bits = [(x >> i) & 1 for i in range(NUM_EDGES)]
    if is_good_coloring(bits):
        good_states.append(x)

M = len(good_states)
print(f"Classical enumeration: {M} of {N_STATES} 2-colorings of K5 avoid a "
      f"monochromatic triangle (this witnesses R(3,3) > 5, i.e. R(3,3) >= 6, "
      f"the classical fact underlying the OEIS A389335 family for problem "
      f"556).")
assert M > 0, "classical search found no triangle-free 2-coloring of K5"

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classical ground truth
# ---------------------------------------------------------------------------

diagonal_phases = [1.0] * N_STATES
for x in good_states:
    diagonal_phases[x] = -1.0

oracle_gate = Diagonal(diagonal_phases)


def diffusion_operator(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffusion")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


num_qubits = NUM_EDGES
iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M)))
print(f"Grover: N={N_STATES}, M={M}, using {iterations} iteration(s).")

qc = QuantumCircuit(num_qubits, num_qubits)
qc.h(range(num_qubits))

diff_gate = diffusion_operator(num_qubits).to_gate()

for _ in range(iterations):
    qc.append(oracle_gate, range(num_qubits))
    qc.append(diff_gate, range(num_qubits))

qc.measure(range(num_qubits), range(num_qubits))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 4096
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order: rightmost char of the bitstring is qubit 0.
def bitstring_to_x(bs):
    bits = [int(c) for c in reversed(bs)]
    x = 0
    for i, b in enumerate(bits):
        x |= (b << i)
    return x

measured = {bitstring_to_x(bs): c for bs, c in counts.items()}

good_state_set = set(good_states)
good_shots = sum(c for x, c in measured.items() if x in good_state_set)
prob_good = good_shots / shots
baseline = M / N_STATES

print(f"Fraction of shots landing on a classically-verified triangle-free "
      f"coloring: {prob_good:.4f} (uniform-random baseline would be "
      f"{baseline:.4f}).")

top_x = max(measured, key=measured.get)
top_bits = [(top_x >> i) & 1 for i in range(NUM_EDGES)]
top_is_good = is_good_coloring(top_bits)
print(f"Most frequent measured outcome (edge-coloring index {top_x}) is "
      f"{'a valid' if top_is_good else 'NOT a valid'} triangle-free coloring "
      f"of K5.")

# Grover succeeds if it clearly amplified the marked (good) subspace above
# the classical baseline, and the single most likely outcome is itself a
# verified-good coloring.
verified_against_classical = top_is_good and (prob_good > baseline * 1.5)

if verified_against_classical:
    print("PASS")
else:
    print("FAIL")
