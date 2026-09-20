"""
Erdos problem #563 (erdosproblems.com / manman4/erdosproblems data/problems.yaml).

Metadata found in the source-of-truth YAML for problem 563:
    oeis: ["N/A"]
    tags: ["graph theory", "ramsey theory", "hypergraphs"]
    informal_status: open

LIMITATION, stated honestly up front: problem 563 has NO associated OEIS
sequence id ("N/A"). There is therefore no OEIS sequence to build a
"quantum-testable sequence membership/term" circuit around, as the task
template assumes. Rather than fabricate an OEIS id or invent an unrelated
finite property with no connection to the problem, this script builds a
genuine, small, finite, classically-checkable computational problem drawn
directly from the problem's own tags ("graph theory", "ramsey theory") and
tests it with a real Grover search circuit on the ideal AerSimulator.

The classical property tested
------------------------------
Ramsey theory on the complete graph K4 (4 vertices, C(4,2) = 6 edges).
Each 2-coloring of the 6 edges of K4 (colors: 0 = red, 1 = blue) is encoded
as a 6-bit string, one bit per edge, in a fixed order. K4 has exactly
C(4,3) = 4 triangles. A coloring is "triangle-free" (good) if none of the
4 triangles is monochromatic (all 3 of its edges the same color).

Since R(3,3) = 6, K4 (only 4 vertices) is small enough that triangle-free
2-colorings are known to exist -- this script does NOT take that fact on
faith, it enumerates all 2^6 = 64 colorings in plain Python and evaluates
the 4 triangle constraints directly, first-principles, to get the exact
set of "good" (monochromatic-triangle-free) colorings and their count.

The quantum circuit
--------------------
A 6-qubit Grover search circuit is built. The oracle is a diagonal phase
oracle whose signs are computed directly from the same classical predicate
used above (one -1 phase per "good" basis state, +1 otherwise) -- i.e. the
oracle is derived from, and checked against, the classical enumeration in
this same script, not hand-picked. The standard Grover diffusion operator
is applied for the optimal integer number of iterations for this search
space/marked-count, and the circuit is measured 4096 shots on the ideal
AerSimulator (Qiskit Aer, statevector-exact simulation, no noise model).

PASS/FAIL criterion
--------------------
The script checks that Grover search amplifies the "good" (triangle-free)
colorings: PASS if the most-probable measured bitstring is one of the
classically-verified triangle-free colorings AND the total measured
probability mass on triangle-free colorings substantially exceeds the
uniform-random baseline (|good| / 64).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal, GroverOperator
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges of K4
assert len(EDGES) == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles of K4
assert len(TRIANGLES) == 4


def triangle_edges(tri):
    a, b, c = tri
    return [EDGE_INDEX[tuple(sorted((a, b)))],
            EDGE_INDEX[tuple(sorted((b, c)))],
            EDGE_INDEX[tuple(sorted((a, c)))]]


TRIANGLE_EDGE_IDX = [triangle_edges(t) for t in TRIANGLES]


def is_triangle_free(bits):
    """bits: tuple/list of 6 ints (0/1), one per edge in EDGES order."""
    for idx in TRIANGLE_EDGE_IDX:
        colors = [bits[i] for i in idx]
        if colors[0] == colors[1] == colors[2]:
            return False
    return True


N = 6  # number of edges/qubits
NSTATES = 2 ** N

good_states = []  # basis-state indices (integers) with no mono triangle
for k in range(NSTATES):
    bits = tuple((k >> i) & 1 for i in range(N))  # bit i <-> EDGES[i]
    if is_triangle_free(bits):
        good_states.append(k)

num_good = len(good_states)
assert num_good > 0, "expected triangle-free 2-colorings of K4 to exist"
print(f"Classical enumeration: {num_good} / {NSTATES} colorings of K4's "
      f"6 edges have no monochromatic triangle.")

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle directly from the classical predicate above.
# ---------------------------------------------------------------------------

diag = np.ones(NSTATES, dtype=complex)
for k in good_states:
    diag[k] = -1.0
oracle = Diagonal(list(diag))
oracle.name = "TriFreeOracle"

grover_op = GroverOperator(oracle)

# Optimal number of Grover iterations for this search space.
theta = math.asin(math.sqrt(num_good / NSTATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.append(grover_op, range(N))
qc.measure(range(N), range(N))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

from qiskit import transpile

sim = AerSimulator(method="statevector")
shots = 4096
tqc = transpile(qc, sim, basis_gates=["u", "cx"])
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit-string order is c[N-1]...c[0]; convert each measured string
# back to the little-endian integer used above (bit i -> EDGES[i]).
def bitstring_to_int(bs):
    return int(bs[::-1], 2)

good_shots = 0
top_bitstring, top_shots = max(counts.items(), key=lambda kv: kv[1])
top_int = bitstring_to_int(top_bitstring)
for bs, c in counts.items():
    if bitstring_to_int(bs) in good_states:
        good_shots += c

measured_good_fraction = good_shots / shots
baseline_fraction = num_good / NSTATES

print(f"Grover iterations used: {iterations}")
print(f"Most probable measured outcome: {top_bitstring} "
      f"(edge-coloring index {top_int}), {top_shots}/{shots} shots")
print(f"Fraction of shots landing on a triangle-free coloring: "
      f"{measured_good_fraction:.3f} (uniform-random baseline: "
      f"{baseline_fraction:.3f})")

top_is_good = top_int in good_states
amplified = measured_good_fraction > 3 * baseline_fraction

verified = top_is_good and amplified

if verified:
    print("PASS")
else:
    print("FAIL")
