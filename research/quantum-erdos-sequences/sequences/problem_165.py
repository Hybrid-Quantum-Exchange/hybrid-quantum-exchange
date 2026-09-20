"""
Erdos problem #165 (https://erdosproblems.com/165) -- OEIS A000791.

A000791 is the sequence of Ramsey numbers R(3,n): a(n) is the smallest N
such that every 2-coloring of the edges of the complete graph K_N contains
either a red triangle or a blue n-clique.  For n = 3 this is the classical
diagonal Ramsey number R(3,3) = 6, whose two defining facts are:

  (a) every 2-coloring of K_6's edges contains a monochromatic triangle, and
  (b) K_5 (one vertex fewer) admits at least one 2-coloring of its edges
      with NO monochromatic triangle at all (this is what makes 6, not 5,
      the answer -- R(3,3)-1 = 5 is exactly the size of the extremal
      "good" coloring).

The property tested here is (b), restricted to K_5, which is small and
finite: does there exist a 2-coloring of the 10 edges of K_5 with no
monochromatic triangle, and what is the exact count of such colorings out
of the 2^10 = 1024 possible edge-colorings?  We first answer this
classically, from first principles (brute-force enumeration of all 1024
colorings and all C(5,3) = 10 triangles), then verify a quantum Grover
search over the same 10-qubit space (one qubit per edge) finds a marked
("good") coloring with amplified probability, using a phase oracle built
from that same classical enumeration (Grover needs to be told which
computational-basis states are marked -- that is the oracle's job -- but
the *set* of marked states, and its size, are derived here in Python, not
copied from OEIS).

Classical ground truth (computed in this script, not hard-coded):
  - Vertices of K_5: 0..4.  Edges: the 10 unordered pairs.
  - Triangles: the 10 unordered triples of vertices.
  - A coloring (bitstring of length 10, one bit per edge, 0=red/1=blue) is
    "good" if no triangle's three edges are all the same color.
  - We count the good colorings by brute force over all 1024 possibilities.
    (Known combinatorial fact: this count is 12, coming from the two
    "pentagon / pentagram" colorings of K_5 under its 10 automorphisms
    that fix a 2-coloring class; we do not assume this number, we compute
    it.)

Quantum method: exact Grover search (built by hand -- qiskit_algorithms is
not installed in this environment) over the 10-qubit space of edge
colorings.  The oracle is a diagonal phase-flip gate built directly from
the classical "good coloring" set computed above (this is the standard way
an oracle is specified in a toy Grover instance -- the oracle "knows" the
marked set the same way a boolean function passed to Grover's algorithm
always must). The diffuser is the standard Grover diffusion operator. The
number of Grover iterations is computed from the exact count of marked
states (also computed classically here, not looked up). We run the
circuit on the ideal AerSimulator and check that the most frequently
measured bitstring is indeed one of the classically-verified good
colorings, and that the total measured probability mass landing on good
colorings is amplified far above the uniform baseline of 12/1024 ~= 1.2%.

PASS criterion: the single most-sampled 10-bit outcome is a good coloring
(no monochromatic triangle when checked from scratch), AND the measured
probability mass on good colorings exceeds 10x the uniform baseline.
"""

import itertools
import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical ground truth: enumerate K_5's edges/triangles and find all
#    2-colorings with no monochromatic triangle, entirely from scratch.
# ---------------------------------------------------------------------

VERTICES = list(range(5))
EDGES = list(itertools.combinations(VERTICES, 2))  # 10 edges
assert len(EDGES) == 10
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 10 triangles
assert len(TRIANGLES) == 10

# For each triangle, the indices (into EDGES) of its three edges.
TRIANGLE_EDGE_IDX = []
for (a, b, c) in TRIANGLES:
    e1 = EDGE_INDEX[(a, b)]
    e2 = EDGE_INDEX[(a, c)]
    e3 = EDGE_INDEX[(b, c)]
    TRIANGLE_EDGE_IDX.append((e1, e2, e3))


def is_good_coloring(bits):
    """bits: length-10 sequence of 0/1, bits[i] = color of EDGES[i].
    Returns True iff no triangle is monochromatic (no mono triangle)."""
    for (e1, e2, e3) in TRIANGLE_EDGE_IDX:
        if bits[e1] == bits[e2] == bits[e3]:
            return False
    return True


N_QUBITS = 10
N_STATES = 2 ** N_QUBITS  # 1024

good_states = []  # list of integers 0..1023 whose binary form (bit i = edge i) is good
for state in range(N_STATES):
    bits = [(state >> i) & 1 for i in range(N_QUBITS)]
    if is_good_coloring(bits):
        good_states.append(state)

num_good = len(good_states)
assert num_good > 0, "classical search found no good coloring of K_5 -- would contradict R(3,3)=6"

print(f"Classical brute force over all {N_STATES} edge-colorings of K_5:")
print(f"  number of colorings with NO monochromatic triangle: {num_good}")
print(f"  (this is the classical witness underlying R(3,3) = 6, OEIS A000791 a(3))")

uniform_baseline = num_good / N_STATES


# ---------------------------------------------------------------------
# 2. Build a Grover search circuit over the 10-qubit edge-coloring space,
#    with the oracle marking exactly `good_states` (built from the
#    classical enumeration above -- this is what "the oracle" means in
#    Grover's algorithm: a black box evaluating the predicate).
# ---------------------------------------------------------------------

# Diagonal phase oracle: -1 on marked (good) states, +1 elsewhere.
diag = np.ones(N_STATES, dtype=complex)
for s in good_states:
    diag[s] = -1.0
oracle_gate = DiagonalGate(diag.tolist())

# Standard Grover diffuser (inversion about the mean) on N_QUBITS qubits.
def diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


diffuser_gate = diffuser(N_QUBITS).to_gate()

# Optimal number of Grover iterations for M marked out of N states:
# floor(pi/4 * sqrt(N/M)).
iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N_STATES / num_good)))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle_gate, range(N_QUBITS))
    qc.append(diffuser_gate, range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

print(f"\nGrover search: N={N_STATES} states, M={num_good} marked, iterations={iterations}")


# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------

backend = AerSimulator()
shots = 8192
tqc = transpile(qc, backend)
job = backend.run(tqc, shots=shots)
result = job.result()
counts = result.get_counts()

# qiskit bit ordering: classical register bit c_i corresponds to qubit i,
# and the returned bitstring is printed c_(n-1) ... c_0 (MSB first, i.e.
# rightmost character is clbit 0 = qubit 0). That is exactly the standard
# binary representation of an integer with qubit 0 as the LSB, so no
# reversal is needed -- interpreting the bitstring directly as a base-2
# integer already matches our state convention (bit i = edge i = qubit i).
def bitstring_to_state(bs):
    return int(bs, 2)

state_counts = Counter()
for bitstring, c in counts.items():
    state_counts[bitstring_to_state(bitstring)] += c

most_common_state, most_common_count = state_counts.most_common(1)[0]
most_common_is_good = most_common_state in good_states

good_mass = sum(c for s, c in state_counts.items() if s in good_states) / shots
amplification = good_mass / uniform_baseline if uniform_baseline > 0 else float("inf")

print(f"\nMost-sampled state: {most_common_state:010b} "
      f"(count {most_common_count}/{shots}), good={most_common_is_good}")
print(f"Measured probability mass on good colorings: {good_mass:.4f} "
      f"(uniform baseline would be {uniform_baseline:.4f})")
print(f"Amplification factor over uniform baseline: {amplification:.2f}x")

# Independently re-verify (classically, from scratch) that the top result
# really is a valid no-monochromatic-triangle coloring of K_5.
top_bits = [(most_common_state >> i) & 1 for i in range(N_QUBITS)]
top_reverified = is_good_coloring(top_bits)

passed = most_common_is_good and top_reverified and (amplification > 10.0)

print(f"\nRe-verified top result classically: {top_reverified}")
print("RESULT:", "PASS" if passed else "FAIL")
