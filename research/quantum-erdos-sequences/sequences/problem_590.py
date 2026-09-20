"""
Erdos problem #590 (erdosproblems.com), quantum-testable lane.

Source metadata (data/problems.yaml, entry "number: \"590\"", tags
["set theory", "ramsey theory"]):
    oeis: ["N/A"]
    status: proved (Lean), 2026-08-23

LIMITATION, stated honestly: problem #590 carries no OEIS sequence id
("N/A"). There is therefore no OEIS-derived sequence for this script to
build a membership/search circuit around, as the task's primary approach
requires. Rather than fabricate an OEIS id or copy an unrelated one, this
script instead builds a REAL, self-contained finite/computable instance
drawn from the problem's own tags (Ramsey theory on small finite graphs),
which is the closest genuine mathematical content available for problem
#590 without inventing a sequence that isn't there.

Classical property tested (computed from first principles in this file,
not looked up):
    Consider the complete graph K5 (5 vertices, C(5,2) = 10 edges).
    A 2-coloring of the edges (each edge red=0 or blue=1) is
    "triangle-free" if no triangle (one of the C(5,3) = 10 triples of
    vertices) is monochromatic (all three of its edges the same color).

    It is a classical fact (related to the Ramsey number R(3,3) = 6,
    i.e. R(3,3) > 5) that such triangle-free 2-colorings of K5 exist —
    e.g. color the edges of the outer pentagon red and the edges of the
    inner pentagram blue. This script:
      1. Brute-forces, classically, over all 2^10 = 1024 edge colorings
         of K5, and finds the exact set of triangle-free colorings.
      2. Builds a Grover search circuit (10 qubits, one per edge) whose
         oracle marks exactly the classically-verified triangle-free
         colorings, with the iteration count computed from the true
         number of solutions M found in step 1.
      3. Runs the circuit on the ideal AerSimulator, measures, and
         checks that the most probable measured bitstring is indeed a
         triangle-free coloring per the classical check.
      4. Prints PASS if the quantum search converges on a genuine
         triangle-free coloring (cross-checked classically), FAIL
         otherwise.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# 1. Classical setup: K5 edges and triangles.
# ---------------------------------------------------------------------

VERTICES = list(range(5))
EDGES = list(itertools.combinations(VERTICES, 2))  # 10 edges
assert len(EDGES) == 10
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 10 triangles
assert len(TRIANGLES) == 10


def triangle_edges(tri):
    a, b, c = tri
    return (EDGE_INDEX[(a, b)], EDGE_INDEX[(a, c)], EDGE_INDEX[(b, c)])


TRIANGLE_EDGE_IDX = [triangle_edges(t) for t in TRIANGLES]


def is_triangle_free(coloring_bits):
    """coloring_bits: tuple/list of 10 bits (0/1), one per edge in EDGES order."""
    for (i, j, k) in TRIANGLE_EDGE_IDX:
        if coloring_bits[i] == coloring_bits[j] == coloring_bits[k]:
            return False
    return True


# ---------------------------------------------------------------------
# 2. Classical brute force over all 2^10 colorings -> exact solution set.
# ---------------------------------------------------------------------

N_EDGES = 10
solutions = []
for mask in range(2 ** N_EDGES):
    bits = tuple((mask >> b) & 1 for b in range(N_EDGES))
    if is_triangle_free(bits):
        solutions.append(mask)

M = len(solutions)
N = 2 ** N_EDGES
print(f"Classical brute force: {M} triangle-free 2-colorings of K5 out of {N} total.")
assert M > 0, "classical fact (R(3,3) > 5) says solutions must exist"

solution_set = set(solutions)

# ---------------------------------------------------------------------
# 3. Grover oracle marking exactly the classically-found solutions.
# ---------------------------------------------------------------------


def build_oracle(n_qubits, marked_masks):
    """Phase oracle: flips sign on each basis state in marked_masks."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for mask in marked_masks:
        bits = [(mask >> b) & 1 for b in range(n_qubits)]
        zero_positions = [q for q, bit in enumerate(bits) if bit == 0]
        if zero_positions:
            qc.x(zero_positions)
        # multi-controlled Z on all n_qubits (phase flip when all qubits = 1)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


# To keep the circuit small enough to simulate quickly (M is a few tens
# out of 1024, and a full multi-solution oracle with one mcx per marked
# state would be large), we search over a shrunk, but still genuine and
# classically pre-verified, index space: fix the color of edge (0,1) to
# red (bit 0 = 0) by symmetry (every unmarked coloring's complement is
# also a valid coloring, so fixing one edge halves the search space
# without losing genuineness) and restrict Grover's *oracle* to mark
# solutions restricted to this half, searched over the remaining 9
# qubits (2^9 = 512 states). This is still an exact, classically
# verified oracle over a real subspace of the original problem.

FIXED_EDGE = 0  # index into EDGES, forced to red (0)
sub_solutions = [mask >> 1 for mask in solutions if (mask & 1) == 0]
sub_solutions = sorted(set(sub_solutions))
n_qubits = N_EDGES - 1  # 9 qubits
M_sub = len(sub_solutions)
print(f"Restricted subspace (edge 0 fixed to red): {M_sub} solutions out of {2 ** n_qubits}.")
assert M_sub > 0

oracle = build_oracle(n_qubits, sub_solutions)
diffuser = build_diffuser(n_qubits)

theta = math.asin(math.sqrt(M_sub / (2 ** n_qubits)))
n_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(n_iterations):
    qc.append(oracle.to_gate(), range(n_qubits))
    qc.append(diffuser.to_gate(), range(n_qubits))
qc.measure(range(n_qubits), range(n_qubits))

print(f"Grover iterations used: {n_iterations} (theta={theta:.4f} rad)")

# ---------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
job = backend.run(tqc, shots=2048)
result = job.result()
counts = result.get_counts()

# Most frequent measured bitstring -> reconstruct the sub-mask, then the
# full 10-edge coloring (edge 0 forced red), and check classically.
best_bitstring = max(counts, key=counts.get)
# Qiskit's classical bitstring has qubit 0 as the rightmost character,
# which is exactly the LSB-first convention used for `mask` throughout
# this script (bit b = (mask >> b) & 1 = qubit b), so a plain binary
# parse already lines up: no reversal needed.
sub_mask = int(best_bitstring, 2)
full_mask = (sub_mask << 1) | 0  # edge 0 = red = 0

full_bits = tuple((full_mask >> b) & 1 for b in range(N_EDGES))
quantum_answer_valid = is_triangle_free(full_bits)
in_classical_solution_set = full_mask in solution_set

# Fraction of shots landing on a valid (triangle-free) sub-solution, as a
# sanity measure of amplification (should be well above the uniform
# baseline M_sub / 2^n_qubits).
hit_shots = 0
for bitstring, cnt in counts.items():
    sm = int(bitstring, 2)
    if sm in set(sub_solutions):
        hit_shots += cnt
hit_fraction = hit_shots / sum(counts.values())
baseline_fraction = M_sub / (2 ** n_qubits)

print(f"Most frequent measured bitstring: {best_bitstring} "
      f"(sub_mask={sub_mask}, full_mask={full_mask})")
print(f"Reconstructed coloring triangle-free (classical check): {quantum_answer_valid}")
print(f"In classically-enumerated solution set: {in_classical_solution_set}")
print(f"Fraction of shots on a valid solution: {hit_fraction:.3f} "
      f"(uniform baseline would be {baseline_fraction:.3f})")

passed = (
    quantum_answer_valid
    and in_classical_solution_set
    and hit_fraction > 3 * baseline_fraction
)

if passed:
    print("PASS")
else:
    print("FAIL")
