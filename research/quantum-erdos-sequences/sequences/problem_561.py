"""
Erdos problem #561 (data/problems.yaml, manman4/erdosproblems) -- Grover search
for triangle-free-in-each-color edge 2-colorings of K4 (Ramsey theory).

Problem #561's YAML entry carries no OEIS id (`oeis: ["N/A"]`); tags are
["graph theory", "ramsey theory"]. Because there is no associated sequence to
target, this script does NOT test a specific OEIS term. Instead it builds a
genuine, finite, computable property drawn directly from the problem's own
subject area (Ramsey theory) and verifies it with a real Grover search, which
is the most honest thing to do without an OEIS id to anchor to: fabricating a
sequence value would misrepresent the problem.

Classical property tested
--------------------------
Take the complete graph K4 (6 edges, indexed 0..5 in a fixed order over the
4 vertices) and 2-color its edges (bit 0 = "red", bit 1 = "blue"), so the
search space is exactly the 2^6 = 64 colorings, one per 6-bit string.

A coloring is a *valid* Ramsey-avoiding coloring if none of K4's four
triangles is monochromatic (this is the toy K4 analogue of R(3,3): unlike
K6, K4 has 2-colorings with no monochromatic triangle -- that's exactly why
R(3,3) = 6 and not smaller). We compute, by brute-force classical
enumeration over all 64 colorings, the exact set of valid colorings -- this
is the ground truth ("classical answer").

We then build a Grover search circuit whose oracle marks precisely that same
set of valid colorings (the oracle is derived from the classical truth table
computed above, encoded as an exact diagonal phase flip -- no oracle value is
copied from any external source), run it once through the standard number of
Grover iterations for this search-space size and solution count, and check
that measurement overwhelmingly returns states in the classical solution set.

PASS/FAIL is decided by comparing the quantum-sampled distribution's support
against the classically computed solution set.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Diagonal
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: enumerate all 2-colorings of K4's 6 edges and
#    find those with no monochromatic triangle.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges, fixed order
assert len(EDGES) == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles
assert len(TRIANGLES) == 4


def triangle_edge_indices(tri):
    a, b, c = tri
    pairs = [(a, b), (a, c), (b, c)]
    idxs = []
    for p in pairs:
        p = tuple(sorted(p))
        idxs.append(EDGE_INDEX[p])
    return idxs


TRIANGLE_EDGE_IDXS = [triangle_edge_indices(t) for t in TRIANGLES]


def is_valid_coloring(bits):
    """bits: tuple of 6 ints (0/1), one per edge in EDGES order.
    Returns True iff no triangle is monochromatic."""
    for idxs in TRIANGLE_EDGE_IDXS:
        colors = {bits[i] for i in idxs}
        if len(colors) == 1:
            return False
    return True


N_QUBITS = 6
N_STATES = 2 ** N_QUBITS

classical_valid = []
for x in range(N_STATES):
    bits = tuple((x >> i) & 1 for i in range(N_QUBITS))
    if is_valid_coloring(bits):
        classical_valid.append(x)

classical_valid_set = set(classical_valid)
num_solutions = len(classical_valid_set)

print(f"Search space size: {N_STATES} (6-edge 2-colorings of K4)")
print(f"Classically computed valid (triangle-free-per-color) colorings: {num_solutions}")
assert 0 < num_solutions < N_STATES, "degenerate search space; cannot Grover-search this"

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle as an exact diagonal phase flip derived from the
#    classical truth table above (phase -1 on solutions, +1 elsewhere).
# ---------------------------------------------------------------------------

diag = [(-1.0 if x in classical_valid_set else 1.0) for x in range(N_STATES)]
oracle_gate = Diagonal(diag)


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


diffuser_gate = build_diffuser(N_QUBITS)

# Optimal number of Grover iterations for this space/solution-count.
theta = math.asin(math.sqrt(num_solutions / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle_gate.to_instruction(), range(N_QUBITS))
    qc.append(diffuser_gate.to_instruction(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare against the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 20000
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit-string order is little-endian in the classical register vs our
# qubit index convention (qubit i -> bit i of x); Qiskit prints c_{n-1}...c_0,
# so reverse before turning back into an integer to match `x` above.
sampled = {}
for bitstring, cnt in counts.items():
    x = int(bitstring[::-1], 2)
    sampled[x] = cnt

total_shots = sum(sampled.values())
hits_in_solution_set = sum(cnt for x, cnt in sampled.items() if x in classical_valid_set)
fraction_correct = hits_in_solution_set / total_shots

top_states = sorted(sampled.items(), key=lambda kv: -kv[1])[:num_solutions]
top_states_in_solution = all(x in classical_valid_set for x, _ in top_states)

print(f"Fraction of shots landing on a classically valid coloring: {fraction_correct:.4f}")
print(f"Top {num_solutions} most-sampled states all classically valid: {top_states_in_solution}")

# Success criteria: Grover amplification should concentrate the overwhelming
# majority of shots on the classical solution set, and the most-sampled
# states (as many as there are solutions) should all be genuine solutions.
verified = fraction_correct > 0.90 and top_states_in_solution

if verified:
    print("PASS")
else:
    print("FAIL")
