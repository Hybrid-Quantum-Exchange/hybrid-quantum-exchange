"""
Erdos problem #191 (erdosproblems.com / manman4/erdosproblems data/problems.yaml)
tags: ["combinatorics", "ramsey theory"], oeis: ["N/A"]

Problem #191's YAML record carries no OEIS sequence id at all (oeis: ["N/A"]),
only the tags "combinatorics" and "ramsey theory". Per the task instructions,
when no OEIS id is available the honest move is to pick a small, finite,
genuinely-computable property drawn from the problem's own stated area
(Ramsey theory) rather than inventing an unrelated OEIS-backed sequence or
faking a link. This script is therefore NOT built from an OEIS sequence; it
tests a real, self-contained Ramsey-theory fact adjacent to problem #191's
tags, and this limitation (no OEIS id available for #191) is reported
honestly in the final output below, as instructed.

Classical property tested
--------------------------
R(3,3) = 6 is the classical 2-colour Ramsey number for triangles: every
2-colouring of the edges of K6 contains a monochromatic triangle, but K5 (and
a fortiori K4) admits 2-colourings that avoid one. We test the K4 case
directly, which is small enough for an exhaustive quantum search:

    Does there exist a 2-colouring of the 6 edges of the complete graph K4
    that contains NO monochromatic triangle?

K4 has 4 vertices {0,1,2,3}, 6 edges, and C(4,3) = 4 triangles. Each edge
colouring is a 6-bit string (bit i = colour of edge i, 0 or 1); there are
2^6 = 64 colourings total. The classical answer (computed from first
principles by brute force below, independent of the quantum part) is: yes,
"colouring-avoiding-mono-triangle" colourings exist, and we compute exactly
which 6-bit strings they are.

Quantum approach
-----------------
Grover search over the 64 edge-colourings of K4. A phase oracle is built as
an exact diagonal unitary from the classical truth table (the same
mono-triangle-free predicate computed classically for verification -- the
oracle is derived from the real combinatorial condition, not hard-coded to a
single "answer" bitstring). A standard Grover diffuser then amplifies the
marked (mono-triangle-free) colourings. The circuit is run on the ideal
AerSimulator; the top measured bitstrings are checked against the classical
solution set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed from first principles, no lookups)
# ---------------------------------------------------------------------------

N_VERTICES = 4
VERTICES = list(range(N_VERTICES))
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges, index 0..5
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles
N_EDGES = len(EDGES)
assert N_EDGES == 6

TRIANGLE_EDGE_IDXS = []
for (a, b, c) in TRIANGLES:
    idxs = (
        EDGE_INDEX[(a, b)],
        EDGE_INDEX[(a, c)],
        EDGE_INDEX[(b, c)],
    )
    TRIANGLE_EDGE_IDXS.append(idxs)


def is_mono_triangle_free(bits):
    """bits: tuple of 6 ints (0/1), one per edge in EDGES order.
    Returns True iff no triangle of K4 is monochromatic under this colouring."""
    for (i, j, k) in TRIANGLE_EDGE_IDXS:
        if bits[i] == bits[j] == bits[k]:
            return False
    return True


classical_solutions = []
for combo in itertools.product([0, 1], repeat=N_EDGES):
    if is_mono_triangle_free(combo):
        classical_solutions.append(combo)

N_TOTAL = 2 ** N_EDGES
M_SOLUTIONS = len(classical_solutions)

assert M_SOLUTIONS > 0, "classical brute force found no mono-triangle-free colouring of K4 -- unexpected"

# bitstring form (Qiskit little-endian: qubit 0 is the rightmost character)
def bits_to_qiskit_string(bits):
    # bits[0] is edge 0 -> should be the LSB (rightmost) in Qiskit's convention
    return "".join(str(b) for b in reversed(bits))


classical_solution_strings = {bits_to_qiskit_string(b) for b in classical_solutions}

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle as an exact diagonal unitary from the truth table
# ---------------------------------------------------------------------------

diag = np.ones(N_TOTAL, dtype=complex)
for state_int in range(N_TOTAL):
    # state_int's binary representation, Qiskit little-endian: bit i (from LSB)
    # corresponds to edge i.
    bits = tuple((state_int >> i) & 1 for i in range(N_EDGES))
    if is_mono_triangle_free(bits):
        diag[state_int] = -1.0

oracle_unitary = Operator(np.diag(diag))

oracle_circ = QuantumCircuit(N_EDGES, name="MonoTriFreeOracle")
oracle_circ.unitary(oracle_unitary, range(N_EDGES), label="oracle")

# ---------------------------------------------------------------------------
# 3. Standard Grover diffuser (reflection about the uniform superposition)
# ---------------------------------------------------------------------------

diffuser = QuantumCircuit(N_EDGES, name="Diffuser")
diffuser.h(range(N_EDGES))
diffuser.x(range(N_EDGES))
diffuser.h(N_EDGES - 1)
diffuser.append(MCXGate(N_EDGES - 1), list(range(N_EDGES - 1)) + [N_EDGES - 1])
diffuser.h(N_EDGES - 1)
diffuser.x(range(N_EDGES))
diffuser.h(range(N_EDGES))

# ---------------------------------------------------------------------------
# 4. Assemble the full Grover circuit with the optimal iteration count
# ---------------------------------------------------------------------------

n_iterations = max(1, round((math.pi / 4) * math.sqrt(N_TOTAL / M_SOLUTIONS)))

qc = QuantumCircuit(N_EDGES, N_EDGES)
qc.h(range(N_EDGES))
for _ in range(n_iterations):
    qc.append(oracle_circ.to_instruction(), range(N_EDGES))
    qc.append(diffuser.to_instruction(), range(N_EDGES))
qc.measure(range(N_EDGES), range(N_EDGES))

# ---------------------------------------------------------------------------
# 5. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_k = min(M_SOLUTIONS, 8)  # look at as many top outcomes as there are solutions (capped)
top_outcomes = sorted_counts[:top_k]

hits_in_solution_set = sum(
    c for s, c in counts.items() if s in classical_solution_strings
)
success_fraction = hits_in_solution_set / SHOTS

# The measured mode (most frequent outcome) must itself be a genuine solution,
# and Grover amplification should concentrate a clear majority of shots on
# the marked (mono-triangle-free) subspace, well above the 1/64 baseline.
most_common_string, most_common_count = sorted_counts[0]
mode_is_valid = most_common_string in classical_solution_strings
baseline_fraction = M_SOLUTIONS / N_TOTAL

verified = mode_is_valid and (success_fraction > max(0.5, 2 * baseline_fraction))

print("Erdos problem #191 -- quantum-testable companion (no OEIS id available)")
print(f"K4 edges: {N_EDGES}, total colourings: {N_TOTAL}, mono-triangle-free colourings (classical): {M_SOLUTIONS}")
print(f"Grover iterations used: {n_iterations}")
print(f"Most frequent measured colouring: {most_common_string} (count {most_common_count}/{SHOTS}), valid solution: {mode_is_valid}")
print(f"Fraction of shots landing on a mono-triangle-free colouring: {success_fraction:.4f} (uniform baseline would be {baseline_fraction:.4f})")
print(f"Top {top_k} measured outcomes: {top_outcomes}")

if verified:
    print("PASS")
else:
    print("FAIL")
