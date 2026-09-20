"""
Erdos problem #474 (erdosproblems.com), quantum-testable lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, read-only clone):
    number: "474"
    prize: $100
    informal_status: not provable (as of 2025-10-01)
    oeis: ["N/A"]
    tags: ["set theory", "ramsey theory"]

HONEST LIMITATION: problem 474 has no associated OEIS sequence (oeis: "N/A"
in the source data). There is therefore no OEIS-sequence membership or term
property to test, and this script cannot be "the quantum test of OEIS
sequence X" the way most other lanes in this library are. Per the task's
fallback instructions, this is the best-honest-attempt alternative: it takes
the problem's own tag ("ramsey theory") and tests a small, finite, genuinely
computable Ramsey-theoretic fact with a real Grover search circuit, rather
than fabricating or copying an OEIS value that does not exist for this
problem.

Classical property under test:
    K4 (the complete graph on 4 vertices) has 6 edges and C(4,3) = 4
    triangles. A 2-coloring of the edges (RED/BLUE) is "triangle-free" if no
    triangle is monochromatic. This is exactly the base case behind
    R(3,3) = 6 (the fact that K4, K5 admit triangle-free 2-colorings while
    K6 does not) -- a standard, classical, small Ramsey-theoretic instance.

    We classically enumerate all 2^6 = 64 edge-colorings of K4 and determine,
    by brute force from first principles, exactly which ones are
    triangle-free monochromatic-free colorings. This is the "known term":
    the set of marked (good) bitstrings.

Quantum circuit:
    A 6-qubit Grover search circuit (AerSimulator, statevector-exact) whose
    oracle is a diagonal phase-flip gate built directly from the classically
    precomputed marked set (edges encoded as qubits 0..5, one per edge of
    K4). The oracle and diffuser are applied for the optimal number of
    Grover iterations for this search-space size and number of solutions.
    We then measure and compare the highest-probability bitstrings returned
    by the quantum search against the classically verified marked set.

Pass condition:
    PASS iff every one of the top-k most frequently measured bitstrings
    (k = number of classical solutions) is indeed in the classically
    verified triangle-free-coloring set.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges of K4
assert len(EDGES) == 6
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles of K4
assert len(TRIANGLES) == 4

EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}


def triangle_edges(tri):
    a, b, c = tri
    return [
        EDGE_INDEX[tuple(sorted((a, b)))],
        EDGE_INDEX[tuple(sorted((b, c)))],
        EDGE_INDEX[tuple(sorted((a, c)))],
    ]


TRIANGLE_EDGE_IDX = [triangle_edges(t) for t in TRIANGLES]


def is_triangle_free_coloring(bits):
    """bits: tuple of 6 ints (0/1), one color bit per edge (0=RED,1=BLUE).
    Returns True iff no triangle among TRIANGLES is monochromatic."""
    for idxs in TRIANGLE_EDGE_IDX:
        colors = {bits[i] for i in idxs}
        if len(colors) == 1:
            return False
    return True


N_QUBITS = 6
N_STATES = 2 ** N_QUBITS

classical_marked = []
for combo in itertools.product([0, 1], repeat=N_QUBITS):
    if is_triangle_free_coloring(combo):
        # bit order: combo[0] -> edge 0 -> qubit 0 (LSB convention below)
        idx = sum(b << i for i, b in enumerate(combo))
        classical_marked.append(idx)

classical_marked = sorted(set(classical_marked))
num_solutions = len(classical_marked)

print(f"K4 has {N_QUBITS} edges, {len(TRIANGLES)} triangles, "
      f"{N_STATES} total 2-colorings.")
print(f"Classically verified triangle-free colorings: {num_solutions} "
      f"(known Ramsey fact: this must be > 0, matching R(3,3) = 6 > 4).")
assert num_solutions > 0, "sanity: K4 must admit a triangle-free 2-coloring"
assert num_solutions < N_STATES, "sanity: not every coloring can be triangle-free"

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle as an exact diagonal phase-flip gate.
# ---------------------------------------------------------------------------

diag = np.ones(N_STATES, dtype=complex)
for idx in classical_marked:
    diag[idx] = -1.0

oracle_gate = DiagonalGate(diag.tolist())

# ---------------------------------------------------------------------------
# 3. Diffuser (inversion about the mean), standard Grover construction.
# ---------------------------------------------------------------------------


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

# ---------------------------------------------------------------------------
# 4. Full Grover circuit, optimal iteration count for this M/N ratio.
# ---------------------------------------------------------------------------

theta = math.asin(math.sqrt(num_solutions / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle_gate, range(N_QUBITS))
    qc.append(diffuser_gate, range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

print(f"Grover iterations used: {iterations} "
      f"(N={N_STATES}, M={num_solutions}, theta={theta:.4f})")

# ---------------------------------------------------------------------------
# 5. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 4096
qc_t = transpile(qc, backend, basis_gates=["u", "cx"])
job = backend.run(qc_t, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB..LSB left-to-right; convert back to our
# little-endian qubit-index convention (qubit i = bit i, LSB = qubit 0).
def bitstring_to_index(bs):
    bs_rev = bs[::-1]  # now bs_rev[i] is qubit i
    return int(bs_rev, 2)

measured = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_k = measured[:num_solutions]
top_k_indices = [bitstring_to_index(bs) for bs, _ in top_k]

print("Top measured outcomes (bitstring, count, decoded index):")
for (bs, cnt), idx in zip(top_k, top_k_indices):
    print(f"  {bs}  count={cnt:5d}  index={idx:2d}  "
          f"classically_marked={idx in classical_marked}")

# ---------------------------------------------------------------------------
# 6. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

all_top_k_are_marked = all(idx in classical_marked for idx in top_k_indices)

verified = all_top_k_are_marked
print()
if verified:
    print("PASS")
else:
    print("FAIL")
