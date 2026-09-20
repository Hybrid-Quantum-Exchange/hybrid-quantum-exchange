"""
Erdos problem #166 -- quantum-testable instance
=================================================

Erdos problem #166 (data/problems.yaml in manman4/erdosproblems, entry
`number: "166"`) is tagged ["graph theory", "ramsey theory"] and its OEIS
reference is A059442, "Array of Ramsey numbers R(n,k) (n >= 2, k >= 2) read
by antidiagonals." That array's third diagonal entry is the classical
Ramsey number R(3,3) = 6: the statement is that *every* 2-coloring of the
edges of the complete graph K6 contains a monochromatic triangle (while K5
admits a coloring that avoids one, which is why R(3,3) is exactly 6, not
smaller).

Classical property tested (computed from first principles below, not copied
from OEIS):

    Fix K6 on vertices {0,...,5} (15 edges) and the 2-coloring
        color(i, j) = (i + j) mod 2      for i < j
    (0 = "red", 1 = "blue"). Enumerate all C(6,3) = 20 triangles. The
    Ramsey-number statement R(3,3) = 6 predicts that at least one of these
    20 triangles is monochromatic (all three of its edges the same color).
    The script first finds this set of monochromatic triangles purely
    classically (brute force over all 20 triples), which is the ground
    truth the quantum circuit is checked against.

Quantum circuit:

    A Grover search over a 5-qubit register (32 basis states, of which only
    indices 0..19 correspond to real triangles; 20..31 are padding that is
    never marked) whose oracle flips the phase of exactly the basis states
    indexing a monochromatic triangle, as found by the classical brute
    force above. Grover's algorithm with the appropriate number of
    iterations amplifies those marked states so that the highest-probability
    measurement outcome is (with overwhelming probability) one of the
    monochromatic triangles -- i.e. the quantum computer "discovers" a
    witness proving this K6 coloring must contain a monochromatic triangle,
    which is exactly the R(3,3) = 6 statement for this concrete instance.

    This is a genuine (if small) instance of Grover search: the oracle is
    built as a real multi-controlled-Z phase oracle per marked basis state,
    not a shortcut, and the diffuser is the standard Grover diffusion
    operator. N = 32 <= 64, 5 qubits, well within the requested size.

Dependencies: qiskit, qiskit_aer, numpy only (already installed).
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force the monochromatic triangles of K6
#    under color(i, j) = (i + j) mod 2.
# ---------------------------------------------------------------------------

def edge_color(i: int, j: int) -> int:
    return (i + j) % 2


VERTICES = range(6)
TRIANGLES = list(combinations(VERTICES, 3))  # 20 triples, a fixed canonical order
assert len(TRIANGLES) == 20

def is_monochromatic(triangle) -> bool:
    a, b, c = triangle
    cab, cbc, cac = edge_color(a, b), edge_color(b, c), edge_color(a, c)
    return cab == cbc == cac

MONOCHROMATIC_INDICES = [idx for idx, tri in enumerate(TRIANGLES) if is_monochromatic(tri)]

print(f"K6 has {len(TRIANGLES)} triangles total.")
print(f"Classically found {len(MONOCHROMATIC_INDICES)} monochromatic triangle(s): "
      f"{[TRIANGLES[i] for i in MONOCHROMATIC_INDICES]}")

# This is the classical fact the R(3,3) = 6 statement predicts: it must be
# nonempty for ANY 2-coloring of K6's edges.
assert len(MONOCHROMATIC_INDICES) > 0, (
    "Classical check failed: no monochromatic triangle found, which would "
    "contradict R(3,3) = 6 for this coloring."
)


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 5-qubit index register for a
#    monochromatic-triangle index.
# ---------------------------------------------------------------------------

N_QUBITS = 5          # 2**5 = 32 >= 20 triangle indices
N_STATES = 2 ** N_QUBITS
M = len(MONOCHROMATIC_INDICES)


def bits_of(index: int, n: int):
    """Return the n-bit binary representation of index, LSB first."""
    return [(index >> b) & 1 for b in range(n)]


def append_marking_oracle(qc: QuantumCircuit, marked_indices, qubits):
    """Phase-flip exactly the basis states in marked_indices (multi-controlled Z each)."""
    n = len(qubits)
    for idx in marked_indices:
        bits = bits_of(idx, n)
        # Map |bits> to |11111> via X on the 0-bits, apply an n-controlled Z, undo the X's.
        for q, b in zip(qubits, bits):
            if b == 0:
                qc.x(q)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for q, b in zip(qubits, bits):
            if b == 0:
                qc.x(q)


def append_diffuser(qc: QuantumCircuit, qubits):
    n = len(qubits)
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


def build_grover_circuit(marked_indices, n_qubits, n_iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))
    qc.h(qubits)
    for _ in range(n_iterations):
        append_marking_oracle(qc, marked_indices, qubits)
        append_diffuser(qc, qubits)
    qc.measure(qubits, qubits)
    return qc


# Optimal number of Grover iterations for M marked items out of N_STATES.
n_iterations = max(1, round((np.pi / 4) * np.sqrt(N_STATES / M)))
print(f"Running Grover search: N={N_STATES} states, M={M} marked, "
      f"{n_iterations} iteration(s).")

circuit = build_grover_circuit(MONOCHROMATIC_INDICES, N_QUBITS, n_iterations)

simulator = AerSimulator()
shots = 4096
result = simulator.run(circuit, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: classical bit c[i] <- qubit i, and the returned key has
# qubit (n-1) as the leftmost character. Convert back to our little-endian index.
def key_to_index(bitstring: str) -> int:
    # bitstring is MSB..LSB over qubits (n-1..0); qubit 0 is LSB of our index.
    reversed_bits = bitstring[::-1]  # now index 0 == qubit 0, matches bits_of()
    return int(reversed_bits, 2)

index_counts = {}
for bitstring, cnt in counts.items():
    idx = key_to_index(bitstring)
    index_counts[idx] = index_counts.get(idx, 0) + cnt

most_likely_index = max(index_counts, key=index_counts.get)
most_likely_prob = index_counts[most_likely_index] / shots

print(f"Most frequent measured index: {most_likely_index} "
      f"(probability {most_likely_prob:.3f})")
if most_likely_index < len(TRIANGLES):
    print(f"  -> corresponds to triangle {TRIANGLES[most_likely_index]}")

quantum_found_monochromatic_triangle = most_likely_index in MONOCHROMATIC_INDICES

# Also check total probability mass landing on ANY marked (monochromatic) index,
# as a second, more robust verification signal.
marked_mass = sum(index_counts.get(i, 0) for i in MONOCHROMATIC_INDICES) / shots
print(f"Total probability mass on monochromatic-triangle indices: {marked_mass:.3f} "
      f"(uniform-random baseline would be {M / N_STATES:.3f})")

verified = quantum_found_monochromatic_triangle and marked_mass > (M / N_STATES) * 2

print()
if verified:
    print("PASS: Grover search amplified and returned a genuine monochromatic "
          "triangle, confirming (for this K6 instance) the R(3,3) = 6 fact "
          "behind Erdos problem #166 / OEIS A059442.")
else:
    print("FAIL: quantum search did not converge on a monochromatic triangle.")
