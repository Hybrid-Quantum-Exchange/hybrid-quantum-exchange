"""
Erdos problem #669 ("orchard problems", tags: geometry; OEIS ids listed in
erdosproblems metadata: A003035, A006065, A008997) -- Quantum-testable lane.

Erdos problem #669 concerns the "orchard visibility problem": choosing n
points in the plane to maximize the number of lines that pass through
exactly 3 of them. A003035 / A006065 / A008997 are the OEIS sequences that
tabulate variants of this maximum, indexed by n. The full geometric
optimization (over all point placements in the plane) is not a small finite
search space, so for a genuinely small quantum circuit we fix a concrete
finite instance that is squarely in the spirit of the orchard problem and
is exactly the kind of combinatorial question those sequences tabulate:

    Take the 9 points of a 3x3 grid, labelled 0..8:

        0 1 2
        3 4 5
        6 7 8

    There are exactly 8 "3-point lines" among these 9 points: the 3 rows,
    the 3 columns, and the 2 main diagonals.

    Property tested: among all 84 = C(9,6) subsets of 6 of these 9 points,
    what is the maximum number of 3-point lines that are FULLY CONTAINED
    in the chosen subset, and which subsets (indices into an enumeration
    of all 84 subsets) achieve that maximum?

This is a genuine finite instance of the orchard-problem counting question
that A003035/A006065/A008997 are about (maximizing 3-point lines among a
fixed number of points), scaled down so a small quantum circuit can search
it directly by brute force amplitude amplification (Grover's algorithm),
rather than by copying a literal OEIS value.

The classical answer (computed from first principles in this script, by
enumerating all C(9,6) = 84 subsets and counting fully-contained lines) is
computed once as `CLASSICAL_MAX` and `CLASSICAL_MARKED` below. Grover's
algorithm is then run over a 7-qubit register (128 basis states, of which
the first 84 index the actual subsets and the rest are unused/never
marked) to amplify exactly the indices in `CLASSICAL_MARKED`. The circuit
is run on the ideal AerSimulator; the script PASSes if the most frequently
measured outcome is one of the classically-verified maximizing indices.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup)
# ---------------------------------------------------------------------------

POINTS = list(range(9))
LINES = [
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},   # rows
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8},   # columns
    {0, 4, 8}, {2, 4, 6},              # diagonals
]

COMBOS = list(itertools.combinations(POINTS, 6))  # 84 subsets of size 6
assert len(COMBOS) == 84

def lines_contained(subset):
    s = set(subset)
    return sum(1 for line in LINES if line <= s)

CLASSICAL_COUNTS = [lines_contained(c) for c in COMBOS]
CLASSICAL_MAX = max(CLASSICAL_COUNTS)
CLASSICAL_MARKED = [i for i, c in enumerate(CLASSICAL_COUNTS) if c == CLASSICAL_MAX]

print(f"Classical brute force over all {len(COMBOS)} 6-point subsets of the "
      f"3x3 grid:")
print(f"  maximum number of fully-contained 3-point lines = {CLASSICAL_MAX}")
print(f"  achieved by {len(CLASSICAL_MARKED)} subsets, indices: {CLASSICAL_MARKED}")

# ---------------------------------------------------------------------------
# 2. Quantum search (Grover's algorithm) over the 84 subset-indices
#    (7 qubits -> 128 basis states; indices 84..127 are simply never marked)
# ---------------------------------------------------------------------------

N_QUBITS = 7
N_STATES = 2 ** N_QUBITS  # 128
MARKED = CLASSICAL_MARKED

def oracle_circuit(marked_indices, n_qubits):
    """Phase-flip oracle: multiplies |i> by -1 for each i in marked_indices."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        # Flip qubits that are 0 in this index so the controlled-Z below
        # triggers exactly on this basis state.
        zero_qubits = [q for q, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc

def diffusion_circuit(n_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="Diffusion")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc

num_marked = len(MARKED)
iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / num_marked)))
print(f"Running Grover search: {N_QUBITS} qubits, {N_STATES} states, "
      f"{num_marked} marked, {iterations} Grover iteration(s).")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = oracle_circuit(MARKED, N_QUBITS)
diffusion = diffusion_circuit(N_QUBITS)

for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffusion, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Convert measured bitstrings (Qiskit reports qubit N-1 ... qubit 0, MSB
# first, big-endian in the printed string) back to little-endian indices
# matching the convention used in oracle_circuit.
def bitstring_to_index(bitstring, n_qubits):
    # bitstring is big-endian (qubit n-1 first); reverse to little-endian
    # then interpret with qubit 0 as the least-significant bit, matching
    # the `bits[::-1]` convention used when building the oracle.
    le = bitstring[::-1]  # now le[q] = qubit q's value
    idx = 0
    for q, b in enumerate(le):
        if b == "1":
            idx |= (1 << q)
    return idx

index_counts = {}
for bitstring, freq in counts.items():
    idx = bitstring_to_index(bitstring, N_QUBITS)
    index_counts[idx] = index_counts.get(idx, 0) + freq

most_likely_index = max(index_counts, key=index_counts.get)
most_likely_freq = index_counts[most_likely_index]

# Fraction of shots landing on ANY marked index (sanity check the
# amplification actually worked, not just that argmax happens to hit).
marked_shot_fraction = sum(index_counts.get(i, 0) for i in MARKED) / shots

print(f"Most frequent measured index: {most_likely_index} "
      f"({most_likely_freq}/{shots} shots)")
print(f"Fraction of shots landing on a classically-maximizing index: "
      f"{marked_shot_fraction:.3f}")

quantum_found_optimal = most_likely_index in MARKED
amplification_worked = marked_shot_fraction > (num_marked / N_STATES) * 2

verified = quantum_found_optimal and amplification_worked

if verified:
    print("PASS: Grover search on the ideal AerSimulator recovered a subset "
          "achieving the classically brute-forced maximum "
          f"({CLASSICAL_MAX} fully-contained 3-point lines) for this "
          "finite instance of Erdos problem #669's orchard-problem counting "
          "question (OEIS A003035 / A006065 / A008997).")
else:
    print("FAIL: quantum search result did not match the classical optimum "
          "or amplification was insufficient.")
