"""
Erdos problem #958 -- quantum-testable instance.

Source metadata (data/problems.yaml in the erdosproblems repo, entry
`number: "958"`): tags = ["distances", "geometry"], oeis = ["N/A"],
status = "disproved (Lean)". There is NO OEIS sequence id attached to
problem 958 -- the yaml literally records "N/A". Per the task's own
instructions ("do not just copy a literal OEIS value without deriving
it"; "if no OEIS id ... write the script anyway with your best honest
attempt, note the limitation clearly"), this script cannot test a
genuine OEIS-958 term, because no such sequence exists to test.

LIMITATION (stated honestly): this is not a verification of problem
958's own (unspecified) mathematical content. Instead, since the
problem's tags are "distances" and "geometry" -- the Erdos
distinct-distances family -- this script builds a real, small,
finite, classically-checkable instance from that same family: the
minimum number of *distinct pairwise distances* achievable by 3 points
chosen from 5 collinear integer positions {0,1,2,3,4}. This is a
genuine combinatorial-geometry quantity (the discrete analogue of the
Erdos distinct-distances problem), computed here from first principles
by brute force, then re-derived with a real Grover search circuit run
on Qiskit's AerSimulator. It is offered as the closest honest,
computable stand-in given problem 958 carries no OEIS id -- not as a
literal test of problem 958's disproved geometric statement.

Classical part
--------------
SPACE = all 3-element subsets of {0,1,2,3,4} (C(5,3) = 10 subsets),
indexed 0..9. For each subset (as points on a line) we compute the
number of *distinct* pairwise distances among its 3 points. We find
the minimum such count over the whole space, and the set of subset
indices ("winners") achieving it. This minimum and the winner set are
computed directly by brute force in this script -- no external table
is consulted.

Quantum part
------------
We build a Grover search circuit over a 4-qubit index register
(16 basis states, 10 of which are valid subset indices 0..9; the
remaining 6 are simply never marked). The oracle marks exactly the
winner indices found classically. We run the standard Grover
diffusion operator for the optimal number of iterations for this
search-space size, execute on AerSimulator, and take the
most-frequently-measured index as the quantum answer.

PASS/FAIL
---------
The script prints PASS if the most-probable quantum measurement
decodes to one of the classically-computed winner indices, and FAIL
otherwise.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical part: brute-force the minimum-distinct-distances subsets.
# ---------------------------------------------------------------------------

POSITIONS = [0, 1, 2, 3, 4]
SUBSETS = list(itertools.combinations(POSITIONS, 3))  # 10 subsets, index 0..9
N_INDEX_QUBITS = 4  # 2^4 = 16 >= 10


def distinct_distance_count(subset):
    pts = subset
    dists = set()
    for a, b in itertools.combinations(pts, 2):
        dists.add(abs(a - b))
    return len(dists)


def classical_answer():
    counts = [distinct_distance_count(s) for s in SUBSETS]
    min_count = min(counts)
    winners = [i for i, c in enumerate(counts) if c == min_count]
    return min_count, winners, counts


MIN_COUNT, WINNERS, ALL_COUNTS = classical_answer()

print("Erdos problem #958 -- quantum-testable sequence lane")
print("Subsets (index: points -> distinct distance count):")
for i, s in enumerate(SUBSETS):
    marker = "  <-- winner" if i in WINNERS else ""
    print(f"  {i}: {s} -> {ALL_COUNTS[i]}{marker}")
print(f"Classical minimum distinct-distance count = {MIN_COUNT}")
print(f"Classical winner indices = {WINNERS}")


# ---------------------------------------------------------------------------
# 2. Quantum part: Grover search for the winner indices.
# ---------------------------------------------------------------------------

def int_to_bits(x, n):
    return [(x >> k) & 1 for k in range(n)]


def build_oracle(n_qubits, marked_indices):
    """Phase-flip oracle marking each index in marked_indices."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = int_to_bits(idx, n_qubits)
        # Flip qubits that are 0 in this index so the all-ones pattern
        # corresponds to "current basis state == idx".
        for q, b in enumerate(bits):
            if b == 0:
                qc.x(q)
        # Multi-controlled Z on all qubits (phase flip when all are |1>).
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q, b in enumerate(bits):
            if b == 0:
                qc.x(q)
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


def build_grover_circuit(n_qubits, marked_indices, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(n_qubits, marked_indices)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


N_STATES = 2 ** N_INDEX_QUBITS  # 16
n_marked = len(WINNERS)
# Standard optimal Grover iteration count for M marked out of N states.
iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N_STATES / n_marked)))

circuit = build_grover_circuit(N_INDEX_QUBITS, WINNERS, iterations)

simulator = AerSimulator()
compiled = transpile(circuit, simulator)
job = simulator.run(compiled, shots=4096)
result = job.result()
counts = result.get_counts()

# Qiskit bit ordering: rightmost classical bit is qubit 0 (LSB) already,
# since we measured qubit k -> classical bit k with matching indices.
def bitstring_to_index(bitstring):
    # Qiskit prints classical bits MSB-first (leftmost char = clbit n-1),
    # and clbit k was measured from qubit k, so interpreting the string
    # directly as a binary number already gives sum_k qubit_k * 2**k.
    return int(bitstring, 2)

decoded_counts = {}
for bitstring, freq in counts.items():
    idx = bitstring_to_index(bitstring)
    decoded_counts[idx] = decoded_counts.get(idx, 0) + freq

most_likely_index = max(decoded_counts, key=decoded_counts.get)
most_likely_freq = decoded_counts[most_likely_index]

print(f"\nGrover circuit: {N_INDEX_QUBITS} index qubits, {iterations} iteration(s), 4096 shots")
print("Top measured indices (index: shots):")
for idx, freq in sorted(decoded_counts.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {idx}: {freq}")
print(f"Most probable quantum answer: index {most_likely_index} ({most_likely_freq}/4096 shots)")

quantum_matches_classical = most_likely_index in WINNERS

print(f"\nClassical winners: {WINNERS}")
print(f"Quantum most-likely index: {most_likely_index}")

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
