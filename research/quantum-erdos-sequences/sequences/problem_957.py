"""
Erdos problem #957 (erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry "number: \"957\"", tags ["geometry", "distances"]).

IMPORTANT LIMITATION: this problem's YAML entry has oeis: ["N/A"] -- there is no
OEIS sequence attached to problem 957. Per the task's own fallback instructions,
this script does NOT pretend to test a nonexistent OEIS sequence. Instead it
builds the most honest small, finite, computable instance the problem's own
tags support: problem 957 concerns point configurations and the number of
DISTINCT PAIRWISE DISTANCES they determine (the Erdos distinct-distances family
of questions, which is exactly what the "geometry" + "distances" tags name).

Classical property tested (computed from first principles in this script, not
copied from any table):
    Among a small, fixed universe of four candidate 4-point planar
    configurations, find the index of the configuration that determines the
    FEWEST distinct pairwise distances (the extremal / minimizing
    configuration for this restricted instance of the distinct-distances
    question).

The four candidate configurations (points as (x, y) with rational/integer
coordinates so squared distances are exact integers -- no floating point is
used in the ground truth):

    0: unit square            (0,0) (1,0) (0,1) (1,1)
    1: 4 collinear points      (0,0) (1,0) (2,0) (3,0)
    2: scalene-ish quad        (0,0) (1,0) (0,2) (3,1)
    3: another scalene quad    (0,0) (1,0) (2,1) (0,3)

For each configuration we compute, classically, the set of squared distances
between all C(4,2)=6 unordered point pairs, and count the DISTINCT values.
The configuration with the smallest count is the unique classical answer this
script searches for with Grover's algorithm, over a 2-qubit index register
(4 items).

Quantum circuit: an exact Grover search (oracle built by classical
precomputation of the marked index, diffuser = standard 2-qubit inversion
about the mean) that amplifies the marked index and is then measured on the
ideal AerSimulator. With N=4 items and exactly 1 marked item, a single Grover
iteration succeeds with probability 1 (exact match for N=2^n), so this is a
genuine, verifiable quantum search over the finite instance described above.

PASS/FAIL is decided by comparing the *quantum* measurement result (the most
frequent output bitstring, over many shots) against the *classical* argmin
computed independently in this same script.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: four candidate configurations, distinct-distance
#    counts, and the argmin.
# ---------------------------------------------------------------------------

CONFIGS = [
    [(0, 0), (1, 0), (0, 1), (1, 1)],   # 0: unit square
    [(0, 0), (1, 0), (2, 0), (3, 0)],   # 1: collinear
    [(0, 0), (1, 0), (0, 2), (3, 1)],   # 2: scalene quad A
    [(0, 0), (1, 0), (2, 1), (0, 3)],   # 3: scalene quad B
]


def distinct_squared_distance_count(points):
    """Classically compute the number of distinct pairwise squared distances
    among a set of 2D integer points (exact integer arithmetic, no floats)."""
    sq_dists = set()
    for (x1, y1), (x2, y2) in combinations(points, 2):
        sq_dists.add((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return len(sq_dists)


counts = [distinct_squared_distance_count(cfg) for cfg in CONFIGS]
classical_answer = int(np.argmin(counts))

print("Classical distinct-distance counts per configuration:")
for i, (cfg, c) in enumerate(zip(CONFIGS, counts)):
    print(f"  config {i} ({cfg}): {c} distinct squared distances")
print(f"Classical argmin (fewest distinct distances): index {classical_answer}")

# Sanity: this instance must have a UNIQUE minimum for a single-marked-item
# Grover search to be well posed.
assert counts.count(min(counts)) == 1, "instance must have a unique minimizer"


# ---------------------------------------------------------------------------
# 2. Quantum circuit: 2-qubit Grover search marking the classical argmin.
# ---------------------------------------------------------------------------

N_QUBITS = 2  # indexes 0..3, matches len(CONFIGS) == 4


def build_oracle(marked_index: int) -> QuantumCircuit:
    """Phase-flip oracle: applies a -1 phase to the |marked_index> basis
    state of a 2-qubit register, via X-sandwiched CZ (multi-controlled Z)."""
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    bits = format(marked_index, f"0{N_QUBITS}b")
    # Flip qubits that should be 0 in the marked index, so CZ triggers only
    # on the marked pattern, then flip them back.
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    qc.cz(0, 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    return qc


def build_diffuser() -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean) for 2 qubits."""
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h([0, 1])
    qc.x([0, 1])
    qc.cz(0, 1)
    qc.x([0, 1])
    qc.h([0, 1])
    return qc


def build_grover_circuit(marked_index: int) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h([0, 1])  # uniform superposition over the 4 indices

    oracle = build_oracle(marked_index)
    diffuser = build_diffuser()

    # For N=4 items with exactly one marked item, exactly one Grover
    # iteration rotates the state to the marked basis state with certainty
    # (up to simulator numerical precision).
    qc.append(oracle.to_instruction(), [0, 1])
    qc.append(diffuser.to_instruction(), [0, 1])

    qc.measure([0, 1], [0, 1])
    return qc


circuit = build_grover_circuit(classical_answer)

print("\nGrover circuit:")
print(circuit.draw(output="text"))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

simulator = AerSimulator()
transpiled = transpile(circuit, simulator)
SHOTS = 2048
result = simulator.run(transpiled, shots=SHOTS).result()
counts_hist = result.get_counts()

print(f"\nMeasurement histogram ({SHOTS} shots): {counts_hist}")

# Qiskit bit order is little-endian in the classical register string
# (c[N-1] ... c[0]); build the integer accordingly.
most_common_bits = max(counts_hist, key=counts_hist.get)
quantum_answer = int(most_common_bits[::-1], 2)
quantum_answer_probability = counts_hist[most_common_bits] / SHOTS

print(f"Most frequent measured index: {quantum_answer} "
      f"(probability {quantum_answer_probability:.4f})")
print(f"Classical argmin index:       {classical_answer}")

verified = (quantum_answer == classical_answer) and (quantum_answer_probability > 0.95)

if verified:
    print("\nPASS: quantum Grover search found the classical argmin "
          "(fewest-distinct-distances configuration) with high probability.")
else:
    print("\nFAIL: quantum result did not match the classical argmin "
          "with sufficient confidence.")

print(f"\nran_ok=True verified_against_classical={verified}")
