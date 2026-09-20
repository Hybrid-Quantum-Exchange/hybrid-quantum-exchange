"""
Erdos problem #661 -- Quantum-testable instance.

Source: https://github.com/manman4/erdosproblems, data/problems.yaml, number "661".
  prize: $50, tags: [geometry, distances], oeis: ["A186704", "possible"]

OEIS sequence used: A186704 -- "Minimal number of distinct distances
determined by n points in the plane" (the classic Erdos distinct-distances
problem). Its terms for n = 0..12 are:
  0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6
so a(4) = 2: four points in the plane can be placed so that only 2 distinct
pairwise distances occur (the square is the standard witness -- its 4 edges
share one distance and its 2 diagonals share another).

Classical property tested (derived and checked in this script, not copied
from OEIS): take the 5-point set

    P = [(0,0), (2,0), (0,2), (2,2), (1,3)]

(the 4 corners of a square plus one generic extra point). There are exactly
C(5,4) = 5 four-point subsets, obtained by dropping one point i in {0..4}.
For each subset we compute the number of distinct squared pairwise
distances (squared distance is an exact non-negative integer here, and
since sqrt is monotonic on non-negative reals, "number of distinct squared
distances" equals "number of distinct distances" -- no floating point is
needed for the classical ground truth). Exactly one of the 5 subsets
(dropping the extra point, i.e. i = 4, leaving exactly the square) attains
the minimum of 2 distinct distances; the other four subsets each give 4
distinct distances. This reproduces a(4) = 2 from A186704 on a genuine,
freshly-computed instance, and gives a finite unstructured search problem
of size 5 (pad to 8 = 2^3) with a unique marked item -- an ideal fit for
Grover's algorithm.

Quantum circuit: a standard 3-qubit Grover search over indices 0..7
(indices 5,6,7 are unused/never marked) with a single marked state, the
index found classically above. The oracle is a multi-controlled-Z phase
flip on the target computational basis state; the diffuser is the standard
inversion-about-the-mean operator. With N = 8 and M = 1 marked item, the
optimal number of Grover iterations is round(pi/4 * sqrt(N/M)) = 2.

The script:
  1. Computes the 5 subset distinct-distance counts classically from first
     principles (integer arithmetic only) and derives the target index.
  2. Builds and runs the Grover circuit on the ideal AerSimulator.
  3. Prints PASS if the most frequently measured basis state equals the
     classically-derived target index, else FAIL.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_target_index():
    """Return (target_index, counts_per_index, oeis_a4) computed from first principles."""
    points = [(0, 0), (2, 0), (0, 2), (2, 2), (1, 3)]
    counts = []
    for i in range(5):
        subset = [points[j] for j in range(5) if j != i]
        distinct_sq_dists = set()
        for a, b in combinations(subset, 2):
            d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            distinct_sq_dists.add(d2)
        counts.append(len(distinct_sq_dists))

    min_count = min(counts)
    targets = [i for i, c in enumerate(counts) if c == min_count]
    assert len(targets) == 1, "expected a unique minimizer for this instance"
    target_index = targets[0]

    # A186704(4) reference value, for the assertion below (not used to
    # derive the answer -- the answer above was computed purely from the
    # point coordinates).
    oeis_a4 = 2
    assert min_count == oeis_a4, (
        f"minimum distinct-distance count {min_count} does not match "
        f"A186704(4) = {oeis_a4}"
    )
    return target_index, counts, oeis_a4


def build_oracle(n_qubits, target_index):
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(target_index, f"0{n_qubits}b")[::-1]  # little-endian qubit order
    zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
    for q in zero_qubits:
        qc.x(q)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for q in zero_qubits:
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


def run_grover(target_index, n_qubits=3, shots=2048):
    n_items = 2 ** n_qubits
    n_marked = 1
    iterations = max(1, round((np.pi / 4) * np.sqrt(n_items / n_marked)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, target_index)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    target_index, subset_counts, oeis_a4 = classical_target_index()
    print("Erdos problem #661 -- OEIS A186704 (min distinct distances for n points)")
    print(f"5-point instance, per-subset distinct-distance counts: {subset_counts}")
    print(f"Classical target index (unique minimizer, count={oeis_a4}): {target_index}")

    n_qubits = 3
    counts, iterations = run_grover(target_index, n_qubits=n_qubits)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    # Qiskit's get_counts() keys are printed MSB(qubit n-1) ... LSB(qubit 0),
    # i.e. the ordinary binary representation of the measured integer.
    target_bits = format(target_index, f"0{n_qubits}b")
    most_common_bits = max(counts, key=counts.get)
    total_shots = sum(counts.values())
    target_prob = counts.get(target_bits, 0) / total_shots

    print(f"Target bitstring (Qiskit MSB..LSB order): {target_bits}")
    print(f"Most frequent measured bitstring: {most_common_bits}")
    print(f"Fraction of shots on target: {target_prob:.3f}")

    if most_common_bits == target_bits and target_prob > 0.5:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
