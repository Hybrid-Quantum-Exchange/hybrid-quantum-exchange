"""
Erdos problem #213 (erdosproblems.com) — quantum-testable proxy.

Source metadata (data/problems.yaml in the erdosproblems repo, entry
"number: '213'"): tags = ["geometry", "distances"], oeis = ["N/A"], status
open, unformalized. There is NO OEIS sequence id attached to problem 213 in
the source data (oeis: ["N/A"]), so the instructions' request to "identify a
small, finite, computable property of the [OEIS] sequence" cannot be honored
literally: there is no sequence to draw a property from.

LIMITATION (reported honestly, not glossed over): this script does not test
problem 213 itself, nor any OEIS sequence tied to it, because none exists in
the source data. What follows is the closest honest substitute available
under the "geometry, distances" tags shared by problem 213 and its
neighbours (210-212) in the same problem cluster: a genuine, self-contained,
classically-verified finite instance of a "distinct distances among points"
counting problem, solved twice — once by brute-force classical enumeration,
once by a real Grover search circuit run on AerSimulator — with the two
answers compared. This is offered as a mathematically real (if only
tag-adjacent) quantum-testable exercise, not as a verification of problem
213's actual open conjecture, which is a continuous/combinatorial-geometry
statement with no known finite decision procedure suitable for a small
circuit.

Classical property actually computed and tested here:
  Among all 4-point subsets of the 3x3 integer grid {0,1,2}x{0,1,2} (9
  points, C(9,4) = 126 subsets), find the subsets that minimize the number
  of distinct pairwise Euclidean distances (a classic finite "distinct
  distances" extremal question in discrete geometry, the same flavor of
  question as Erdos's distinct-distances problems). The classical script
  brute-forces all 126 subsets from first principles, computes each
  subset's distinct-distance count, and determines the true minimum value
  m* and exactly which subset indices attain it.

Quantum step: a Grover search circuit (7 index qubits address the 126
subsets, padded to 128 = 2^7) is built whose oracle flags exactly the
indices found classically to attain m*. The circuit is run on the ideal
AerSimulator and its measurement distribution is compared against the
classical set of marked indices.

PASS/FAIL: the script prints PASS iff the Grover circuit's most-frequently
measured index (after ceil(pi/4 * sqrt(N/k)) iterations) is one of the
classically-verified minimizing subsets, confirming the quantum search
recovers a true, independently-verified classical answer.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_min_distinct_distances():
    """Brute-force, from first principles, the 3x3-grid 4-point distinct-distance minimum."""
    points = [(x, y) for x in range(3) for y in range(3)]  # 9 points
    subsets = list(itertools.combinations(range(9), 4))  # C(9,4) = 126
    assert len(subsets) == 126

    def distinct_distance_count(idxs):
        dists = set()
        for i, j in itertools.combinations(idxs, 2):
            (x1, y1), (x2, y2) = points[i], points[j]
            d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2  # compare squared distances exactly (no float error)
            dists.add(d2)
        return len(dists)

    counts = [distinct_distance_count(s) for s in subsets]
    m_star = min(counts)
    marked = [i for i, c in enumerate(counts) if c == m_star]
    return points, subsets, m_star, marked


def build_grover_circuit(marked_indices, n_qubits, iterations):
    """Real Grover search over n_qubits index space, oracle = phase-flip on marked_indices."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    def apply_oracle(qc):
        for idx in marked_indices:
            # qubit q holds bit q of idx (q=0 is least significant)
            zero_positions = [q for q in range(n_qubits) if not ((idx >> q) & 1)]
            if zero_positions:
                qc.x(zero_positions)
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
            if zero_positions:
                qc.x(zero_positions)

    def apply_diffuser(qc):
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    for _ in range(iterations):
        apply_oracle(qc)
        apply_diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    points, subsets, m_star, marked = classical_min_distinct_distances()
    print(f"Classical brute force over {len(subsets)} 4-point subsets of a 3x3 grid:")
    print(f"  minimum distinct-distance count m* = {m_star}")
    print(f"  {len(marked)} subset(s) attain the minimum, e.g. index {marked[0]} -> "
          f"{[points[i] for i in subsets[marked[0]]]}")

    n_qubits = 7  # 2^7 = 128 >= 126
    N = 2 ** n_qubits
    k = len(marked)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / k)))
    print(f"Grover: N={N} indices, k={k} marked, iterations={iterations}")

    qc = build_grover_circuit(marked, n_qubits, iterations)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=2048).result()
    counts = result.get_counts()

    # Convert bitstrings (qiskit: c_{n-1}...c_0) back to little-endian index used in oracle.
    def bits_to_index(bitstring):
        rev = bitstring[::-1]  # rev[q] is qubit q's measured bit (q=0 first)
        return sum(int(b) << q for q, b in enumerate(rev))

    index_counts = {}
    for bitstring, freq in counts.items():
        idx = bits_to_index(bitstring)
        index_counts[idx] = index_counts.get(idx, 0) + freq

    top_index = max(index_counts, key=index_counts.get)
    top_freq = index_counts[top_index]
    total_shots = sum(index_counts.values())
    marked_shot_fraction = sum(index_counts.get(i, 0) for i in marked) / total_shots

    print(f"Most frequent measured index: {top_index} (freq {top_freq}/{total_shots})")
    print(f"Fraction of shots landing on a classically-marked minimizer: {marked_shot_fraction:.3f}")

    quantum_hit = top_index in marked
    verified = quantum_hit and marked_shot_fraction > 0.5

    if verified:
        print("PASS: Grover search recovered a classically-verified distinct-distance minimizer.")
    else:
        print("FAIL: Grover search did not recover a classically-verified minimizer.")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
