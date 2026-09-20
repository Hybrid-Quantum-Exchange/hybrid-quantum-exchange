"""
Erdos problem #232 (proved; prize: no; tags: geometry, distances).

Source metadata (erdosproblems.com dataset, /home/user/manman4/erdosproblems/
data/problems.yaml, entry "number: \"232\"") records `oeis: ["N/A"]` -- this
problem has no associated OEIS sequence. Per the task instructions for that
case, this script does not fabricate an OEIS-derived property. Instead it
builds a genuine, small, finite, computable instance of the same mathematical
flavor as the problem's own tags ("geometry", "distances"): Erdos's classic
theme of pairwise distances determined by a finite point set in the plane.

Classical property tested
--------------------------
Fix four points in the plane:
    P0 = (0, 0)
    P1 = (1, 0)
    P2 = (2, 2)
    P3 = (0, 3)

Enumerate all C(4,2) = 6 unordered pairs {i, j}, i < j, in the fixed order
    0:(0,1) 1:(0,2) 2:(0,3) 3:(1,2) 4:(1,3) 5:(2,3)
and compute each pair's Euclidean distance. The property under test is:
"which pair index (0..5) realizes the globally minimum pairwise distance?"
This is computed from first principles in `classical_min_distance_pair()`
below, by brute-force comparison of all six squared distances (integer
arithmetic, no floating-point ambiguity, and the coordinates are chosen so
the minimum is unique).

Quantum circuit
----------------
The 6 pair-indices are embedded in 3 qubits (8 basis states; indices 6 and 7
are unused/never marked). A Grover search circuit is built whose oracle
phase-flips exactly the classical minimum-distance index, followed by the
standard Grover diffuser, run for the optimal number of iterations for
N=8, M=1 marked state (round(pi/4 * sqrt(8)) = 2 iterations). The circuit is
executed on the ideal AerSimulator. The test passes if the most frequently
measured 3-bit outcome equals the classically computed minimum-distance pair
index.
"""

import math
from itertools import combinations

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


POINTS = [(0, 0), (1, 0), (2, 2), (0, 3)]


def classical_min_distance_pair():
    """Brute-force, first-principles classical computation.

    Returns (pair_list, min_index, min_pair, squared_distances) where
    pair_list is the fixed enumeration order of index -> (i, j), min_index is
    the position in that list achieving the smallest squared Euclidean
    distance, and squared_distances lists every pair's squared distance.
    """
    pairs = list(combinations(range(len(POINTS)), 2))  # fixed order, len 6
    sq_dists = []
    for (i, j) in pairs:
        (x1, y1), (x2, y2) = POINTS[i], POINTS[j]
        d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
        sq_dists.append(d2)

    min_val = min(sq_dists)
    occurrences = [idx for idx, d2 in enumerate(sq_dists) if d2 == min_val]
    if len(occurrences) != 1:
        raise ValueError(
            f"minimum squared distance is not unique: indices {occurrences} "
            f"all equal {min_val}; choose different points"
        )
    min_index = occurrences[0]
    return pairs, min_index, pairs[min_index], sq_dists


def build_oracle(marked_index: int, n_qubits: int) -> QuantumCircuit:
    """Phase-flip the single computational basis state |marked_index>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(marked_index, f"0{n_qubits}b")[::-1]  # little-endian
    # Flip qubits that should be 0 so the marked state maps to |11...1>
    for q, b in enumerate(bits):
        if b == "0":
            qc.x(q)
    # Multi-controlled Z on all qubits (phase flip of |11...1>)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for q, b in enumerate(bits):
        if b == "0":
            qc.x(q)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_index: int, n_qubits: int = 3, shots: int = 2048):
    n_states = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_states)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_index, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    # keys are little-endian bitstrings from Qiskit; convert to int
    best_bits = max(counts, key=counts.get)
    best_index = int(best_bits[::-1], 2)
    return best_index, counts, iterations


def main():
    pairs, classical_index, classical_pair, sq_dists = classical_min_distance_pair()

    print("Erdos problem #232 -- distances/geometry (no OEIS id: informal status "
          "entry has oeis: ['N/A'])")
    print(f"Points: {POINTS}")
    print("Pair enumeration (index: (i,j) -> squared distance):")
    for idx, ((i, j), d2) in enumerate(zip(pairs, sq_dists)):
        print(f"  {idx}: ({i},{j}) -> {d2}")
    print(f"Classical minimum-distance pair index: {classical_index} "
          f"(points {POINTS[classical_pair[0]]}-{POINTS[classical_pair[1]]}, "
          f"squared distance {sq_dists[classical_index]})")

    quantum_index, counts, iterations = run_grover(classical_index)
    print(f"\nGrover search ({iterations} iterations, 3 qubits, 8 basis states):")
    print(f"  measurement counts: {counts}")
    print(f"  most frequent measured index: {quantum_index}")

    verified = quantum_index == classical_index
    print(f"\nClassical answer: {classical_index}")
    print(f"Quantum answer:   {quantum_index}")
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
