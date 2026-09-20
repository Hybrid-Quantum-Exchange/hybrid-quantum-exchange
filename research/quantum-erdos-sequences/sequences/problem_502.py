"""
Erdos problem #502 (erdosproblems.com), OEIS id: A027627
"Maximal cardinality of 2-distance sets in R^n" -- a(n) is the largest
number of points that can be placed in n-dimensional Euclidean space such
that the set of pairwise distances between them takes on exactly two
distinct values. A027627 records a(1)=3, a(2)=5, a(3)=6, a(4)=10, ...

Classical property tested here (finite, computable, derived from a(1)=3):
On the line R^1, take the 4 points p = [0, 1, 2, 5]. Among the four
3-point subsets of p (obtained by omitting one point at a time), exactly
ONE is a "2-distance set" (its 3 pairwise distances take exactly 2
distinct values); the other three are not (their pairwise distances take
3 distinct values). This is computed directly in this script by brute
force over all C(4,3) = 4 subsets -- no OEIS value is copied without
derivation, only the fact a(1)=3 (points on a line can realize a
2-distance set of size 3) motivated the instance.

Concretely, indexing the 4 subsets by which point (0,1,2,5) is omitted:
  omit 0 -> {1,2,5}: distances {1,4,3} -> 3 distinct values -> NOT
  omit 1 -> {0,2,5}: distances {2,5,3} -> 3 distinct values -> NOT
  omit 2 -> {0,1,5}: distances {1,5,4} -> 3 distinct values -> NOT
  omit 3 -> {0,1,2}: distances {1,2,1} -> 2 distinct values -> 2-distance set

So exactly 1 of the 4 subsets (index 3, i.e. binary '11') witnesses the
a(1)=3 fact that 3 points can form a 2-distance set on a line.

Quantum approach: Grover search over the 4 candidate subsets (2 qubits,
computational basis states |00>,|01>,|10>,|11> <-> subset indices 0..3),
with a phase oracle (built from the classically-computed marked set) that
flips the sign of the single marked basis state |11>. With N=4 possible
states and M=1 marked state, exactly ONE Grover iteration drives the
success probability to exactly 1 (a known exact special case of Grover's
algorithm: theta = 2*arcsin(sqrt(1/4)) = 60 degrees, and (2*1+1)*theta/2 =
90 degrees, so sin^2(90 degrees) = 1), so the ideal AerSimulator should
return only '11'.

This is run on qiskit_aer's ideal AerSimulator. PASS/FAIL is decided by
comparing the simulator's measured outcomes against the classically
brute-forced marked set.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator


def classical_two_distance_subsets(points):
    """Brute-force, from first principles, which 3-point subsets of
    `points` (a list of positions on the real line) are 2-distance sets
    (exactly 2 distinct pairwise distances). Returns a dict mapping the
    omitted-index (0..len(points)-1) to True/False."""
    n = len(points)
    all_indices = list(range(n))
    result = {}
    for omit in all_indices:
        subset_idx = [i for i in all_indices if i != omit]
        subset = [points[i] for i in subset_idx]
        dists = set()
        for a, b in combinations(subset, 2):
            dists.add(abs(a - b))
        result[omit] = (len(dists) == 2)
    return result


def build_grover_circuit(marked_indices, n_qubits=2):
    """Build a Grover search circuit over 2^n_qubits basis states, marking
    the computational basis states whose integer index is in
    `marked_indices`, using exactly 1 Grover iteration (exact for
    N=4, M=1)."""
    n_states = 2 ** n_qubits

    # Phase oracle: diagonal of +1/-1, -1 exactly at the marked indices.
    diag = [-1.0 if i in marked_indices else 1.0 for i in range(n_states)]
    oracle = DiagonalGate(diag)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    # --- one Grover iteration ---
    qc.append(oracle, range(n_qubits))

    # Diffuser: inversion about the mean.
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    # --- end iteration ---

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    points = [0, 1, 2, 5]
    classical = classical_two_distance_subsets(points)
    marked_indices = sorted(i for i, is_two_distance in classical.items() if is_two_distance)

    print("Erdos problem #502 / OEIS A027627 (max 2-distance set cardinality)")
    print(f"Points on the line: {points}")
    print("Per-subset (omitted index -> is 2-distance set):")
    for i in sorted(classical):
        print(f"  omit {i}: {classical[i]}")
    print(f"Classically marked (2-distance) subset indices: {marked_indices}")

    # a(1) = 3 for A027627 means 3 points CAN form a 2-distance set; we
    # expect a nonempty, proper subset of the 4 candidates to be marked.
    assert marked_indices, "expected at least one 2-distance 3-point subset"
    assert len(marked_indices) == 1, "expected exactly one 2-distance 3-point subset for this instance"

    qc = build_grover_circuit(marked_indices, n_qubits=2)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    print(f"\nGrover circuit measurement counts ({shots} shots): {counts}")

    # Qiskit bit order: classical bit c0 (LSB) is qubit 0. The subset
    # index i (0..3) is int(bitstring, 2) read with qubit0 as the LSB,
    # i.e. bitstring = format(i, '02b') with the *reversed* qubit order
    # that Qiskit prints (c1 c0). Decode explicitly and robustly:
    def bitstring_to_index(bs):
        # bs is Qiskit's default 'c1c0' order (MSB first as printed)
        c1, c0 = int(bs[0]), int(bs[1])
        return c0 + 2 * c1  # qubit0 -> bit0 (LSB), qubit1 -> bit1

    measured_indices = {bitstring_to_index(bs) for bs in counts}
    total_shots = sum(counts.values())
    marked_shots = sum(v for bs, v in counts.items() if bitstring_to_index(bs) in marked_indices)
    marked_fraction = marked_shots / total_shots

    print(f"Decoded measured subset indices: {sorted(measured_indices)}")
    print(f"Fraction of shots landing on classically-marked subsets: {marked_fraction:.4f}")

    # With N=4, M=1 and one exact Grover iteration, the ideal simulator
    # should place ~100% of amplitude (and therefore shots) on the single
    # marked state, and no other computational basis state should ever
    # be observed.
    quantum_matches_classical = (
        measured_indices == set(marked_indices)
        and marked_fraction > 0.99
    )

    if quantum_matches_classical:
        print("\nPASS: Grover search result matches the classically computed "
              "2-distance-set subsets.")
    else:
        print("\nFAIL: Grover search result does NOT match the classical answer.")

    return quantum_matches_classical


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
