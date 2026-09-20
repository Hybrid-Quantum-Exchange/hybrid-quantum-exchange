"""
Erdos problem #93 (source: manman4/erdosproblems data/problems.yaml,
number: "93", tags: ["geometry", "convex", "distances"], oeis: ["N/A"]).

LIMITATION, stated up front: problem #93's entry in problems.yaml carries no
OEIS sequence id ("N/A"). There is therefore no OEIS-defined "sequence" to
target with a small quantum circuit, contrary to the general recipe for this
library. To still produce a genuine, honest, finite instance in the spirit of
the problem's own tags (distances determined by a finite point set), this
script defines and classically verifies the following small, self-contained,
computable property, then searches for it with a real Grover circuit:

    Property tested: among the four 3-point subsets ("triangles") of the
    4-point set
        P0 = (0, 0)
        P1 = (2, 0)
        P2 = (1, sqrt(3))
        P3 = (5, 7)
    exactly one subset has the MINIMUM POSSIBLE number of distinct pairwise
    distances a triangle can have, namely 1 (an equilateral triangle -- P0,
    P1, P2 by construction). The other three subsets (each of which drops one
    of P0, P1, P2 and keeps P3) are scalene and have 3 distinct pairwise
    distances. This is exactly the "how many distinct distances can a finite
    point set determine" question the problem's tags (geometry / convex /
    distances) are about, instantiated at the smallest interesting size (a
    triangle): the achievable counts are 1, 2, or 3, and this instance
    realizes the extremal case 1 for one subset out of four.

    The four subsets are indexed by "which point is OMITTED", 0..3, encoded
    as a 2-qubit basis state |idx>. The classical answer (computed from first
    principles below, by literally enumerating each subset's three pairwise
    Euclidean distances and counting distinct values up to floating-point
    tolerance) is idx = 3 (the subset omitting P3, i.e. {P0, P1, P2}).

Circuit: a textbook 2-qubit Grover search (1 iteration is optimal for 1
marked item out of 4) whose oracle marks basis state |11> (idx=3, the unique
equilateral triangle) via a controlled-Z, followed by the standard 2-qubit
diffusion operator. Run on the ideal AerSimulator (statevector method,
noiseless), sampled with many shots. PASS iff the most frequent measured
outcome equals the classically-computed idx, with all its probability mass
(Grover on a single marked item out of 4, 1 iteration, is exact -- amplitude
of the marked state is 1 up to floating point).
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_answer():
    """Enumerate the four 3-point subsets, count distinct pairwise distances
    in each, and return the index (which point is omitted) of the unique
    subset achieving the minimum possible count (1, i.e. equilateral)."""
    points = [
        (0.0, 0.0),
        (2.0, 0.0),
        (1.0, math.sqrt(3.0)),
        (5.0, 7.0),
    ]
    idx_all = [0, 1, 2, 3]
    results = {}
    for omit in idx_all:
        subset = [points[i] for i in idx_all if i != omit]
        dists = []
        for a, b in itertools.combinations(subset, 2):
            d = math.hypot(a[0] - b[0], a[1] - b[1])
            dists.append(d)
        # count distinct distances up to floating point tolerance
        distinct = []
        for d in dists:
            if not any(math.isclose(d, e, rel_tol=1e-9, abs_tol=1e-9) for e in distinct):
                distinct.append(d)
        results[omit] = len(distinct)

    min_count = min(results.values())
    winners = [omit for omit, c in results.items() if c == min_count]
    assert min_count == 1, f"expected minimum distinct-distance count 1, got {min_count}"
    assert winners == [3], f"expected the unique minimizer to be idx=3 (omit P3), got {winners}"
    return winners[0], results


def build_grover_circuit(marked_idx: int) -> QuantumCircuit:
    """2-qubit Grover search, oracle marks |marked_idx> (0..3), 1 iteration
    (optimal for 1 marked item out of N=4)."""
    assert 0 <= marked_idx <= 3
    qc = QuantumCircuit(2, 2)

    # uniform superposition
    qc.h(0)
    qc.h(1)

    # --- oracle: phase-flip the basis state |marked_idx> ---
    bits = format(marked_idx, "02b")  # bits[0] -> qubit1 (MSB), bits[1] -> qubit0 (LSB)
    # X on any qubit whose corresponding bit of marked_idx is 0, so that the
    # target state maps to |11> before the controlled-Z
    if bits[1] == "0":  # qubit 0
        qc.x(0)
    if bits[0] == "0":  # qubit 1
        qc.x(1)
    qc.cz(0, 1)
    if bits[1] == "0":
        qc.x(0)
    if bits[0] == "0":
        qc.x(1)

    # --- diffusion (inversion about the mean) ---
    qc.h(0)
    qc.h(1)
    qc.x(0)
    qc.x(1)
    qc.cz(0, 1)
    qc.x(0)
    qc.x(1)
    qc.h(0)
    qc.h(1)

    qc.measure([0, 1], [0, 1])
    return qc


def run():
    marked_idx, per_subset_counts = classical_answer()
    print(f"Classical: per-subset distinct-distance counts = {per_subset_counts}")
    print(f"Classical: unique minimizer (equilateral subset) idx = {marked_idx}")

    qc = build_grover_circuit(marked_idx)

    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    print(f"Quantum: measurement counts over {shots} shots = {counts}")

    # bit order in qiskit counts is c1c0 (qubit1 qubit0), matching our
    # 'bits' encoding above (bits[0]=qubit1 MSB, bits[1]=qubit0 LSB)
    best_outcome = max(counts, key=counts.get)
    measured_idx = int(best_outcome, 2)
    measured_prob = counts[best_outcome] / shots

    print(f"Quantum: most frequent outcome = {best_outcome} -> idx {measured_idx} "
          f"(probability ~{measured_prob:.4f})")

    ok = (measured_idx == marked_idx) and (measured_prob > 0.99)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    success = run()
    if not success:
        raise SystemExit(1)
