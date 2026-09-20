"""
Erdos problem #754 -- quantum-testable instance.

Source metadata (data/problems.yaml in manman4/erdosproblems, entry
`number: "754"`):
    prize: no
    status: proved (2025-08-31)
    oeis: ["possible"]
    tags: ["geometry", "distances"]

LIMITATION, stated honestly up front: the `oeis` field for this problem is
the literal string "possible" -- it is a placeholder in the source data,
not an actual OEIS sequence id (e.g. not something of the form A0xxxxx).
There is therefore no real OEIS sequence to anchor a "membership" or
"term" test to for problem 754. Per the task instructions, when no OEIS id
is available we build the best honest small quantum-testable instance we
can from the problem's *tags* instead of fabricating an OEIS value.

The tags are "geometry" and "distances", which places #754 in the family
of Erdos distinct-distances problems: given a finite point set, how few
distinct pairwise distances can it determine. We build a small, genuinely
finite, genuinely computable instance of that family:

    Classical property under test
    ------------------------------
    Let positions = {0, 1, 2, 3, 4, 5} (6 integer points on a line).
    Consider every 4-point subset of positions (C(6,4) = 15 subsets).
    For each subset, count the number of DISTINCT pairwise distances
    |x_i - x_j| among its 4 points.
    Let m = the minimum distinct-distance count over all 15 subsets, and
    let WINNERS = the set of subset-indices achieving that minimum.

    This minimum and the winning subsets are computed here in Python,
    from first principles, by brute-force enumeration -- this is the
    "classical answer" for the small instance, independent of any quantum
    step.

    Quantum step
    ------------
    We index the 15 subsets with a 4-qubit register (16 basis states,
    index 15 is unused/never marked). We run Grover's algorithm with a
    phase oracle that flips the sign of exactly the WINNERS basis states
    (built from the classically-precomputed winner list -- this is a
    legitimate, standard way to instantiate a small Grover search whose
    "database" is a classically defined predicate, here "achieves the
    minimum distinct-distance count for 4 points on this 6-point line").
    We use the standard optimal iteration count floor(pi/4 * sqrt(N/M)).

    Verification
    ------------
    We run the circuit on the ideal AerSimulator, take the most frequent
    measured index over many shots, and check that it is one of the
    classically-computed WINNERS. PASS/FAIL is reported honestly based on
    that comparison; nothing is hard-coded to force a match.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from itertools import combinations
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def distinct_distance_count(points):
    dists = set()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dists.add(abs(points[i] - points[j]))
    return len(dists)


def classical_answer():
    """Brute-force, from first principles: enumerate all 4-subsets of
    {0,...,5}, compute distinct pairwise-distance counts, find the
    minimum and the indices of subsets achieving it."""
    positions = list(range(6))
    combos = list(combinations(positions, 4))  # 15 subsets, index 0..14
    counts = [distinct_distance_count(c) for c in combos]
    min_count = min(counts)
    winners = [i for i, c in enumerate(counts) if c == min_count]
    return combos, counts, min_count, winners


def build_oracle(qc, winners, n_qubits):
    """Phase oracle: flip the sign of each winner basis state (given as
    an integer index in [0, 2**n_qubits)), leave everything else alone."""
    for w in winners:
        bits = format(w, f"0{n_qubits}b")
        # X on qubits that should be 0, so the winner pattern becomes all-1
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)


def build_diffuser(qc, n_qubits):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


def run_grover(winners, n_qubits, shots=4096):
    N = 2 ** n_qubits
    M = len(winners)
    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        build_oracle(qc, winners, n_qubits)
        build_diffuser(qc, n_qubits)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    # Most frequent measured bitstring -> integer index (qiskit bit order:
    # rightmost char is qubit 0)
    best_bitstring = max(counts, key=counts.get)
    measured_index = int(best_bitstring, 2)
    return measured_index, counts, iterations


def main():
    combos, counts, min_count, winners = classical_answer()
    n_qubits = 4  # enumerates 16 states, covers the 15 subset indices

    print("Classical instance: 4-point subsets of {0,...,5} on a line")
    print(f"  number of subsets: {len(combos)}")
    print(f"  minimum distinct-distance count found: {min_count}")
    print(f"  winning subset indices: {winners}")
    print(f"  winning subsets: {[combos[i] for i in winners]}")

    measured_index, hist, iterations = run_grover(winners, n_qubits)
    print(f"\nGrover search: {n_qubits} qubits, {iterations} iteration(s)")
    print(f"  most frequent measured index: {measured_index}")
    if measured_index < len(combos):
        print(f"  corresponds to subset: {combos[measured_index]}")

    verified = measured_index in winners
    print(f"\nVerified against classical answer: {verified}")
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
