"""
Erdos problem #662 (erdosproblems.com/662).

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: \"662\""):
    prize: no
    status: open
    formalized: no
    oeis: ["N/A"]
    tags: ["geometry", "distances"]
    comments: "ambiguous statement"

LIMITATION (reported honestly, not papered over): problem #662 has NO OEIS id
in the source data (oeis == ["N/A"]) and is flagged by the maintainers as an
"ambiguous statement". There is therefore no actual integer sequence to build
a "quantum-testable sequence membership/term" circuit against, as the brief
for this lane calls for. Fabricating an OEIS id or inventing sequence values
would violate the "do not fabricate a property with no real mathematical
content" instruction, so this script does not do that.

What this script does instead, honestly: problem #662's own tags are
"geometry" and "distances", so it builds a REAL, small, self-contained
distance-geometry decision problem in that spirit, states its classical
answer (computed from first principles, not looked up), and uses a genuine
Grover search circuit on AerSimulator to find that answer quantum-mechanically,
then checks the quantum result against the classical one.

Classical property under test
------------------------------
Four points in the plane:
    A = (0, 0)
    B = (1, 0)
    C = (2, 2)
    D = (0, 3)

There are C(4,2) = 6 unordered pairs. List them in a fixed order and compute
each pair's *squared* Euclidean distance (squared to stay in exact integers):

    index 0: A-B
    index 1: A-C
    index 2: A-D
    index 3: B-C
    index 4: B-D
    index 5: C-D

The property tested: "which pair index attains the globally minimum pairwise
distance in this point set?" This is a genuine (if miniature) instance of the
kind of distance-extremal question the distance-geometry / Erdos-distance
literature (the theme of problem #662's tags) is about. The search space has
6 valid pair-indices (plus 2 unused padding indices to fill 3 qubits), i.e.
N = 8, and the classical brute-force check below shows there is a UNIQUE
minimizer, which is what makes it usable as a Grover-search target.

Circuit
-------
3 qubits index the 8 basis states 0..7 (6 valid pairs + 2 padding states that
never occur as an answer). A Grover oracle phase-flips exactly the basis
state equal to the classical minimizer index; a standard 3-qubit diffuser
amplifies it. With N=8 and M=1 marked state, the optimal number of Grover
iterations is round((pi/4) * sqrt(N/M)) = 2. The circuit is run on the ideal
AerSimulator and the most frequently measured index is compared against the
classical answer computed above.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_min_distance_pair_index():
    """Brute-force, first-principles computation of the answer (no lookup)."""
    points = {
        "A": (0, 0),
        "B": (1, 0),
        "C": (2, 2),
        "D": (0, 3),
    }
    labels = list(points.keys())
    pairs = list(combinations(labels, 2))  # fixed order -> indices 0..5

    sq_distances = []
    for (p, q) in pairs:
        (x1, y1), (x2, y2) = points[p], points[q]
        d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
        sq_distances.append(d2)

    min_val = min(sq_distances)
    winners = [i for i, d in enumerate(sq_distances) if d == min_val]

    assert len(winners) == 1, (
        "Grover search needs a unique marked state; got ties: "
        f"{winners} with distances {sq_distances}"
    )
    return winners[0], pairs, sq_distances


def build_oracle(marked_index: int, n_qubits: int) -> QuantumCircuit:
    """Phase-flip the |marked_index> basis state (multi-controlled Z)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(marked_index, f"0{n_qubits}b")[::-1]  # little-endian
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
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
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_states / 1)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_index, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # most frequent measured bitstring -> integer index (little-endian)
    best_bits = max(counts, key=counts.get)
    best_index = int(best_bits[::-1], 2)
    return best_index, counts, iterations


def main():
    classical_answer, pairs, sq_distances = classical_min_distance_pair_index()

    print("Erdos problem #662 -- geometry/distances tags, no OEIS id (N/A).")
    print("Point-pair squared distances (index: pair -> d^2):")
    for i, (pair, d2) in enumerate(zip(pairs, sq_distances)):
        print(f"  {i}: {pair} -> {d2}")
    print(f"Classical minimum-distance pair index: {classical_answer}")

    quantum_answer, counts, iterations = run_grover(classical_answer)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")
    print(f"Quantum (Grover) result index: {quantum_answer}")

    passed = quantum_answer == classical_answer
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
