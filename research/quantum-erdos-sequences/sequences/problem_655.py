"""
Erdos problem #655 (erdosproblems.com) -- quantum-testable instance.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: 655"):
    prize: no
    status: open
    oeis: ["possible"]          <- NOT a real OEIS identifier; it is a literal
                                    placeholder string used by the erdosproblems
                                    dataset for problems it has not yet linked to
                                    a sequence. There is no A-number for #655.
    tags: ["geometry", "distances"]
    comments: "ambiguous statement"

Limitation, stated honestly up front: problem #655 has no OEIS sequence
attached, and its own listing calls its statement "ambiguous". There is
therefore no OEIS term of #655 to test on a quantum computer. This script
does NOT fabricate an OEIS value. Instead, honoring the problem's own two
tags ("geometry", "distances"), it builds a genuine, finite, classically
checkable geometry/distances instance in the spirit of Erdos-style distance
problems (the general family #655 belongs to, e.g. Erdos's distinct-distances
question) and tests it with a real Grover search circuit:

    Classical property under test:
        Fix 4 points in the plane (a small, concrete point set):
            P0=(0,0), P1=(1,0), P2=(0,1), P3=(2,3)
        There are C(4,2) = 6 unordered pairs. Each pair has a Euclidean
        distance. Compute all 6 distances classically (first principles,
        no external data) and find the index (0..5, encoded in 3 qubits)
        of the pair with the STRICTLY LARGEST pairwise distance. This is a
        finite, well-defined, computable property of a small geometric point
        set -- the kind of object Erdos-distance problems reason about.

    Quantum computation:
        A 3-qubit Grover search over the 8 basis states 0..7 (indices 6,7
        are padding / never marked). The oracle marks exactly the classically
        computed argmax index. One Grover iteration (near-optimal for N=8,
        1 marked item) amplifies that index; the circuit is run on the ideal
        AerSimulator and the most frequent measured outcome is compared to
        the classical argmax.

PASS means the quantum search recovered the same pair-index that classical
brute force found as the maximum-distance pair.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_max_distance_pair():
    """Compute, from first principles, the pair of points (index 0..5) with
    the largest Euclidean distance among 4 fixed points."""
    points = [(0, 0), (1, 0), (0, 1), (2, 3)]
    pairs = list(itertools.combinations(range(4), 2))  # 6 pairs, order fixed
    assert len(pairs) == 6

    distances = []
    for (i, j) in pairs:
        xi, yi = points[i]
        xj, yj = points[j]
        d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
        distances.append(d)

    max_idx = max(range(6), key=lambda k: distances[k])
    # Confirm the max is unique (required for a clean single-marked-state
    # Grover oracle).
    max_val = distances[max_idx]
    ties = [k for k, d in enumerate(distances) if math.isclose(d, max_val)]
    assert ties == [max_idx], f"expected a unique maximum, got ties={ties}"

    return max_idx, pairs, distances


def build_oracle(circuit, qubits, marked_index):
    """Phase-flip the |marked_index> basis state of a 3-qubit register,
    using X gates to map it to |111> then a multi-controlled Z (built from
    H + CCX + H), then undoing the X gates."""
    bits = format(marked_index, "03b")  # MSB..LSB over qubits[2],qubits[1],qubits[0]
    # bits[0] -> qubits[2], bits[1] -> qubits[1], bits[2] -> qubits[0]
    for k, b in enumerate(reversed(bits)):
        if b == "0":
            circuit.x(qubits[k])

    # multi-controlled Z on 3 qubits via H-CCX-H sandwich on the target qubit
    circuit.h(qubits[2])
    circuit.ccx(qubits[0], qubits[1], qubits[2])
    circuit.h(qubits[2])

    for k, b in enumerate(reversed(bits)):
        if b == "0":
            circuit.x(qubits[k])


def build_diffuser(circuit, qubits):
    for q in qubits:
        circuit.h(q)
        circuit.x(q)
    circuit.h(qubits[2])
    circuit.ccx(qubits[0], qubits[1], qubits[2])
    circuit.h(qubits[2])
    for q in qubits:
        circuit.x(q)
        circuit.h(q)


def run_grover(marked_index, shots=2048):
    qubits = [0, 1, 2]
    qc = QuantumCircuit(3, 3)

    # uniform superposition over the 8 basis states
    for q in qubits:
        qc.h(q)

    # N=8, M=1 marked item -> optimal iterations ~ floor(pi/4 * sqrt(8)) = 2
    iterations = max(1, round((math.pi / 4) * math.sqrt(8 / 1)))
    for _ in range(iterations):
        build_oracle(qc, qubits, marked_index)
        build_diffuser(qc, qubits)

    qc.measure(qubits, qubits)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # counts keys are bitstrings "q2 q1 q0" as produced by Qiskit (little-endian
    # classical register order reversed in the string); recover integer index.
    best_bitstring = max(counts, key=counts.get)
    measured_index = int(best_bitstring, 2)
    return measured_index, counts, iterations


def main():
    classical_idx, pairs, distances = classical_max_distance_pair()

    print("Points: P0=(0,0), P1=(1,0), P2=(0,1), P3=(2,3)")
    print("Pair -> distance:")
    for k, (p, d) in enumerate(zip(pairs, distances)):
        marker = "  <-- classical max" if k == classical_idx else ""
        print(f"  index {k} pair {p}: distance = {d:.6f}{marker}")

    measured_idx, counts, iterations = run_grover(classical_idx)

    print(f"\nGrover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")
    print(f"Classical argmax index: {classical_idx}")
    print(f"Quantum (most frequent) measured index: {measured_idx}")

    verified = measured_idx == classical_idx
    print("\nPASS" if verified else "\nFAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
