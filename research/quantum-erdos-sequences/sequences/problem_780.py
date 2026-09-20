"""
Erdos problem #780 -- quantum-testable instance
=================================================

Source metadata (data/problems.yaml, erdosproblems clone):
    number: "780"
    prize: no
    status: proved (2025-08-31)
    oeis: ["N/A"]
    tags: ["combinatorics", "hypergraphs", "chromatic number"]

LIMITATION, stated up front: this problem carries no OEIS sequence id
("N/A" in the source data), so there is no literal integer sequence to
test membership/terms against. In place of fabricating an OEIS value,
this script builds a genuine finite/computable instance of the exact
mathematical object the problem's own tags name -- hypergraph
2-colorability, a.k.a. "Property B" -- and verifies a real Qiskit
Grover search against a from-scratch classical brute-force check on
that instance. This is an honest best-effort substitute for a missing
OEIS-anchored sequence, not a claim that OEIS data was used.

The instance
------------
Take the complete 3-uniform hypergraph on 4 vertices {0,1,2,3}: its
edges are all four 3-subsets of {0,1,2,3}:

    {0,1,2}, {0,1,3}, {0,2,3}, {1,2,3}

A 2-coloring of the vertices (colors in {0,1}) has "Property B" for
this hypergraph if no edge is monochromatic (not all three of its
vertices get the same color). Because every 3-subset of a 4-set is an
edge here, a coloring avoids all monochromatic edges exactly when no
color class has size >= 3, i.e. exactly when the color split is 2-2
(one could also ask about 0-4 or 1-3 splits, but any split with a
color class of size 3 or 4 necessarily contains a monochromatic edge,
since every 3 vertices form an edge).

So: colorings satisfying Property B <=> bitstrings of length 4 with
Hamming weight exactly 2 (two 0s and two 1s). There are C(4,2) = 6 of
them out of 2^4 = 16 total colorings. This is computed classically
in `classical_valid_colorings()` below, by brute force over all 16
assignments and an explicit check of all 4 edges -- not asserted.

The quantum circuit
--------------------
A genuine Grover search over the 4-qubit computational basis, whose
oracle phase-flips exactly the computational basis states matching a
Property-B-satisfying coloring (the 6 bitstrings found classically,
used only as the oracle's phase targets -- the oracle is built from
the edge structure indirectly via that classical enumeration, and the
search's success is judged independently against the same classical
enumeration). With N=16 states and M=6 marked, one Grover iteration
(round(pi/4 * sqrt(N/M)) = 1) should sharply concentrate measurement
outcomes on the 6 marked ("valid coloring") strings.

PASS/FAIL: after running the circuit on AerSimulator and taking 4096
shots, the script checks that at least 80% of measured outcomes are
among the classically-verified set of 6 valid-coloring bitstrings --
well above the ~37.5% (6/16) a uniform superposition would give with
no amplification, and consistent with the ~84.4% theoretical success
probability of a single Grover iteration on this M=6, N=16 instance
(sin^2(3*asin(sqrt(6/16))) ~= 0.844).
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_VERTICES = 4
EDGES = list(itertools.combinations(range(N_VERTICES), 3))  # all four 3-subsets


def classical_valid_colorings():
    """Brute-force, from first principles: all 2-colorings of 4 vertices
    that leave every edge (every 3-subset) non-monochromatic.

    Returns a sorted list of bitstrings (str, length N_VERTICES), qubit
    0 = least significant bit, matching Qiskit's little-endian
    measurement string convention.
    """
    valid = []
    for bits in itertools.product([0, 1], repeat=N_VERTICES):
        ok = True
        for edge in EDGES:
            colors = {bits[v] for v in edge}
            if len(colors) == 1:  # monochromatic edge
                ok = False
                break
        if ok:
            # bits[0] is vertex 0; Qiskit prints qubit N-1 ... qubit 0,
            # so build the string with vertex 0 as the rightmost char.
            s = "".join(str(bits[v]) for v in reversed(range(N_VERTICES)))
            valid.append(s)
    return sorted(valid)


def mark_bitstring(qc, bitstring):
    """Phase-flip the single computational basis state matching
    `bitstring` (Qiskit little-endian convention: bitstring[-1] is
    qubit 0). Implemented as an n-controlled Z via the standard
    H - MCX - H sandwich, with X gates remapping 0-bits to 1-bits
    for the duration of the controls.
    """
    n = qc.num_qubits
    zero_qubits = [q for q in range(n) if bitstring[n - 1 - q] == "0"]

    for q in zero_qubits:
        qc.x(q)

    controls = list(range(n - 1))
    target = n - 1
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)

    for q in zero_qubits:
        qc.x(q)


def build_oracle(n, marked_strings):
    qc = QuantumCircuit(n, name="oracle")
    for s in marked_strings:
        mark_bitstring(qc, s)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(n, marked_strings, iterations):
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    oracle = build_oracle(n, marked_strings)
    diffuser = build_diffuser(n)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n), range(n))
    return qc


def main():
    valid_colorings = classical_valid_colorings()
    total_states = 2 ** N_VERTICES
    expected_count = math.comb(N_VERTICES, 2)  # 2-2 splits: C(4,2) = 6

    print("Erdos problem #780 -- hypergraph 2-colorability (Property B) instance")
    print(f"Vertices: {N_VERTICES}, edges (all 3-subsets): {EDGES}")
    print(f"Classically valid (non-monochromatic) colorings: {valid_colorings}")
    print(f"Count = {len(valid_colorings)} (expected C(4,2) = {expected_count})")

    assert len(valid_colorings) == expected_count, (
        "classical brute force disagrees with the C(4,2) combinatorial count"
    )

    n = N_VERTICES
    m = len(valid_colorings)
    iterations = max(1, round((math.pi / 4) * math.sqrt(total_states / m)))
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(n, valid_colorings, iterations)

    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    shots = 4096
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    hits = sum(c for bitstring, c in counts.items() if bitstring in valid_colorings)
    fraction = hits / shots

    print(f"Measured distribution (top 10): {sorted(counts.items(), key=lambda kv: -kv[1])[:10]}")
    print(f"Fraction of shots landing on a valid coloring: {fraction:.4f}")

    threshold = 0.80
    success = fraction >= threshold

    if success:
        print(f"PASS: quantum Grover search concentrated {fraction:.4f} "
              f"(>= {threshold}) of shots on the classically-verified "
              f"Property-B colorings.")
    else:
        print(f"FAIL: quantum Grover search only concentrated {fraction:.4f} "
              f"(< {threshold}) of shots on the classically-verified "
              f"Property-B colorings.")

    return success


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
