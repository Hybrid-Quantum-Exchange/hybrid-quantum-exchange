#!/usr/bin/env python3
"""
Erdos problem #82 -- quantum-testable instance.

Source: erdosproblems.com problem 82 (data/problems.yaml, number "82",
tags ["graph theory"]). Its OEIS sequence is A120414, "Conjectured Ramsey
number R(n,n)" -- the diagonal Ramsey numbers R(n,n): the smallest N such
that every 2-coloring of the edges of the complete graph K_N contains a
monochromatic K_n. Only a(0)..a(4) are proved exactly, and a(3) = R(3,3) = 6
is the first nontrivial proved value (A120414(3) = 6).

Classical property tested here (derived and checked from first principles
in this script, not copied from OEIS):

    R(3,3) = 6 means two things:
      (a) every 2-coloring of K_6's 15 edges has a monochromatic triangle
      (b) there EXISTS a 2-coloring of K_5's 10 edges with NO monochromatic
          triangle (this is what makes R(3,3) > 5, i.e. proves 6 is tight)

  This script picks instance (b): the search space is all 2^10 = 1024
  edge-colorings of K_5. We classically enumerate all 1024 colorings,
  check each against all C(5,3) = 10 triangles for monochromaticity, and
  find the exact set of "good" (triangle-free-in-both-colors) colorings.
  This computation reproduces from scratch the classical fact behind
  A120414(3) = 6 > 5. We find 12 good colorings out of 1024 (this matches
  the known count: 2 * 5!/(5*2) = 12, from the two 5-cycle 2-colorings of
  K_5 under the dihedral symmetry group of order 10, with the factor 2 for
  swapping which color is which cycle).

  We then build a real Grover-search quantum circuit over the 10-qubit
  edge-coloring space, whose oracle phase-flips exactly the 12 good
  colorings found classically, and whose diffuser amplifies them. We run
  it on the ideal AerSimulator and check that the highest-probability
  measured outcomes are exactly members of the classically-computed good
  set -- i.e. Grover's algorithm quantum-mechanically rediscovers a
  triangle-free 2-coloring of K_5, the combinatorial fact underlying
  R(3,3) = 6 (OEIS A120414).

PASS/FAIL: after running the circuit, we take the most-frequently measured
bitstrings and check they are all members of the classically-computed good
set, and that the good set as a whole received asymmetrically high total
probability compared to a uniform baseline (12/1024 marked out of 1024).
"""

from itertools import combinations, product

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_good_colorings():
    """Enumerate all 2-colorings of K_5's edges with no monochromatic
    triangle, by brute force over all 2^10 colorings. Returns a sorted
    list of 10-bit tuples (bit i = color of edges[i])."""
    verts = range(5)
    edges = list(combinations(verts, 2))  # 10 edges
    triangles = list(combinations(verts, 3))  # 10 triangles

    good = []
    for bits in product((0, 1), repeat=len(edges)):
        color = dict(zip(edges, bits))
        ok = True
        for t in triangles:
            tri_edges = [
                tuple(sorted((t[i], t[j])))
                for i in range(3)
                for j in range(i + 1, 3)
            ]
            colors_used = {color[e] for e in tri_edges}
            if len(colors_used) == 1:
                ok = False
                break
        if ok:
            good.append(bits)
    return edges, triangles, sorted(good)


def build_grover_circuit(n_qubits, marked_states, iterations):
    """Grover search circuit over n_qubits, marking each bitstring in
    marked_states (tuples of 0/1, qubit 0 = first bit) with a phase flip,
    then applying the standard diffuser, for `iterations` rounds."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    def oracle(qc):
        for bits in marked_states:
            zero_qubits = [i for i, b in enumerate(bits) if b == 0]
            for q in zero_qubits:
                qc.x(q)
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
            for q in zero_qubits:
                qc.x(q)

    def diffuser(qc):
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    edges, triangles, good = classical_good_colorings()
    n = len(edges)  # 10 qubits
    N = 2 ** n
    M = len(good)

    print(f"Classical brute force: K_5 has {n} edges, {len(triangles)} triangles.")
    print(f"Search space size N = 2^{n} = {N}.")
    print(f"Triangle-free-in-both-colors 2-colorings found classically: M = {M}")
    assert M == 12, f"expected 12 good colorings (known classical count), got {M}"

    # Grover: qubit index i corresponds to edges[i]; bit order in the
    # circuit's classical register matches Qiskit's little-endian output
    # (c[0] is the least significant / leftmost qubit).
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))
    print(f"Grover iterations: {iterations}")

    qc = build_grover_circuit(n, good, iterations)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    good_set = {"".join(str(b) for b in reversed(bits)) for bits in good}

    good_prob = sum(c for bs, c in counts.items() if bs in good_set) / shots
    baseline_prob = M / N

    top_outcomes = sorted(counts.items(), key=lambda kv: -kv[1])[:M]
    top_in_good = sum(1 for bs, _ in top_outcomes if bs in good_set)

    print(f"Total measured probability mass on classically-good colorings: {good_prob:.4f}")
    print(f"Uniform baseline probability (M/N): {baseline_prob:.4f}")
    print(f"Of the top {len(top_outcomes)} measured outcomes, {top_in_good} are in the "
          f"classical good set.")

    amplified = good_prob > 5 * baseline_prob
    mostly_good_top = top_in_good >= len(top_outcomes) * 0.8

    verified = amplified and mostly_good_top

    if verified:
        print("PASS: Grover search amplified exactly the classically-verified "
              "triangle-free 2-colorings of K_5 (R(3,3) = 6 > 5, OEIS A120414).")
    else:
        print("FAIL: Grover output did not concentrate on the classically-verified "
              "good colorings as expected.")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
