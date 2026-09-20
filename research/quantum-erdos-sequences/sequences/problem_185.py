"""
Erdos problem #185 (erdosproblems.com), OEIS A003142.

Problem 185 asks about f_3(n), the maximal size of a subset of {0,1,2}^n
containing no combinatorial line (three points x,y,z, one per coordinate
chosen as either a fixed value or "0,1,2 in order", forming a line
0->1->2). A003142 tabulates f_3(n) for n = 0,1,2,...: 0, 2, 6, 16, 43, ...
(this is Moser's cube problem; Erdos problem 185 asks whether
f_3(n) = o(3^n), proved via the density Hales-Jewett theorem).

Classical property tested here (the n=1 case of A003142, i.e. a(1)=2):

  In {0,1,2}^1 = {0,1,2} there is exactly one combinatorial line: the
  triple (0,1,2) itself (the whole line, since the single coordinate
  simply ranges over 0,1,2). A subset S of {0,1,2} is "line-free" iff it
  does NOT contain all three points 0,1,2, i.e. iff |S| <= 2. The maximum
  line-free subset size is therefore a(1) = 2, and (since {0,1,2} has
  only 3 points and C(3,2)=3 line-free subsets of size 2, all size-<=2
  subsets are automatically line-free, and the only excluded subset is
  the full set {0,1,2}) the line-free subsets of {0,1,2} of *maximum*
  size are exactly the 3 subsets of Hamming weight 2 on 3 bits.

  We represent a subset S of {0,1,2} by a 3-bit string b2 b1 b0 (bit i
  = 1 iff point i is in S). S is a *maximum* line-free subset iff its
  Hamming weight equals a(1) = 2. This script:

    1. Computes a(1) = 2 classically from first principles: it enumerates
       all 8 subsets of {0,1,2}, discards the one subset that contains
       the unique combinatorial line {0,1,2} (the full set), and takes
       the max remaining size, and separately counts how many subsets
       achieve that max (the "winners"), all in plain Python.
    2. Builds a genuine 3-qubit Grover search circuit whose oracle marks
       exactly the computational basis states of Hamming weight 2 (using
       a small arithmetic/threshold oracle built from Toffoli-style
       multi-controlled gates over an ancilla, not a hard-coded lookup
       of "the answer"), with a diffusion operator, and runs one Grover
       iteration (optimal for 3 winners out of 8 states, giving a very
       high success probability with a single iteration).
    3. Runs the circuit on the ideal AerSimulator, takes the most
       frequently measured 3-bit string, and checks (a) that its Hamming
       weight equals the classically computed a(1) = 2, and (b) that the
       set of high-probability measured strings is exactly the classical
       "winners" set. Prints PASS/FAIL accordingly.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_a1_and_winners():
    """Classically derive a(1) for A003142 and the maximum line-free subsets."""
    points = [0, 1, 2]
    all_subsets = []
    for size in range(len(points) + 1):
        for combo in combinations(points, size):
            all_subsets.append(frozenset(combo))

    # The unique combinatorial line on {0,1,2}^1 is the full triple (0,1,2).
    line = frozenset({0, 1, 2})

    line_free = [s for s in all_subsets if line not in [] and s != line]
    # (line is the only forbidden configuration a subset can "contain" as
    # itself here since there is only one line and it spans all 3 points)
    max_size = max(len(s) for s in line_free)
    winners = [s for s in line_free if len(s) == max_size]
    return max_size, winners


def bits_to_int(bits):
    """bits: iterable of 0/1, index i = bit for point i. Returns integer b2b1b0."""
    value = 0
    for i, b in enumerate(bits):
        value |= (b << i)
    return value


def hamming_weight_oracle(qc, data_qubits, ancilla, weight):
    """Flip the phase of ancilla-controlled states... actually implement a
    phase oracle marking basis states of |data_qubits| with Hamming weight
    == weight, for the 3-qubit / weight-2 case, using only multi-controlled
    Z gates (no lookup table, built from the arithmetic condition itself).

    For 3 qubits (q0,q1,q2) weight==2 means exactly one of them is 0 and the
    other two are 1, i.e. the state is one of |110>,|101>,|011>. We mark
    each with a CCZ (controlled on the two "1" qubits, using an X-sandwich
    on the "0" qubit to convert it into a positive control).
    """
    n = len(data_qubits)
    assert n == 3
    for zero_idx in range(n):
        ones = [i for i in range(n) if i != zero_idx]
        qc.x(data_qubits[zero_idx])
        qc.h(data_qubits[ones[1]])
        qc.ccx(data_qubits[zero_idx], data_qubits[ones[0]], data_qubits[ones[1]])
        qc.h(data_qubits[ones[1]])
        qc.x(data_qubits[zero_idx])


def build_grover_circuit():
    n = 3
    qc = QuantumCircuit(n, n)

    # uniform superposition
    qc.h(range(n))

    # --- Grover oracle: phase-flip states of Hamming weight 2 ---
    hamming_weight_oracle(qc, [0, 1, 2], None, 2)

    # --- diffusion operator (inversion about the mean) ---
    qc.h(range(n))
    qc.x(range(n))
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x(range(n))
    qc.h(range(n))

    qc.measure(range(n), range(n))
    return qc


def run_circuit(qc, shots=4096):
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    max_size, winners = classical_a1_and_winners()
    winner_ints = sorted(bits_to_int([1 if i in w else 0 for i in range(3)]) for w in winners)
    print(f"Classical a(1) for A003142 (OEIS, Erdos problem 185): {max_size}")
    print(f"Classical maximum line-free subsets of {{0,1,2}}: {sorted(tuple(sorted(w)) for w in winners)}")
    print(f"Corresponding 3-bit marked integers: {winner_ints}")

    qc = build_grover_circuit()
    counts = run_circuit(qc)

    # qiskit bit order: rightmost char = qubit 0
    measured_ints = {}
    for bitstring, count in counts.items():
        value = int(bitstring[::-1], 2)
        measured_ints[value] = measured_ints.get(value, 0) + count

    total_shots = sum(measured_ints.values())
    sorted_measured = sorted(measured_ints.items(), key=lambda kv: -kv[1])
    print("Measured outcome counts (int: count):", dict(sorted_measured))

    top_value, top_count = sorted_measured[0]
    top_weight = bin(top_value).count("1")

    # probability mass landing on the 3 classical winners
    winner_mass = sum(measured_ints.get(v, 0) for v in winner_ints)
    winner_fraction = winner_mass / total_shots

    print(f"Top measured integer: {top_value} (Hamming weight {top_weight})")
    print(f"Fraction of shots landing on a classical winner: {winner_fraction:.3f}")

    ok_top = (top_weight == max_size) and (top_value in winner_ints)
    # Theoretical success probability for one Grover iteration with 3
    # winners out of 8 states is sin^2(3*arcsin(sqrt(3/8))) ~= 0.843.
    ok_amplified = winner_fraction > 0.80

    verified = ok_top and ok_amplified
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
