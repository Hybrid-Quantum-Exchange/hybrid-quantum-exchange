"""
Erdos problem #325 -- quantum-testable instance.

Erdos problem #325 (erdosproblems.com), tags ["number theory", "powers"],
lists OEIS sequence ids A004825, A004832, A004843, A004854, A004865 (plus
an informal "possible" marker in the source data). This script uses
OEIS A004825: "Numbers that are the sum of at most 3 positive cubes" --
i.e. n = x^3 + y^3 + z^3 for some integers x, y, z with 0 <= x, y, z <= 3
(a zero digit means that cube is simply omitted from the sum, so "at most
3" terms are used). The sequence's first terms, taken directly from OEIS
A004825, are:

    0, 1, 2, 3, 8, 9, 10, 16, 17, 24, 27, 28, 29, 35, 36, ...

CLASSICAL PROPERTY TESTED
--------------------------
Fix N = 29, a genuine term of A004825 (27 + 1 + 1 = 29). The finite,
computable property under test is:

    "There exist x, y, z in {0, 1, 2, 3} with x^3 + y^3 + z^3 == 29."

The script first computes, from first principles (brute force over the
4 x 4 x 4 = 64 possible (x, y, z) triples), the exact set of triples that
satisfy this equation. This is the classical ground truth.

QUANTUM CIRCUIT
----------------
Each of x, y, z is represented by a 2-qubit register (values 0..3), for a
6-qubit search space of size 64. A Grover search is run over these 6
qubits. The oracle is built by marking (with multi-controlled Z gates,
using X gates to remap "0" bits to controls) exactly the computational
basis states corresponding to the (x, y, z) triples found classically to
satisfy x^3 + y^3 + z^3 == 29 -- i.e. the oracle encodes the classically
verified solution set, not an arbitrary guess. The number of Grover
iterations is chosen from the standard formula using the true number of
marked states out of 64. The circuit is run on the ideal AerSimulator
and the most frequent measured bitstring(s) are decoded back to (x, y, z)
and checked against the classical solution set.

PASS/FAIL
---------
The script prints PASS if the quantum circuit's most-sampled outcome(s)
decode to a valid classical solution of x^3 + y^3 + z^3 == 29 with
x, y, z in {0, 1, 2, 3}, and the measured solutions found are exactly a
subset of the classically-computed solution set (no false positives).
Otherwise it prints FAIL.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

N_TARGET = 29
DOMAIN = [0, 1, 2, 3]  # each register holds a 2-bit value 0..3


def classical_solutions(target):
    """Brute-force, from first principles, all (x, y, z) in DOMAIN^3 with
    x^3 + y^3 + z^3 == target."""
    sols = []
    for x in DOMAIN:
        for y in DOMAIN:
            for z in DOMAIN:
                if x ** 3 + y ** 3 + z ** 3 == target:
                    sols.append((x, y, z))
    return sols


def triple_to_bits(x, y, z):
    """Encode (x, y, z), each in 0..3, as a 6-bit string q5 q4 q3 q2 q1 q0
    with qubits [x1 x0 y1 y0 z1 z0] (x most significant register)."""
    def two_bits(v):
        return format(v, "02b")

    return two_bits(x) + two_bits(y) + two_bits(z)


def build_oracle(marked_bitstrings, n_qubits):
    """Multi-controlled-Z oracle that flips the phase of exactly the
    computational basis states listed in marked_bitstrings (each a
    string of n_qubits '0'/'1' chars, MSB-first as qubit n_qubits-1 down
    to qubit 0)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bitstring in marked_bitstrings:
        # bitstring[0] corresponds to the most significant qubit index
        # (n_qubits - 1); map to actual qubit indices.
        zero_positions = [
            n_qubits - 1 - i for i, b in enumerate(bitstring) if b == "0"
        ]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked_bitstrings, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_bitstrings, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def bits_to_triple(bitstring_msb_first):
    x = int(bitstring_msb_first[0:2], 2)
    y = int(bitstring_msb_first[2:4], 2)
    z = int(bitstring_msb_first[4:6], 2)
    return (x, y, z)


def main():
    n_qubits = 6
    domain_size = len(DOMAIN) ** 3  # 64

    sols = classical_solutions(N_TARGET)
    print(f"Classical solutions for x^3+y^3+z^3 == {N_TARGET}, "
          f"x,y,z in {DOMAIN}: {sols}")
    assert len(sols) > 0, "N_TARGET must be a genuine A004825 term"

    marked_bitstrings = [triple_to_bits(*s) for s in sols]
    m = len(marked_bitstrings)

    # Standard optimal Grover iteration count for m marked out of domain_size.
    theta = math.asin(math.sqrt(m / domain_size))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = build_grover_circuit(marked_bitstrings, n_qubits, iterations)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit returns bit strings little-endian in classical-register order
    # (c0 c1 ... on the right..left is c[n-1]..c[0]); since we measured
    # qubit i into clbit i in order, Qiskit's returned string has clbit
    # n-1 as the leftmost character already, i.e. MSB-first matching our
    # qubit-index convention (qubit n_qubits-1 first). No reversal needed
    # because we built bitstrings the same way we mapped qubits above.
    sorted_counts = Counter(counts).most_common()
    total_marked_shots = sum(c for bstr, c in counts.items()
                              if bstr in marked_bitstrings)
    marked_fraction = total_marked_shots / shots

    top_bitstrings = [bstr for bstr, _ in sorted_counts[:m]]
    measured_triples = [bits_to_triple(b) for b in top_bitstrings]

    print(f"Marked classical solutions (as bitstrings): {marked_bitstrings}")
    print(f"Grover iterations used: {iterations}")
    print(f"Top {m} measured bitstring(s): {top_bitstrings} "
          f"-> triples {measured_triples}")
    print(f"Fraction of shots landing on a marked solution: "
          f"{marked_fraction:.3f}")

    all_top_are_solutions = all(t in sols for t in measured_triples)
    amplified_enough = marked_fraction > (m / domain_size) * 2

    verified = all_top_are_solutions and amplified_enough

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
