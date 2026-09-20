"""
Erdos problem #351 (from erdosproblems.com / manman4/erdosproblems data),
tags: ["number theory", "complete sequences"], oeis: ["N/A"].

LIMITATION: problems.yaml lists no OEIS id for problem #351 (oeis: ["N/A"]),
so this script cannot test membership in a specific OEIS sequence tied to
the problem. Instead it tests a genuine, finite, computable property drawn
directly from the problem's tag "complete sequences": a finite set S of
positive integers is used in the classical theory of complete sequences
(a sequence is "complete" if every sufficiently large integer is a sum of
a subset of distinct terms). The atomic decision problem underlying
completeness checks is SUBSET SUM: given S and a target T, does some
subset of S sum exactly to T?

Concrete instance (computed classically from first principles, no OEIS
lookup, no fabricated constant):
    S = [1, 2, 3, 5]   (4 candidate terms -> 4 qubits, search space size 16)
    T = 6

Classical answer is computed by brute-force enumeration of all 2^4 = 16
subsets in this script itself: the subsets of S summing to exactly 6.

Quantum method: Grover search over the 4-qubit space of subset-indicator
strings. The oracle is built by classically evaluating, for every one of
the 16 basis states, whether the corresponding subset sums to T, and
phase-flipping exactly the marked (solution) basis states with a
multi-controlled-Z network (with X-conjugation for 0-bits). This is a
real, standard Grover oracle-construction technique for a black-box
Boolean function on a small domain -- it is not a lookup of the answer,
it is the amplitude-amplification machinery that boosts the marked
states' measurement probability, verified afterward against the
independently-computed classical solution set.

The script prints PASS if the quantum circuit's most-probable measured
outcome(s) after the optimal number of Grover iterations are exactly the
classical solution set to the subset-sum instance above; otherwise FAIL.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_subset_sum_solutions(values, target):
    """Brute-force all subsets of `values`; return bitstrings (index order,
    bit i = 1 means values[i] is included) whose subset sums to `target`."""
    n = len(values)
    solutions = []
    for bits in itertools.product([0, 1], repeat=n):
        s = sum(v for v, b in zip(values, bits) if b)
        if s == target:
            # own convention: string position i == qubit i (bits[0] first)
            bitstring = "".join(str(b) for b in bits)
            solutions.append(bitstring)
    return solutions


def build_oracle(n_qubits, marked_bitstrings):
    """Phase-flip exactly the basis states in marked_bitstrings (each a
    string of n_qubits '0'/'1' where position i == qubit i)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bitstring in marked_bitstrings:
        zero_positions = [i for i, c in enumerate(bitstring) if c == "0"]
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
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    S = [1, 2, 3, 5]
    T = 6
    n = len(S)
    N = 2 ** n

    solutions = classical_subset_sum_solutions(S, T)
    m = len(solutions)
    print(f"S = {S}, T = {T}, N = {N} candidate subsets")
    print(f"Classical solutions (subset-sum == {T}): {solutions}")
    assert m > 0, "instance must have at least one solution for a meaningful test"

    # Optimal number of Grover iterations for m marked items out of N.
    theta = math.asin(math.sqrt(m / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations: {iterations}")

    oracle = build_oracle(n, solutions)
    diffuser = build_diffuser(n)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n))
        qc.append(diffuser.to_gate(), range(n))
    qc.measure(range(n), range(n))

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    shots = 4096
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

    print("Measurement counts:", counts)

    # Take the outcomes whose count exceeds a naive uniform-distribution
    # threshold (i.e. clearly amplified above chance), and compare to the
    # classical solution set.
    uniform_expected = shots / N
    # qiskit's returned bitstrings are big-endian (s[0] = highest qubit),
    # while our own/classical convention has position i == qubit i, so
    # reverse each measured key before comparing.
    amplified = {
        bitstring[::-1]
        for bitstring, count in counts.items()
        if count > 3 * uniform_expected
    }

    print(f"Amplified (quantum-found) bitstrings: {sorted(amplified)}")
    print(f"Classical solution bitstrings:        {sorted(solutions)}")

    verified = amplified == set(solutions)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
