"""
Erdos problem #18 ("practical numbers", OEIS A005153).

A positive integer n is *practical* if every integer m with 1 <= m <= n can
be written as a sum of distinct divisors of n. A005153 lists the practical
numbers: 1, 2, 4, 6, 8, 12, 16, 18, 20, 24, 28, 30, 36, ...

n = 8 is a known term of A005153 (verified classically below): its proper
divisors are {1, 2, 4}, and every m in 1..7 is a sum of a distinct subset of
{1, 2, 4} (this is just binary representation, since {1,2,4} are the powers
of two below 8).

The finite, computable property tested here:

    Given n = 8 and its proper divisor set D = {1, 2, 4}, find the unique
    subset S ⊆ D with sum(S) == target, for target = 5.

Classically (computed in this script from first principles, by brute-force
enumeration of all 2^|D| subsets) the unique subset of {1, 2, 4} summing to
5 is {1, 4}. This existence, for every target in 1..7, is exactly the
divisor-sum-coverage condition that makes 8 a practical number.

The quantum part is a genuine Grover search: 3 qubits, one per divisor in D
(qubit i set means "divisor D[i] is included in the subset"). A phase oracle
marks the unique computational basis state |101> (little-endian: bit0=1
included, bit1=2 excluded, bit2=4 included) whose divisor-subset sums to the
target, since 1 + 4 = 5 and this is the only subset of {1,2,4} with that
sum. One Grover iteration (optimal for a search space of size 8 with exactly
one marked item, iterations ~ floor(pi/4 * sqrt(8/1)) = 2, we use 2) is run
on the ideal AerSimulator, and the most frequently measured bitstring is
compared against the classically brute-forced answer.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def is_practical(n: int) -> bool:
    """Classical check: is n a practical number (OEIS A005153 membership)?"""
    if n < 1:
        return False
    if n == 1:
        return True
    divisors = [d for d in range(1, n) if n % d == 0]
    achievable = {0}
    for d in divisors:
        achievable |= {s + d for s in achievable}
    # m == n is trivially achievable via the divisor n itself, so only
    # 1 .. n-1 need to be covered by the proper divisors.
    return all(m in achievable for m in range(1, n))


def brute_force_subset_for_target(divisors, target):
    """Classical brute force: the unique subset of `divisors` summing to
    `target`, returned as a tuple of 0/1 inclusion bits (index 0 first)."""
    solutions = []
    for bits in itertools.product([0, 1], repeat=len(divisors)):
        if sum(d for d, b in zip(divisors, bits) if b) == target:
            solutions.append(bits)
    if len(solutions) != 1:
        raise ValueError(
            f"expected a unique subset summing to {target}, found {solutions}"
        )
    return solutions[0]


def build_grover_circuit(divisors, marked_bits, iterations):
    """Build a Grover search circuit over len(divisors) qubits that marks
    the single basis state `marked_bits` (tuple of 0/1, index0 = qubit0)."""
    n = len(divisors)
    qc = QuantumCircuit(n, n)

    # Uniform superposition.
    qc.h(range(n))

    def apply_oracle():
        # Phase-flip exactly the |marked_bits> state.
        # X on qubits that should be 0 in the marked state, so the marked
        # state becomes |11...1>, apply a multi-controlled Z, then undo X.
        zero_qubits = [i for i, b in enumerate(marked_bits) if b == 0]
        for q in zero_qubits:
            qc.x(q)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for q in zero_qubits:
            qc.x(q)

    def apply_diffuser():
        qc.h(range(n))
        qc.x(range(n))
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))

    for _ in range(iterations):
        apply_oracle()
        apply_diffuser()

    qc.measure(range(n), range(n))
    return qc


def main():
    n = 8
    classical_practical = is_practical(n)
    print(f"Classical: is {n} practical (OEIS A005153 membership)? "
          f"{classical_practical}")
    assert classical_practical, "8 must be a practical number for this test"

    divisors = [d for d in range(1, n) if n % d == 0]  # [1, 2, 4]
    target = 5
    classical_bits = brute_force_subset_for_target(divisors, target)
    classical_subset = [d for d, b in zip(divisors, classical_bits) if b]
    print(f"Divisors of {n} (excluding {n} itself): {divisors}")
    print(f"Classical brute force: unique subset of {divisors} summing to "
          f"{target} is {classical_subset} (bits={classical_bits})")

    num_items = 2 ** len(divisors)
    iterations = max(1, round((math.pi / 4) * math.sqrt(num_items)))
    print(f"Search space size = {num_items}, Grover iterations = {iterations}")

    qc = build_grover_circuit(divisors, classical_bits, iterations)

    sim = AerSimulator()
    shots = 2048
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical register string is big-endian in the printed key
    # (qubit n-1 ... qubit 0); convert to our little-endian bit tuple.
    best_key = max(counts, key=counts.get)
    measured_bits = tuple(int(c) for c in reversed(best_key))
    measured_prob = counts[best_key] / shots

    print(f"Quantum: most frequent measured bitstring (little-endian) = "
          f"{measured_bits}, probability = {measured_prob:.3f} "
          f"(counts={counts})")

    verified = measured_bits == classical_bits and measured_prob > 0.5
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
