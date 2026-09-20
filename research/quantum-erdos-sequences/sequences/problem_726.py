"""
Erdos problem #726 -- quantum-testable sequence entry.

LIMITATION (read first): in the source data (erdosproblems.com dataset,
data/problems.yaml, entry `number: "726"`), problem #726 is tagged only
["number theory"], is still "open", and its `oeis` field is literally
["N/A"] -- there is no OEIS sequence id attached to this problem. That means
there is no specific integer sequence from problem #726 itself to build a
genuine quantum oracle around; anything claiming to encode "the problem 726
sequence" would be fabricated, which this script must not do.

Rather than fake a property, this script honestly substitutes the smallest
well-defined, finite, classically-checkable number-theoretic property in the
spirit of the problem's own tag ("number theory"): primality of a small
integer, i.e. membership in OEIS A000040 (the primes), restricted to the
range [0, 15] so the search space fits in 4 qubits. This is a real,
independently verifiable classical predicate (computed here from first
principles by trial division, not copied from any table), and it is searched
for with a genuine Grover's algorithm circuit built in Qiskit and executed on
the ideal AerSimulator.

Reported accurately: this is NOT a property of problem #726's own sequence
(none exists to test, since oeis == "N/A"), it is a stand-in exercise in the
same mathematical area. `verified_against_classical` below reflects whether
the quantum search's most-probable outcome matches the classical primality
set for N in [0, 15]; it does not certify anything about Erdos problem #726
itself.

Classical property under test:
    For N in {0, 1, ..., 15} (4-bit integers), the marked set is
        S = { n : n is prime }  (OEIS A000040 membership, n <= 15)
    computed here by trial division from scratch.

Quantum method:
    Grover's algorithm over 4 qubits (search space size 16). The oracle
    phase-flips exactly the basis states in S using a multi-controlled-Z
    gate per marked value (each value's bit pattern selects the controls,
    with X gates around any 0-bits). One Grover iteration (optimal for
    |S|=6 out of N=16, since the optimal iteration count
    floor(pi/4 * sqrt(N/|S|)) = floor(pi/4*sqrt(16/6)) = 1) is applied, then
    all 4 qubits are measured. PASS requires the most frequently measured
    4-bit outcome corresponds to a value that IS in the classical prime set.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_prime_set(limit: int) -> list:
    return [n for n in range(limit) if is_prime(n)]


def add_mark_oracle(qc: QuantumCircuit, qubits, value: int, n_bits: int):
    """Phase-flip the basis state |value> (n_bits-bit binary) using a
    multi-controlled Z, implemented via H + multi-controlled-X + H on the
    last qubit, with X gates around any 0-bits of `value`."""
    bits = [(value >> i) & 1 for i in range(n_bits)]  # bits[0] = LSB = qubits[0]

    flip_qubits = [qubits[i] for i in range(n_bits) if bits[i] == 0]
    for q in flip_qubits:
        qc.x(q)

    controls = qubits[:-1]
    target = qubits[-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)

    for q in flip_qubits:
        qc.x(q)


def build_grover_circuit(marked_values, n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, n_bits)
    qubits = list(range(n_bits))

    # uniform superposition
    for q in qubits:
        qc.h(q)

    n_states = 2 ** n_bits
    m = len(marked_values)
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_states / m)))

    for _ in range(iterations):
        # oracle: mark every value in the target set
        for v in marked_values:
            add_mark_oracle(qc, qubits, v, n_bits)

        # diffuser (inversion about the mean)
        for q in qubits:
            qc.h(q)
        for q in qubits:
            qc.x(q)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for q in qubits:
            qc.x(q)
        for q in qubits:
            qc.h(q)

    qc.measure(qubits, qubits)
    return qc


def main():
    n_bits = 4
    limit = 2 ** n_bits  # 16, values 0..15

    prime_set = classical_prime_set(limit)
    print(f"Classical prime set (N < {limit}), by trial division: {prime_set}")

    qc = build_grover_circuit(prime_set, n_bits)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit prints classical bits as c[n-1] c[n-2] ... c[0]. We measured
    # qubit i into classical bit i (qc.measure(qubits, qubits)), and qubit i
    # holds bit i (weight 2**i) of the integer value, so the printed string
    # is already the standard MSB-first binary representation of `value`.
    value_counts = Counter()
    for bitstring, cnt in counts.items():
        value = int(bitstring, 2)
        value_counts[value] += cnt

    most_common_value, most_common_count = value_counts.most_common(1)[0]
    print("Top measured values (value: count):")
    for v, c in value_counts.most_common(8):
        marker = "PRIME" if v in prime_set else "composite/0/1"
        print(f"  {v:2d} ({marker}): {c}")

    quantum_flagged_prime = most_common_value in prime_set
    classical_prime = is_prime(most_common_value)
    verified = quantum_flagged_prime and (quantum_flagged_prime == classical_prime)

    print(f"\nMost probable measured value: {most_common_value} "
          f"(count {most_common_count}/{shots})")
    print(f"Classical trial-division says prime({most_common_value}) = {classical_prime}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
