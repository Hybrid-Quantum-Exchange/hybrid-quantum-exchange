"""
Erdos problem #935 -- quantum-testable instance.

Source: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: 935" (open, unformalized, tags: ["number theory", "powerful"],
oeis: ["A057521", "A389244", "possible"]).

Both listed OEIS sequences (A057521, A389244) are built from the notion of a
"powerful number": a positive integer n such that for every prime p dividing
n, p^2 also divides n (equivalently, every exponent in n's prime
factorization is >= 2). A001694 is the base sequence of powerful numbers;
A057521 and related sequences classify/count powerful numbers by extra
conditions. This script does not fabricate a specific literal OEIS term;
instead it tests the exact, checkable, finite classical property that
underlies both sequences:

    PROPERTY: for n in the finite range [0, 63], is n "powerful"?
    (n is powerful iff n == 1, or for every prime p | n, p^2 | n)

This is a small, finite, computable search-space problem, so it is cast as
a Grover search: build an oracle over 6 qubits (representing integers
0..63) that marks exactly the powerful numbers in that range, run Grover's
algorithm on the ideal AerSimulator, and check that the amplified
(most-frequently-measured) computational basis states are exactly the
powerful numbers -- i.e. that quantum search finds the same "known term"
set that direct classical enumeration finds.

The classical powerful-number test and the enumeration of powerful numbers
in [0, 63] are both computed from first principles in this script (trial
division), not copied from OEIS.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_QUBITS = 6
N = 2 ** N_QUBITS  # 64


def is_powerful(n: int) -> bool:
    """Classical, first-principles powerful-number test.

    n is powerful iff every prime in its factorization appears with
    exponent >= 2. By convention 1 is powerful (empty product); 0 is
    excluded (not a positive integer).
    """
    if n <= 0:
        return False
    if n == 1:
        return True
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            exponent = 0
            while m % p == 0:
                m //= p
                exponent += 1
            if exponent < 2:
                return False
        p += 1
    if m > 1:
        # m is a leftover prime factor with exponent exactly 1
        return False
    return True


def classical_powerful_numbers(limit: int):
    return [n for n in range(limit) if is_powerful(n)]


def build_oracle(marked_states, n_qubits):
    """Phase-flip oracle: applies a -1 phase to each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            mcz = MCXGate(n_qubits - 1)
            qc.h(n_qubits - 1)
            qc.append(mcz, list(range(n_qubits - 1)) + [n_qubits - 1])
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
        mcz = MCXGate(n_qubits - 1)
        qc.append(mcz, list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    marked = classical_powerful_numbers(N)
    print(f"Classical powerful numbers in [0, {N}): {marked}")
    m = len(marked)
    assert m > 0

    # Optimal number of Grover iterations for this marked-set size.
    theta = math.asin(math.sqrt(m / N))
    iterations = max(1, round(math.pi / (4 * theta)))
    print(f"Marked count = {m}, Grover iterations = {iterations}")

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))

    oracle = build_oracle(marked, N_QUBITS)
    diffuser = build_diffuser(N_QUBITS)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(N_QUBITS))
        qc.append(diffuser.to_gate(), range(N_QUBITS))

    qc.measure(range(N_QUBITS), range(N_QUBITS))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 20000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Little-endian bit order -> integer.
    int_counts = Counter()
    for bitstring, c in counts.items():
        # Qiskit's count keys are big-endian (leftmost char = highest
        # qubit index), which already matches "int(bitstring, 2) == n"
        # for a register measured with qc.measure(range(k), range(k)).
        n = int(bitstring, 2)
        int_counts[n] += c

    # Take the top-m most frequently measured integers as Grover's answer.
    top_states = [n for n, _ in int_counts.most_common(m)]
    quantum_answer = sorted(top_states)

    marked_prob = sum(int_counts[n] for n in marked) / shots
    print(f"Total measured probability mass on marked states: {marked_prob:.4f}")
    print(f"Quantum top-{m} states: {quantum_answer}")
    print(f"Classical marked states: {marked}")

    verified = (quantum_answer == marked) and (marked_prob > 0.8)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
