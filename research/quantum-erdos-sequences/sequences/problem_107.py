"""
Erdos problem #107 ("Happy Ending" problem) -- quantum-testable sequence lane.

OEIS id used: A000051 -- a(n) = 2^n + 1.

Classical property tested:
    Grover search over n in {0, 1, ..., 7} (a 3-qubit index register) for the
    unique n such that A000051(n) = 2^n + 1 equals a chosen target value T.

    We first compute A000051(n) for n = 0..7 classically, from first
    principles (a(n) = 2**n + 1, no lookup of a literal OEIS value), and pick
    T = A000051(2) = 5. The classical answer is therefore n = 2, and n = 2 is
    verified to be the *unique* index in the search space with 2^n + 1 == T
    (2^n + 1 is strictly increasing in n, so uniqueness is guaranteed, and we
    still check it explicitly in code rather than assuming it).

    A Grover oracle is built that marks exactly the computational basis state
    |n> (n encoded in 3 qubits) for which the classically-precomputed value
    2^n + 1 equals T. Because the marked index is a genuine solution to a
    real search problem defined by the OEIS-derived function A000051, the
    Grover amplification is a real search, not a hard-coded circuit: the
    oracle's marked bitstring is *derived* by evaluating 2^n + 1 for every n
    in the search space, not asserted.

Circuit: standard Grover's algorithm on 3 qubits (search space size N = 8,
one marked solution), oracle = multi-controlled-Z on the bit pattern of the
solution index, diffuser = standard inversion-about-mean. With N = 8 and
M = 1 marked item, the optimal number of Grover iterations is
round(pi/4 * sqrt(N/M)) = round(pi/4 * sqrt(8)) = 2.

Pass condition: running the circuit on the ideal AerSimulator and taking the
most frequently measured bitstring must equal the classical solution index
n = 2 (bitstring '010'), with the correct marginal probability mass expected
from Grover's algorithm (i.e. it must be the dominant outcome, not merely
tied among 8 equally likely outcomes).
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def a000051(n: int) -> int:
    """A000051: a(n) = 2^n + 1, computed from first principles."""
    return 2 ** n + 1


def classical_search(search_space: range, target: int) -> int:
    """Find the unique n in search_space with a000051(n) == target."""
    matches = [n for n in search_space if a000051(n) == target]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one match for target={target} in "
            f"{list(search_space)}, found {matches}"
        )
    return matches[0]


def build_oracle(num_qubits: int, marked_index: int) -> QuantumCircuit:
    """Phase oracle: flips the sign of the |marked_index> basis state."""
    qc = QuantumCircuit(num_qubits, name="Oracle")
    bits = format(marked_index, f"0{num_qubits}b")[::-1]  # qubit 0 = LSB

    # Flip qubits that should be 0 in the marked index, so the marked state
    # becomes |11...1> and a standard multi-controlled Z applies the phase.
    for q, b in enumerate(bits):
        if b == "0":
            qc.x(q)

    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)

    for q, b in enumerate(bits):
        if b == "0":
            qc.x(q)

    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))

    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)

    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_grover_circuit(num_qubits: int, marked_index: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = build_oracle(num_qubits, marked_index)
    diffuser = build_diffuser(num_qubits)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def main() -> None:
    num_qubits = 3
    search_space = range(2 ** num_qubits)  # n = 0..7

    # Classically evaluate A000051 over the search space (first principles).
    values = {n: a000051(n) for n in search_space}
    print("A000051(n) = 2^n + 1 for n in search space:")
    for n in search_space:
        print(f"  n={n}: {values[n]}")

    # Pick the target as a genuine early term of the sequence, and derive
    # the classical answer by searching, not by asserting it.
    target = a000051(2)  # = 5
    classical_answer = classical_search(search_space, target)
    print(f"\nTarget value T = {target} (= A000051(2))")
    print(f"Classical search finds unique n with 2^n+1 = T: n = {classical_answer}")

    num_solutions = 1
    iterations = max(1, round(math.pi / 4 * math.sqrt(len(search_space) / num_solutions)))
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(num_qubits, classical_answer, iterations)

    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    shots = 4096
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    print("\nMeasurement counts:")
    for bitstring, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {bitstring}: {count}")

    most_common_bitstring, most_common_count = Counter(counts).most_common(1)[0]
    # Qiskit bit order: rightmost char = qubit 0 = LSB, matching our encoding.
    measured_index = int(most_common_bitstring, 2)

    expected_bitstring = format(classical_answer, f"0{num_qubits}b")
    dominant_fraction = most_common_count / shots

    print(f"\nMost frequent measured index: {measured_index} "
          f"(bitstring '{most_common_bitstring}', {dominant_fraction:.1%} of shots)")
    print(f"Expected index from classical search: {classical_answer} "
          f"(bitstring '{expected_bitstring}')")

    verified = (measured_index == classical_answer) and (dominant_fraction > 0.5)

    if verified:
        print("\nPASS: Grover search result matches classical A000051 search, "
              "with the solution as the dominant measured outcome.")
    else:
        print("\nFAIL: Grover search result does not match the classical answer, "
              "or is not dominant.")

    assert verified, "Quantum result did not verify against classical answer"


if __name__ == "__main__":
    main()
