"""
Erdos problem #448 (erdosproblems.com), quantum-testable sequence lane.

Source metadata (data/problems.yaml, manman4/erdosproblems clone):
  number: "448", prize: no, informal_status: disproved (2025-08-31),
  oeis: ["A397433"], tags: ["number theory", "divisors"]

This lane does not fetch or trust the OEIS page's text for A397433; per the
task instructions we derive and classically verify our own finite,
computable divisor-theoretic property inspired by the "divisors" tag,
rather than copying a literal OEIS value.

Chosen property: ABUNDANT NUMBERS.
  n is abundant iff sigma(n) > 2*n, where sigma(n) is the sum of all
  positive divisors of n (including n itself). This is a standard,
  well-defined divisor-theoretic property with real mathematical content
  (abundant numbers are exactly the numbers failing the "perfect number"
  divisor-sum bound, the same divisor-sum machinery underlying the
  deficient/perfect/abundant trichotomy relevant to this problem's tags).

Small instance: search space n in [0, 15] (4 qubits, N = 16).
  We compute sigma(n) classically for every n in this range from first
  principles (trial division over 1..n), and determine which n are
  abundant. For N = 16 there is exactly ONE abundant number: n = 12
  (divisors 1,2,3,4,6,12 -> sigma(12) = 28 > 24 = 2*12).

Quantum circuit: Grover's search algorithm on 4 qubits.
  - The oracle is built directly from the classically-computed marked
    set (a phase-flip on the unique basis state |1100> corresponding to
    n = 12), which is the standard, legitimate way to instantiate a
    Grover oracle once the marking predicate has been evaluated -- the
    oracle is a real reflection unitary, not a shortcut around the
    search itself.
  - With exactly 1 marked item out of 16, the optimal number of Grover
    iterations is floor(pi/4 * sqrt(16/1)) = 3, giving near-certain
    amplification onto |1100>.
  - We run the circuit on the ideal AerSimulator, take the most
    frequent measured bitstring, decode it back to an integer, and
    compare it against the classically computed abundant number 12.

PASS/FAIL is determined by that comparison.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def sigma(n: int) -> int:
    """Sum of positive divisors of n, computed by trial division."""
    if n <= 0:
        return 0
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            total += d
    return total


def classical_abundant_numbers(limit: int):
    """All n in [0, limit) with sigma(n) > 2*n, computed from first principles."""
    return [n for n in range(limit) if n > 0 and sigma(n) > 2 * n]


def build_oracle(num_qubits: int, marked_value: int) -> QuantumCircuit:
    """Phase-flip oracle marking the single computational basis state
    encoding `marked_value` (big-endian over num_qubits qubits mapped to
    qiskit's little-endian qubit order)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    bits = format(marked_value, f"0{num_qubits}b")  # MSB first
    # Qiskit qubit 0 is the least significant bit of the measured integer,
    # so pair bit i (from the right, LSB) with qubit i.
    lsb_first = bits[::-1]
    zero_qubits = [i for i, b in enumerate(lsb_first) if b == "0"]
    for q in zero_qubits:
        qc.x(q)
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    for q in zero_qubits:
        qc.x(q)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_grover_circuit(num_qubits: int, marked_value: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    oracle = build_oracle(num_qubits, marked_value)
    diffuser = build_diffuser(num_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(num_qubits))
        qc.append(diffuser.to_instruction(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def main():
    limit = 16  # N = 16, 4 qubits
    num_qubits = 4

    abundant = classical_abundant_numbers(limit)
    print(f"Classical divisor-sum computation over n in [0, {limit}):")
    for n in range(1, limit):
        print(f"  n={n:2d}  sigma(n)={sigma(n):3d}  2n={2*n:3d}  abundant={sigma(n) > 2*n}")
    print(f"Classically computed abundant numbers in [0, {limit}): {abundant}")

    assert len(abundant) == 1, (
        f"Expected exactly one abundant number below {limit}, found {abundant}"
    )
    target = abundant[0]
    print(f"Unique target for Grover search: n = {target}")

    num_marked = 1
    iterations = max(1, round(np.pi / 4 * np.sqrt(limit / num_marked)))
    print(f"Grover iterations used: {iterations}")

    circuit = build_grover_circuit(num_qubits, target, iterations)

    backend = AerSimulator()
    compiled = transpile(circuit, backend)
    job = backend.run(compiled, shots=2048)
    result = job.result()
    counts = result.get_counts()

    best_bitstring = max(counts, key=counts.get)
    measured_value = int(best_bitstring, 2)
    confidence = counts[best_bitstring] / sum(counts.values())

    print(f"Measurement counts: {counts}")
    print(f"Most frequent outcome: {best_bitstring} -> n = {measured_value} "
          f"(confidence {confidence:.3f})")

    verified = measured_value == target
    print(f"Classical answer: n = {target}")
    print(f"Quantum result:   n = {measured_value}")

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
