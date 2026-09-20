"""
Erdos problem #697 -- quantum-testable instance.

Erdos problem #697 (per erdosproblems.com / manman4/erdosproblems data/problems.yaml,
number: "697") is tagged ["number theory", "divisors"] and its `oeis` field is
["N/A"] -- the dataset records NO OEIS sequence id for this problem. There is
therefore no specific OEIS sequence to bind a circuit to, and this script says
so plainly rather than inventing one.

LIMITATION: because no OEIS id is attached to problem #697, this script does
not test membership in "the" sequence for #697 (there isn't one to test). What
it does instead, honestly scoped to the problem's own tags, is build a genuine
Grover-search quantum circuit over a small, finite, classically-checkable
divisor property that sits squarely inside the problem's tag domain
("number theory", "divisors"):

    Property tested: n is an ABUNDANT NUMBER, i.e. sigma(n) - n > n, where
    sigma(n) is the sum of all positive divisors of n (so sigma(n) - n is the
    sum of the PROPER divisors of n), searched over the instance space
    n in {0, 1, ..., 15} (4 qubits, N = 16).

Classical ground truth (computed here, from first principles, not copied from
OEIS): for n in 0..15, this script computes sigma(n) - n by trial division and
finds the exact set of abundant n in that range. Within 0..15 the classical
answer works out to the single marked value n = 12 (proper divisors
1, 2, 3, 4, 6 sum to 16 > 12), a standard, independently checkable fact about
abundant numbers (the smallest abundant number is 12; OEIS A005101, the
abundant numbers, begins 12, 18, 20, 24, ... -- consistent with n=12 being the
only one below 16).

Circuit: exact single-target Grover's algorithm on 4 qubits, with a
diffusion-optimal number of iterations for a single marked item out of 16
(floor(pi/4 * sqrt(16)) = 3 iterations), then measurement. The oracle is a
plain multi-controlled-Z phase oracle built directly from the classically
computed marked bitstring(s) -- no OEIS values, and no shortcuts, are baked
into the oracle; it is derived mechanically from the classical search result
computed in this same script.

PASS/FAIL: the script runs the circuit on the ideal AerSimulator, takes the
most frequently measured bitstring, converts it back to an integer, and
compares it against the classically computed set of abundant numbers in
0..15. It prints PASS if they match (with high measured probability) and
FAIL otherwise.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16


def divisor_sum(n: int) -> int:
    """sigma(n): sum of all positive divisors of n, by trial division."""
    if n <= 0:
        return 0
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            total += d
    return total


def is_abundant(n: int) -> bool:
    """n is abundant iff sum of PROPER divisors (sigma(n) - n) exceeds n."""
    if n <= 0:
        return False
    return (divisor_sum(n) - n) > n


def classical_marked_set(n_max: int):
    return sorted(n for n in range(n_max) if is_abundant(n))


def build_oracle(n_qubits: int, marked_values):
    """Phase oracle: flips the sign of each marked computational basis state.

    Implemented with a multi-controlled Z per marked bitstring, using X gates
    to map 0-bits to the "control on |1>" convention MCX/MCZ expect.
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")  # MSB first; qubit order handled below
        # Qiskit qubit 0 is least-significant in the bitstring produced by
        # measurement (Qiskit reports classical bits with qubit 0 as the
        # rightmost character), so map bit i of `bits` (MSB-first) to qubit
        # (n_qubits-1-i).
        zero_qubits = [
            (n_qubits - 1 - i) for i, b in enumerate(bits) if b == "0"
        ]
        if zero_qubits:
            qc.x(zero_qubits)
        # multi-controlled Z on all n_qubits qubits (last qubit is target,
        # phase kickback via H-MCX-H realizes a symmetric multi-controlled Z)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits: int, marked_values, iterations: int):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    marked = classical_marked_set(N)
    print(f"Instance: n in 0..{N - 1} (N={N}, {N_QUBITS} qubits)")
    print(f"Classical search: abundant numbers in range = {marked}")

    if len(marked) != 1:
        # Our chosen iteration count assumes exactly one marked item; if the
        # search space ever changes this guards against a silent mismatch.
        raise RuntimeError(
            f"Expected exactly one abundant number below {N}, found {marked}"
        )

    m = len(marked)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / m)))
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(N_QUBITS, marked, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert measured bitstrings (Qiskit: qubit 0 is rightmost char) to ints.
    int_counts = Counter()
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        int_counts[value] += c

    most_common_value, most_common_count = int_counts.most_common(1)[0]
    probability = most_common_count / shots

    print(f"Measured top outcome: n={most_common_value} "
          f"(probability={probability:.3f} over {shots} shots)")
    print(f"Full measured distribution (by n): "
          f"{dict(sorted(int_counts.items()))}")

    classical_answer = marked[0]
    quantum_found_correct_value = (most_common_value == classical_answer)
    # Grover with the optimal iteration count for a single marked item out of
    # 16 should concentrate the vast majority of amplitude on the target.
    high_confidence = probability > 0.8

    success = quantum_found_correct_value and high_confidence

    print(f"Classical answer: n={classical_answer}")
    print(f"Quantum circuit found correct value: {quantum_found_correct_value}")
    print(f"Quantum result high-confidence (p>0.8): {high_confidence}")

    if success:
        print("PASS")
    else:
        print("FAIL")

    return success


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
