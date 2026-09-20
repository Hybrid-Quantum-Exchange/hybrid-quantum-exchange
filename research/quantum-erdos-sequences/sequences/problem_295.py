"""
Erdos problem #295 -- quantum-testable instance.

OEIS id used: A192881 -- "Number of terms in the shortest Egyptian fraction
representation of 1 in which the first term is 1/n."  a(2) = 3, meaning the
shortest way to write 1 as a sum of distinct unit fractions starting with
1/2 uses exactly 3 terms.  The classical witness for a(2) = 3 is the
well-known identity

    1/2 + 1/3 + 1/6 = 1

Classical property tested here (derived and checked from first principles
in this script, not copied from OEIS):

    Among all pairs of distinct integers (a, b) with 2 <= a < b <= 9,
    a != 2, b != 2, find the pair(s) for which

        1/2 + 1/a + 1/b == 1   (exact rational equality)

    This finite brute-force search is done first with Python's `fractions`
    module to get the ground-truth answer.  There is exactly one solution
    in this range: (a, b) = (3, 6), which is exactly the identity underlying
    A192881(2) = 3 (three unit-fraction terms: 1/2, 1/3, 1/6).

Quantum circuit:

    We encode a in {2,...,9} as a 3-qubit register (offset value = a - 2,
    0..7) and b in {2,...,9} as a second 3-qubit register (offset value =
    b - 2, 0..7), for a 6-qubit, 64-state search space (N = 64, satisfying
    N <= ~64).  We build a genuine Grover search circuit:

      - The oracle phase-flips exactly the basis state(s) whose (a, b)
        satisfy the classically-verified equation 1/2 + 1/a + 1/b == 1
        (the oracle's marked bitstring(s) come directly from the classical
        brute-force search above -- nothing is hard-coded independently).
      - The standard Grover diffusion operator amplifies the marked
        state(s).
      - With 64 items and 1 marked solution, the optimal number of Grover
        iterations is round(pi/4 * sqrt(64/1)) = 6.

    We run the circuit on the ideal AerSimulator and check that the most
    frequently measured bitstring decodes to the classically-found
    solution (a, b) = (3, 6).

PASS/FAIL: the script prints PASS if the quantum measurement's most likely
outcome matches the classical solution, FAIL otherwise.
"""

import math
from fractions import Fraction
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force search for (a, b) with
#    2 <= a < b <= 9, a != 2, b != 2, satisfying 1/2 + 1/a + 1/b == 1.
# ---------------------------------------------------------------------------

LOW, HIGH = 2, 9  # inclusive range for a and b

classical_solutions = []
for a in range(LOW, HIGH + 1):
    for b in range(LOW, HIGH + 1):
        if a == 2 or b == 2 or a >= b:
            continue
        total = Fraction(1, 2) + Fraction(1, a) + Fraction(1, b)
        if total == 1:
            classical_solutions.append((a, b))

assert classical_solutions == [(3, 6)], (
    f"Unexpected classical search result: {classical_solutions}"
)

# This confirms, from first principles, the identity 1/2 + 1/3 + 1/6 = 1,
# i.e. a 3-term Egyptian fraction representation of 1 starting with 1/2,
# which is exactly the fact A192881(2) = 3 records.
target_a, target_b = classical_solutions[0]
print(f"Classical brute force: unique solution (a, b) = ({target_a}, {target_b})")
print(f"  Check: 1/2 + 1/{target_a} + 1/{target_b} = "
      f"{Fraction(1, 2) + Fraction(1, target_a) + Fraction(1, target_b)}")

# ---------------------------------------------------------------------------
# 2. Encode the solution as a 6-bit marked bitstring for Grover search.
#    a, b in {2,...,9} -> offset value (a-2), (b-2) in {0,...,7}, 3 bits each.
# ---------------------------------------------------------------------------

def to_bits(value, width):
    return format(value, f"0{width}b")

a_off = target_a - LOW  # 0..7
b_off = target_b - LOW  # 0..7
a_bits = to_bits(a_off, 3)
b_bits = to_bits(b_off, 3)
# Qubit order: q0..q2 = register A (a), q3..q5 = register B (b).
# Qiskit bit ordering in the statevector/counts string is q(n-1)...q0, so
# build the target string accordingly (MSB-first = highest-index qubit).
marked_bitstring = b_bits + a_bits  # matches Qiskit's little-endian count keys

n_qubits = 6
N = 2 ** n_qubits  # 64
M = 1  # one marked item
iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Marked bitstring (qiskit order, q5..q0): {marked_bitstring}")
print(f"Grover iterations: {iterations}")


def oracle(qc: QuantumCircuit, target: str):
    """Phase-flip the single basis state |target> (target given in
    Qiskit's q(n-1)...q0 string order)."""
    n = len(target)
    # target[0] corresponds to qubit n-1, target[-1] to qubit 0.
    zero_qubits = [n - 1 - i for i, bit in enumerate(target) if bit == "0"]
    for q in zero_qubits:
        qc.x(q)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for q in zero_qubits:
        qc.x(q)


def diffuser(qc: QuantumCircuit, n: int):
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))

for _ in range(iterations):
    oracle(qc, marked_bitstring)
    diffuser(qc, n_qubits)

qc.measure(range(n_qubits), range(n_qubits))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 2048
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

most_common_bitstring, most_common_count = Counter(counts).most_common(1)[0]
probability = most_common_count / shots
print(f"Most frequent measured bitstring: {most_common_bitstring} "
      f"(p = {probability:.3f}, count = {most_common_count}/{shots})")

# Decode back to (a, b): bits are q5..q0 = b_bits(3) + a_bits(3)
measured_b_bits = most_common_bitstring[0:3]
measured_a_bits = most_common_bitstring[3:6]
measured_a = int(measured_a_bits, 2) + LOW
measured_b = int(measured_b_bits, 2) + LOW

print(f"Decoded quantum result: (a, b) = ({measured_a}, {measured_b})")
print(f"Classical result:       (a, b) = ({target_a}, {target_b})")

verified = (
    most_common_bitstring == marked_bitstring
    and (measured_a, measured_b) == (target_a, target_b)
    and probability > 0.5
)

if verified:
    print("PASS")
else:
    print("FAIL")
