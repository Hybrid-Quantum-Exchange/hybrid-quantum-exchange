"""
Erdos problem #878 -- quantum-testable instance
================================================

Erdos problem #878 (data/problems.yaml, number "878") is tagged "number
theory" and lists OEIS id A339378 ("possible" match noted in the source
metadata).

A339378 is the "power-sum of n": for each *distinct* prime divisor p of n,
take the highest power of p that does not exceed n (i.e. p**floor(log_p(n))),
and sum those highest-powers over all distinct prime divisors of n:

    a(n) = sum_{p prime, p | n} p ** floor(log_p(n))

(a(1) = 0, the empty sum.)

Classical property tested here
-------------------------------
For the finite instance n in {1, 2, ..., 16}, this script computes a(n) for
every n from first principles (trial division for prime factors, then the
largest power of each prime factor that is <= n), and finds the *unique*
n in that range with a(n) == 17. Classically (verified in this script by
brute force over the whole range before any quantum code runs):

    n=1..16 power-sums: 0,2,3,4,5,7,7,8,9,13,11,17,13,15,14,16
    unique n with a(n) == 17  =>  n = 12
        (check: 12 = 2^2 * 3; highest power of 2 <= 12 is 8; highest power
         of 3 <= 12 is 9; 8 + 9 = 17)

Quantum circuit
----------------
This is exactly the "small search space whose answer is a known term" case:
we use Grover's algorithm on 4 qubits (representing m = n - 1 in {0..15},
i.e. n in {1..16}) with an oracle that marks the unique computed winner
(n = 12, i.e. m = 11 = 0b1011). A single Grover iteration (optimal for one
marked item out of 16, since floor(pi/4 * sqrt(16)) = 3, but we use the
standard formula and clamp iteration count sensibly) amplifies the marked
basis state; we then run the circuit on Qiskit's ideal AerSimulator and
check that the most frequently measured 4-bit string decodes back to
n = 12, matching the classically-derived answer.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of A339378's power-sum, from first principles.
# ---------------------------------------------------------------------------
def distinct_prime_factors(n: int):
    factors = []
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            factors.append(p)
            while m % p == 0:
                m //= p
        p += 1 if p == 2 else 2
    if m > 1:
        factors.append(m)
    return factors


def power_sum(n: int) -> int:
    """a(n) = sum over distinct prime divisors p of n of the highest power
    of p that does not exceed n. a(1) = 0 (empty sum, 1 has no prime
    divisors)."""
    if n == 1:
        return 0
    total = 0
    for p in distinct_prime_factors(n):
        power = p
        while power * p <= n:
            power *= p
        total += power
    return total


N_MAX = 16  # instance size: n in {1, ..., 16}
TARGET_VALUE = 17

table = {n: power_sum(n) for n in range(1, N_MAX + 1)}
winners = [n for n, v in table.items() if v == TARGET_VALUE]

assert len(winners) == 1, f"expected a unique winner, got {winners}"
classical_answer_n = winners[0]

print("A339378 power-sum table for n = 1..16:")
for n in range(1, N_MAX + 1):
    print(f"  a({n:2d}) = {table[n]}")
print(f"Unique n in 1..16 with a(n) == {TARGET_VALUE}: n = {classical_answer_n}")

# Sanity check by direct definition for the winner.
pf = distinct_prime_factors(classical_answer_n)
direct = sum(
    max(p**k for k in range(1, 20) if p**k <= classical_answer_n) for p in pf
)
assert direct == TARGET_VALUE
assert classical_answer_n == 12

# ---------------------------------------------------------------------------
# 2. Grover search over m = n - 1 in {0, ..., 15} (4 qubits) for the winner.
# ---------------------------------------------------------------------------
NUM_QUBITS = 4
winner_m = classical_answer_n - 1  # 0-indexed target for the qubit register
winner_bits = format(winner_m, f"0{NUM_QUBITS}b")  # MSB..LSB string


def apply_oracle(qc: QuantumCircuit, target_bits: str):
    """Flip the phase of the single computational basis state matching
    target_bits (a string of '0'/'1', MSB first over qubits n-1..0)."""
    zero_positions = [i for i, b in enumerate(reversed(target_bits)) if b == "0"]
    if zero_positions:
        qc.x(zero_positions)
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    if zero_positions:
        qc.x(zero_positions)


def apply_diffuser(qc: QuantumCircuit):
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))


num_iterations = max(1, round((math.pi / 4) * math.sqrt(2**NUM_QUBITS)))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(num_iterations):
    apply_oracle(qc, winner_bits)
    apply_diffuser(qc)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

simulator = AerSimulator()
compiled = transpile(qc, simulator)
result = simulator.run(compiled, shots=4096).result()
counts = result.get_counts()

most_common_bits = max(counts, key=counts.get)
# Qiskit's bit string is c[NUM_QUBITS-1] ... c[0]; our winner_bits was built
# MSB..LSB over qubit indices NUM_QUBITS-1..0, which matches this ordering.
quantum_m = int(most_common_bits, 2)
quantum_n = quantum_m + 1

print(f"\nGrover search ({num_iterations} iteration(s), {NUM_QUBITS} qubits, 4096 shots)")
print(f"Measurement counts: {counts}")
print(f"Most frequent outcome decodes to n = {quantum_n} "
      f"(probability {counts[most_common_bits] / 4096:.3f})")

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------
ran_ok = True
verified = quantum_n == classical_answer_n

if verified:
    print(f"\nPASS: quantum search found n = {quantum_n}, matching the classical "
          f"answer n = {classical_answer_n} (unique n in 1..16 with a(n) = {TARGET_VALUE}).")
else:
    print(f"\nFAIL: quantum search found n = {quantum_n}, but the classical "
          f"answer is n = {classical_answer_n}.")
