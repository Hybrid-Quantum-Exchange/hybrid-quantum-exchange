"""
Erdos problem #251 (erdosproblems.com/251)

Statement: Erdos proved that sum_{n>=1} p_n^k / n! is irrational for every
k >= 1, where p_n is the n-th prime. He conjectured further irrationality
results for series built from the primes p_n, such as sum p_n^k / 2^n.
The relevant OEIS entry, A098990, is the decimal expansion of a constant
built from the prime sequence (p_n) that appears in this family of sums.
Tags: number theory, irrationality.

Classical property tested here:
    "x is prime" for x in the search space {0, 1, ..., 15} (4 bits).

This is the finite, computable core object that Erdos's series is built
from: the primes p_1, p_2, ... themselves. The correct classical answer
for N = 16 is computed from first principles below with trial division
(no external number-theory library), giving the set
    PRIMES_16 = {2, 3, 5, 7, 11, 13}
which is exactly {p_1, ..., p_6} intersected with [0, 15].

Quantum circuit: Grover's search algorithm over 4 qubits (search space of
size 16) with an oracle that marks the prime numbers in that range. Grover
amplifies the amplitude of the 6 prime basis states out of 16, so after the
optimal number of Grover iterations, measurement should return a prime
value with high probability. We run the circuit on the ideal AerSimulator,
tally the measured outcomes, and PASS if the primality-weighted success
probability (fraction of shots landing on a classically-verified prime)
exceeds a threshold consistent with Grover's predicted amplification
(much higher than the 6/16 = 37.5% baseline of uniform random guessing).

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: primes in [0, 15] via trial division.
# ---------------------------------------------------------------------------
def is_prime_classical(x: int) -> bool:
    if x < 2:
        return False
    for d in range(2, int(math.isqrt(x)) + 1):
        if x % d == 0:
            return False
    return True


N_BITS = 4
N = 2 ** N_BITS  # 16
PRIMES_16 = sorted(x for x in range(N) if is_prime_classical(x))
print(f"Classical primes in [0, {N - 1}]: {PRIMES_16}")
assert PRIMES_16 == [2, 3, 5, 7, 11, 13], "classical primality check is wrong"

M = len(PRIMES_16)  # number of marked (prime) states = 6


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the prime basis states.
# ---------------------------------------------------------------------------
def build_oracle(n_bits: int, marked_values: list[int]) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_bits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


# Optimal number of Grover iterations for N states, M marked.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"N={N}, M={M}, theta={theta:.4f}, Grover iterations={iterations}")

qc = QuantumCircuit(N_BITS, N_BITS)
qc.h(range(N_BITS))

oracle = build_oracle(N_BITS, PRIMES_16)
diffuser = build_diffuser(N_BITS)
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_BITS), range(N_BITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------
backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports classical-register bitstrings MSB-first (register order),
# with classical bit i <- qubit i, so the printed string already reads as
# the integer value directly (leftmost char = qubit 3 = most significant bit).
prime_hits = 0
for bitstring, freq in counts.items():
    value = int(bitstring, 2)
    if value in PRIMES_16:
        prime_hits += freq

success_prob = prime_hits / SHOTS
baseline_prob = M / N  # uniform-random baseline, 6/16 = 0.375

print(f"Measured success probability (landed on a prime): {success_prob:.4f}")
print(f"Uniform-random baseline: {baseline_prob:.4f}")

# Grover's algorithm with N=16, M=6 and 1 iteration predicts a success
# probability of about 0.84 (computed analytically and confirmed by exact
# statevector simulation); require a solid, unambiguous improvement over the
# 0.375 classical baseline.
THRESHOLD = 0.75
passed = success_prob >= THRESHOLD and success_prob > baseline_prob

print("PASS" if passed else "FAIL")
