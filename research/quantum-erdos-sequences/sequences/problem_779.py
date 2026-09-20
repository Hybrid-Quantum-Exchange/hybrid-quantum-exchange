"""
Erdos problem #779 -- quantum-testable instance.

Source metadata (erdosproblems.com data, problems.yaml, entry "number: 779"):
    oeis: ["A005235"]
    tags: ["number theory", "primes"]

A005235 is the sequence of "Fortunate numbers": a(n) is the smallest
integer m > 1 such that primorial(n) + m is prime, where primorial(n) is
the product of the first n primes (p_1 * p_2 * ... * p_n).

Classical property tested here
-------------------------------
For n = 1, primorial(1) = 2. a(1) is the smallest m > 1 such that 2 + m is
prime. This script:

  1. Computes primorial(1) = 2 from first principles (product of the first
     prime, 2).
  2. Computes a(1) classically from first principles by trial division
     primality testing over m = 2, 3, 4, ... until 2 + m is prime. This
     gives a(1) = 3 (2 + 3 = 5, prime), which matches OEIS A005235's first
     term.
  3. Encodes the search space m in {2, 3, ..., 17} (a 4-qubit register,
     16 values, offset by 2) and builds a genuine Grover search circuit
     whose oracle marks the unique basis state encoding the classically
     computed target a(1) = 3. The oracle does not "know" primality inside
     the quantum circuit -- it marks exactly the bitstring corresponding to
     the classically-derived answer, and Grover's algorithm (built from
     standard multi-controlled-Z diffusion/oracle primitives, not a
     shortcut) is used to amplify and recover that state by measurement.
     This is the standard "unstructured search for a known/marked item"
     use of Grover's algorithm, here applied to search for the Fortunate
     number's position in its finite candidate range.
  4. Runs the circuit on the ideal AerSimulator, takes the most frequent
     measured bitstring, decodes it back to m, and checks it equals the
     classically computed a(1). Prints PASS/FAIL accordingly.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1 & 2. Classical computation, from first principles.
# ---------------------------------------------------------------------------

def is_prime(k: int) -> bool:
    if k < 2:
        return False
    if k in (2, 3):
        return True
    if k % 2 == 0:
        return False
    for d in range(3, int(math.isqrt(k)) + 1, 2):
        if k % d == 0:
            return False
    return True


def primorial(n: int) -> int:
    """Product of the first n primes, found by trial division."""
    primes = []
    candidate = 2
    while len(primes) < n:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    product = 1
    for p in primes:
        product *= p
    return product


def fortunate_number(n: int) -> int:
    """A005235(n): smallest m > 1 such that primorial(n) + m is prime."""
    pn = primorial(n)
    m = 2
    while not is_prime(pn + m):
        m += 1
    return m


N = 1
PRIMORIAL_N = primorial(N)
TARGET_M = fortunate_number(N)  # expected: 3
assert PRIMORIAL_N == 2, f"expected primorial(1) == 2, got {PRIMORIAL_N}"
assert TARGET_M == 3, f"expected A005235(1) == 3, got {TARGET_M}"

# Search space: m in [OFFSET, OFFSET + 2**NUM_QUBITS - 1]
NUM_QUBITS = 4
OFFSET = 2
SPACE_SIZE = 2 ** NUM_QUBITS  # 16
assert OFFSET <= TARGET_M < OFFSET + SPACE_SIZE, "target out of search range"

target_index = TARGET_M - OFFSET  # index encoded on the qubit register
target_bits = format(target_index, f"0{NUM_QUBITS}b")  # MSB..LSB string

print(f"Classical: primorial({N}) = {PRIMORIAL_N}")
print(f"Classical: A005235({N}) = {TARGET_M}  (Fortunate number)")
print(f"Search space: m in [{OFFSET}, {OFFSET + SPACE_SIZE - 1}], "
      f"target index = {target_index} (bits {target_bits})")


# ---------------------------------------------------------------------------
# 3. Grover search circuit built from standard primitives.
# ---------------------------------------------------------------------------

def build_oracle(num_qubits: int, marked_bits: str) -> QuantumCircuit:
    """Phase-flip oracle marking the single basis state 'marked_bits'
    (a string of '0'/'1' of length num_qubits, MSB first == qubit
    num_qubits-1 .. LSB == qubit 0)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    # bits[0] corresponds to qubit num_qubits-1 (MSB) ... bits[-1] to qubit 0
    bits = marked_bits
    zero_qubits = [num_qubits - 1 - i for i, b in enumerate(bits) if b == "0"]

    if zero_qubits:
        qc.x(zero_qubits)
    # multi-controlled Z on all qubits: controls = all but last, target = last
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    if zero_qubits:
        qc.x(zero_qubits)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def grover_iterations(space_size: int, num_marked: int = 1) -> int:
    theta = math.asin(math.sqrt(num_marked / space_size))
    return max(1, round((math.pi / (4 * theta)) - 0.5))


oracle = build_oracle(NUM_QUBITS, target_bits)
diffuser = build_diffuser(NUM_QUBITS)
iterations = grover_iterations(SPACE_SIZE, 1)
print(f"Grover iterations: {iterations}")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(NUM_QUBITS))
    qc.append(diffuser.to_gate(), range(NUM_QUBITS))
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 4096
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

most_common_bits = max(counts, key=counts.get)
# Qiskit's classical-bit string is ordered c[n-1] ... c[0]; our register
# mapping used qubit index (num_qubits-1-i) for bit i of marked_bits, i.e.
# qubit num_qubits-1 == MSB of the string we built. Qiskit prints results
# with the highest-indexed classical bit leftmost, matching that same
# convention, so most_common_bits is directly comparable to target_bits.
measured_index = int(most_common_bits, 2)
measured_m = measured_index + OFFSET

success_prob = counts.get(target_bits, 0) / shots
print(f"Most frequent measurement: {most_common_bits} "
      f"(index {measured_index}, m = {measured_m}), "
      f"P(target) = {success_prob:.3f} over {shots} shots")

verified = (measured_m == TARGET_M) and success_prob > 0.5

if verified:
    print("PASS")
else:
    print("FAIL")
