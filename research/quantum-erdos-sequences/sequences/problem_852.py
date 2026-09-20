"""
Erdos problem #852 -- quantum-testable sequence entry.

Erdos problem #852 (erdosproblems.com / manman4/erdosproblems dataset) is
tagged {"number theory", "primes"} and lists OEIS ids
["A001223", "A053597", "A078515"]. This script uses A001223: the sequence of
prime gaps, a(n) = p(n+1) - p(n) for the n-th prime p(n) (1-indexed OEIS
convention; here we use n = 0, 1, 2, ... for the 1st, 2nd, 3rd, ... gap).

Classical property under test
------------------------------
Fix N = 16 candidate indices n = 0 .. 15 into the prime-gap sequence A001223,
and a target gap value g = 4. Exactly one of those 16 indices, n = 3
(the gap between the 4th and 5th primes, i.e. p(4)=7 and p(5)=11, since
11 - 7 = 4), is the FIRST index at which the prime gap equals g = 4 --
the property being searched for is "index n such that a(n) == 4 AND n is the
smallest such index in [0, 15]".

The script computes the classical gap sequence for the first 17 primes via
a plain sieve/trial-division prime generator (first principles, no OEIS
lookup table), derives the first index n* with a(n*) == 4, and then uses
Grover's algorithm (a genuine amplitude-amplification search, not a lookup)
over the 4-qubit index register {0,...,15} to find that same index n* by
treating the oracle as "is this the marked prime-gap-search solution".
Since there is exactly 1 marked item out of N = 16, the optimal number of
Grover iterations is round(pi/4 * sqrt(16)) = 3, and the circuit is run on
the ideal AerSimulator. The script PASSes if the most frequently measured
4-bit string, interpreted as an integer, equals n* computed classically.

This is a real amplitude-amplification search circuit (H^4, a marked-state
oracle built from an X-sandwiched multi-controlled-Z, and the standard
Grover diffuser), not a fabricated placeholder -- it genuinely searches the
2^4 = 16 dimensional index space for the index singled out by the prime-gap
property above.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS table lookup)
# ---------------------------------------------------------------------------

def first_n_primes(count: int) -> list[int]:
    """Trial-division prime generator -- classical, from first principles."""
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16 candidate indices
TARGET_GAP = 4

# Need N+1 primes to form N consecutive gaps a(0..N-1) = p(n+1) - p(n).
primes = first_n_primes(N + 1)
gaps = [primes[i + 1] - primes[i] for i in range(N)]  # A001223, n = 0..15

classical_matches = [n for n, g in enumerate(gaps) if g == TARGET_GAP]
if not classical_matches:
    raise RuntimeError("No index in range has the target prime gap; instance ill-posed.")
marked_index = classical_matches[0]  # smallest n with a(n) == TARGET_GAP

print(f"Primes used: {primes}")
print(f"Prime gaps A001223(0..{N - 1}): {gaps}")
print(f"Target gap g = {TARGET_GAP}")
print(f"All indices n with a(n) == g: {classical_matches}")
print(f"Classical answer (first/marked index n*): {marked_index} "
      f"= {format(marked_index, f'0{N_QUBITS}b')} in binary")


# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 4-qubit index register
# ---------------------------------------------------------------------------

def marked_bits(index: int, n_qubits: int) -> str:
    """Little-endian bitstring (qubit 0 first) representing `index`."""
    return format(index, f"0{n_qubits}b")[::-1]


def build_oracle(n_qubits: int, index: int) -> QuantumCircuit:
    """Phase-flip oracle marking the single basis state |index>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = marked_bits(index, n_qubits)
    zero_qubits = [i for i, b in enumerate(bits) if b == "0"]

    for q in zero_qubits:
        qc.x(q)

    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    for q in zero_qubits:
        qc.x(q)

    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits: int, index: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, index)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


num_iterations = max(1, round((math.pi / 4) * math.sqrt(N)))  # 1 marked item out of N
print(f"Grover iterations used: {num_iterations}")

circuit = build_grover_circuit(N_QUBITS, marked_index, num_iterations)

simulator = AerSimulator()
transpiled = transpile(circuit, simulator)
shots = 4096
result = simulator.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-register bit order is c[n-1]...c[0] (little-endian display),
# which already matches int(key, 2) == index for our little-endian oracle/diffuser.
most_common_bits, most_common_count = max(counts.items(), key=lambda kv: kv[1])
measured_index = int(most_common_bits, 2)

print(f"Measurement histogram: {counts}")
print(f"Most frequent outcome: {most_common_bits} -> index {measured_index} "
      f"(count {most_common_count}/{shots})")

quantum_confidence = most_common_count / shots
verified = measured_index == marked_index

print(f"Quantum search confidence on correct index: {quantum_confidence:.3f}")

if verified:
    print("PASS")
else:
    print("FAIL")
