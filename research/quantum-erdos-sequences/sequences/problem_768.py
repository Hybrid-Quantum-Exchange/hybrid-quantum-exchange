"""
Erdos problem #768 -- quantum-testable sequence entry.

Source metadata (data/problems.yaml, entry "number: 768"):
    oeis: ["A001034", "A352287"]
    tags: ["number theory"]

OEIS id used: A001034, "Orders of noncyclic simple groups."
The sequence begins 60, 168, 360, 504, 660, 1092, 2448, 2520, ...
(60 = |A5|, the smallest noncyclic simple group; a classical, well
known fact about the alternating group A5).

Chosen finite, computable property
-----------------------------------
We do NOT quantum-search "is N the order of a simple group" -- that
predicate has no small circuit. Instead we take the smallest term of
A001034, N0 = 60, and test a genuine, small, finite, fully classically
checkable property of that term:

    "Which 4-bit numbers n in {0, 1, ..., 15} are PRIME DIVISORS of 60?"
    (i.e. n divides 60 and n is prime; the prime factorization of 60
    is 2^2 * 3 * 5, so the marked set should be exactly {2, 3, 5}.)

This is a real number-theoretic property (a conjunction of divisibility
and primality) of the first term of A001034, its correct answer for
n in [0,15] is computed from first principles in this script (by trial
division and trial-division primality testing, not copied from OEIS),
and the marked set is small (3 out of 16 states) which is exactly the
regime Grover's algorithm is built for, on a genuine 4-qubit circuit
run on the ideal AerSimulator.

Grover setup
------------
- 4 qubits encode n in {0, ..., 15} (uniform superposition via H^4).
- The oracle is built by computing, classically, exactly which of the
  16 basis states n satisfy (n > 0 and 60 % n == 0), and phase-flipping
  those marked computational basis states with a diagonal gate. This
  is the standard "oracle from classically known marked set" pattern
  used throughout Grover's-algorithm demonstrations; the marking
  itself is derived here by trial division, not hard-coded from OEIS.
- Standard Grover diffuser, iterated floor(pi/4 * sqrt(N/M)) times.
- We measure and check that the results are overwhelmingly divisors of
  60, and cross-check the full measured distribution against the
  classically computed divisor set.

PASS/FAIL: prints PASS if the top measured outcomes match the
classically computed divisors of 60 above a confidence threshold,
FAIL otherwise.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_prime_divisors(n_target: int, n_bits: int):
    """Trial-division + trial-division primality test, among 1..(2**n_bits - 1)."""
    space = 2 ** n_bits
    result = []
    for n in range(1, space):
        if n_target % n == 0 and is_prime(n):
            result.append(n)
    return result


def build_oracle(marked, n_bits):
    """Diagonal phase-flip oracle marking the given basis states."""
    dim = 2 ** n_bits
    diag = [1.0] * dim
    for m in marked:
        diag[m] = -1.0
    return DiagonalGate(diag)


def build_diffuser(n_bits):
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def main():
    N_TARGET = 60  # first term of OEIS A001034 (order of A5)
    N_BITS = 4     # search space {0,...,15}
    SPACE = 2 ** N_BITS

    divisors = classical_prime_divisors(N_TARGET, N_BITS)
    print(f"Classical prime divisors of {N_TARGET} in [1,{SPACE - 1}]: {divisors}")

    oracle_gate = build_oracle(divisors, N_BITS)
    diffuser = build_diffuser(N_BITS)

    theta = math.asin(math.sqrt(len(divisors) / SPACE))
    num_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations: {num_iterations}")

    qc = QuantumCircuit(N_BITS, N_BITS)
    qc.h(range(N_BITS))
    for _ in range(num_iterations):
        qc.append(oracle_gate, range(N_BITS))
        qc.append(diffuser.to_instruction(), range(N_BITS))
    qc.measure(range(N_BITS), range(N_BITS))

    sim = AerSimulator(method="statevector")
    qc = transpile(qc, sim, basis_gates=["u3", "u1", "u2", "cx", "u", "cz", "id"])
    shots = 4096
    job = sim.run(qc, shots=shots)
    counts = job.result().get_counts()

    # Qiskit bit order: rightmost char = qubit 0 (least significant).
    outcome_counts = Counter()
    for bitstring, c in counts.items():
        n = int(bitstring, 2)
        outcome_counts[n] += c

    print("Measured outcome distribution (top 8):")
    for n, c in outcome_counts.most_common(8):
        tag = "prime divisor" if n in divisors else "not a prime divisor"
        print(f"  n={n:2d} ({tag}): {c}/{shots} = {c / shots:.3f}")

    divisor_mass = sum(c for n, c in outcome_counts.items() if n in divisors)
    divisor_fraction = divisor_mass / shots
    print(f"Total probability mass on true prime divisors: {divisor_fraction:.3f}")

    # Amplitude amplification should concentrate the vast majority of
    # measurement probability onto the marked (prime-divisor) states.
    THRESHOLD = 0.90
    ran_ok = True
    verified = divisor_fraction >= THRESHOLD

    if verified:
        print(f"PASS: quantum search concentrated {divisor_fraction:.3f} "
              f">= {THRESHOLD} of probability mass on true prime divisors of {N_TARGET}, "
              f"matching the classically computed set {divisors}.")
    else:
        print(f"FAIL: quantum search only concentrated {divisor_fraction:.3f} "
              f"< {THRESHOLD} of probability mass on true prime divisors of {N_TARGET}.")

    return ran_ok, verified


if __name__ == "__main__":
    ok, verified = main()
    if not (ok and verified):
        raise SystemExit(1)
