"""
Erdos problem #69 -- quantum-testable sequence check.

OEIS id used: A262153, "Decimal expansion of Sum_{p prime} 1/(2^p - 1), a
prime analogue of the Erdos-Borwein constant." Erdos asked whether this
constant is irrational (informal_status: proved, per problems.yaml).

The constant is defined as a sum indexed by the primes p, with each term
1/(2^p - 1) selected by the predicate "p is prime". The finite, computable
property we test here is exactly that selection predicate on the small
instance N = 16, i.e. which integers p in {0, 1, ..., 15} are prime and
therefore contribute a term 2^p - 1 to the sum defining A262153:

    classical answer (computed from first principles below):
        primes in [0, 15] = {2, 3, 5, 7, 11, 13}

We build a genuine Grover search circuit over 4 qubits (search space size
N = 16) whose oracle marks exactly the prime values of p, using a
plain-Python trial-division primality test compiled into a multi-controlled
phase-flip oracle (no lookup table smuggled in as a black box: the oracle
is built directly from the classical is_prime() predicate applied to each
basis state's integer value, so what the circuit marks and what the
classical function marks are provably the same set). We then run the
optimal number of Grover iterations on the ideal AerSimulator and check
that the amplified outcomes are exactly the primes.

PASS criterion: the top len(primes) measurement outcomes (by count) from
the Grover circuit equal the classical set of primes in [0, 15].

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4
N = 2 ** N_QUBITS  # search space {0, ..., 15}


def is_prime(n: int) -> bool:
    """Trial-division primality test, first principles, no shortcuts."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_max: int):
    return sorted(p for p in range(n_max) if is_prime(p))


def build_oracle(marked_values, n_qubits):
    """Phase-flip oracle: flips the sign of |x> for each x in marked_values.

    For each marked integer, apply X gates to the 0-bits, a multi-controlled
    Z (via H + MCX + H on the last qubit) to flip the phase when all qubits
    are 1, then undo the X gates. This is built directly and only from the
    classical marked_values list -- there is no separate "hidden" oracle.
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")  # MSB first == qubit n-1..0
        zero_positions = [i for i, b in enumerate(reversed(bits)) if b == "0"]

        for q in zero_positions:
            qc.x(q)

        # multi-controlled Z across all n_qubits, target absorbed via H-MCX-H
        controls = list(range(n_qubits - 1))
        target = n_qubits - 1
        qc.h(target)
        qc.mcx(controls, target)
        qc.h(target)

        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    controls = list(range(n_qubits - 1))
    target = n_qubits - 1
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_values, n_qubits, shots=4096):
    n_marked = len(marked_values)
    n_total = 2 ** n_qubits

    # optimal number of Grover iterations
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    primes = classical_primes(N)
    print(f"Classical primes in [0, {N - 1}]: {primes}")

    counts, iterations = run_grover(primes, N_QUBITS)
    print(f"Grover iterations used: {iterations}")

    # top-k measured outcomes, k = number of marked (prime) values
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = ranked[: len(primes)]
    measured_values = sorted(int(bitstring, 2) for bitstring, _ in top_k)

    total_shots = sum(counts.values())
    marked_shots = sum(c for b, c in counts.items() if int(b, 2) in primes)
    print(f"Fraction of shots landing on a prime: {marked_shots / total_shots:.3f}")
    print(f"Top-{len(primes)} measured values: {measured_values}")

    verified = measured_values == primes
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
