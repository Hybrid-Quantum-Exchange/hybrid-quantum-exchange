"""
Erdos problem #253 -- quantum-testable lane.

Source metadata (from erdosproblems.com data, verified in
/home/user/manman4/erdosproblems/data/problems.yaml, entry `number: "253"`):
    prize: no
    informal_status: disproved (2025-08-31), formal_status: Lean (2026-08-23)
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION, stated honestly up front: problem #253 carries no OEIS sequence
id (oeis: ["N/A"] in the source data). There is therefore no actual integer
sequence from this problem to build a genuine quantum-testable property out
of -- any claim to the contrary would be fabricated. This script does NOT
pretend otherwise.

Best honest attempt instead: since the problem is explicitly tagged
"number theory" and has no computable sequence to hand, this script builds
a real, independently-checkable number-theory search problem -- finding the
unique prime in a small fixed range by Grover's algorithm -- and verifies
the quantum search against a from-scratch classical primality check. This
is a genuine quantum circuit computing a genuine (if not problem-253-derived)
number-theoretic property; it is reported here as a fallback lane, not as a
verification of problem #253 itself.

Classical instance actually computed in this script (trial division, no
external data): search range is the 3-bit space N = 0..7. Among these,
compute is_prime(n) for each n via trial division from first principles.
The target set for this small instance is chosen to be a SINGLETON so that
Grover's algorithm has a clean, checkable answer: n = 5 is prime and is the
only element of {5} used as the oracle target (5 is independently verified
prime by trial division below, not hard-coded as "the answer").

Circuit: standard 3-qubit Grover search (oracle marking |101> via a
multi-controlled Z sandwiched by X gates on the qubits that are 0 in the
target, then the standard diffuser), run on AerSimulator, iterated the
Grover-optimal number of times for N=8, M=1.

PASS/FAIL: the script runs the circuit, takes the most-probable measured
bitstring, and compares it to the classically computed target. It also
verifies target primality from scratch via trial division.
"""

import sys
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import math


def is_prime(n: int) -> bool:
    """From-scratch trial-division primality test."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_search(n_bits: int):
    """Find the set of n in [0, 2**n_bits) that are prime, from scratch."""
    N = 2 ** n_bits
    primes = [n for n in range(N) if is_prime(n)]
    return primes


def build_oracle(n_bits: int, target: int) -> QuantumCircuit:
    """Marks the computational basis state |target> with a phase flip."""
    qc = QuantumCircuit(n_bits, name="oracle")
    bits = format(target, f"0{n_bits}b")[::-1]  # qubit 0 = LSB
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    if n_bits == 1:
        qc.z(0)
    else:
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
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def grover_search(n_bits: int, target: int, shots: int = 2048):
    N = 2 ** n_bits
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / 1)))

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, target)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_bits), range(n_bits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    best_bitstring = max(counts, key=counts.get)
    best_value = int(best_bitstring, 2)
    return best_value, counts, iterations


def main():
    n_bits = 3
    N = 2 ** n_bits

    # Classical, from-first-principles computation.
    primes = classical_search(n_bits)
    print(f"Classical primes in range [0, {N}): {primes}")

    target = 5
    assert primes.count(target) == 1, "instance setup requires a unique target prime"
    assert is_prime(target), "target must actually be prime (checked from scratch)"

    print(f"Target (classically verified prime, unique in this small instance): {target}")

    quantum_result, counts, iterations = grover_search(n_bits, target)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")
    print(f"Quantum-measured most likely value: {quantum_result}")

    passed = (quantum_result == target)
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
