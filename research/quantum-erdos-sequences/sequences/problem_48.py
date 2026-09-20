"""
Erdos problem #48 (erdosproblems.com/48) -- quantum-testable instance.

Source metadata (from data/problems.yaml in the manman4/erdosproblems clone):
    number: "48"
    prize: no
    informal_status: proved
    formal_status: Lean (formalized 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION, stated honestly up front: problem 48's YAML entry carries no OEIS
sequence id ("N/A"). There is therefore no specific OEIS sequence for this
script to test membership/terms against, and no way to derive a
problem-48-specific property from the metadata alone without fabricating one.
Per the task instructions for this case, this script still builds a real,
honest quantum computation rather than faking a connection to problem 48's
actual (unspecified-here) statement. The property chosen below only shares
problem 48's stated topic tag, "number theory": primality, the most basic
finite/computable number-theoretic property available, over a small search
space that fits a few qubits.

Classical property under test
------------------------------
Search space: integers n in [0, 63] (6 qubits, N = 64).
Property:     n is prime.
Classical set (computed here from first principles by trial division, not
copied from any table):
    PRIMES = {n in [0,63] : n has no divisor d with 1 < d < n}

Quantum computation
--------------------
A Grover search circuit is built over 6 qubits:
  - Oracle: a diagonal phase-flip unitary that negates the amplitude of
    every basis state |n> whose n is classically prime (built directly from
    the classically computed PRIMES set above -- the oracle does not know
    the answer in advance, it is literally the diagonal +1/-1 matrix
    determined by that set, so the circuit is doing real amplitude
    amplification on marked states, not returning a precomputed answer).
  - Diffusion: the standard Grover diffusion operator about the uniform
    superposition.
  - Iteration count: floor(pi/4 * sqrt(N/M)) where M = |PRIMES|, the
    standard optimal Grover iteration count for multiple marked items.

The circuit is run on the ideal AerSimulator (statevector method, no noise).
PASS criterion: sampling the final state's most frequent outcomes, the
probability mass landing on prime values must be much larger than the
uniform-random baseline M/N, and the single most likely measured outcome
must itself be prime (i.e. Grover actually amplified real primes, verified
against the classically computed PRIMES set).
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_max: int):
    return sorted(n for n in range(n_max + 1) if is_prime(n))


def build_oracle(marked: set, n_qubits: int) -> QuantumCircuit:
    """Diagonal phase-flip oracle: |n> -> -|n> for n in `marked`, else |n> -> |n>."""
    dim = 2 ** n_qubits
    diag = np.ones(dim, dtype=complex)
    for n in marked:
        diag[n] = -1.0
    qc = QuantumCircuit(n_qubits, name="Oracle")
    qc.append(Operator(np.diag(diag)), range(n_qubits))
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator about the uniform superposition."""
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover_prime_search(n_qubits: int, marked: set, shots: int = 4096):
    n_max = 2 ** n_qubits - 1
    n_total = 2 ** n_qubits
    m = len(marked)

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    iterations = max(1, round((math.pi / 4) * math.sqrt(n_total / m)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator(method="statevector")
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit orders classical bits with qubit 0 as the rightmost character;
    # our register directly encodes n in binary with qubit 0 as the LSB, so
    # int(bitstring, 2) recovers n correctly.
    n_counts = Counter()
    for bitstring, c in counts.items():
        n_counts[int(bitstring, 2)] += c

    return n_counts, iterations


def main():
    n_qubits = 6
    n_max = 2 ** n_qubits - 1  # 63

    primes = classical_primes(n_max)
    marked = set(primes)
    m = len(marked)
    n_total = n_max + 1

    print(f"Erdos problem #48 -- OEIS: N/A in source metadata (tags: number theory)")
    print(f"Search space: n in [0,{n_max}] ({n_qubits} qubits, N={n_total})")
    print(f"Classical primes (trial division), M={m}: {primes}")

    n_counts, iterations = run_grover_prime_search(n_qubits, marked)
    shots = sum(n_counts.values())

    most_common_n, most_common_count = n_counts.most_common(1)[0]
    prime_mass = sum(c for n, c in n_counts.items() if n in marked) / shots
    baseline = m / n_total

    print(f"Grover iterations used: {iterations}")
    print(f"Most frequent measured n: {most_common_n} "
          f"(count {most_common_count}/{shots}), classically prime: {is_prime(most_common_n)}")
    print(f"Probability mass on prime outcomes: {prime_mass:.4f} "
          f"(uniform-random baseline would be {baseline:.4f})")

    top_prime = is_prime(most_common_n)
    amplified = prime_mass > 3 * baseline

    passed = top_prime and amplified

    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
