"""
Erdos problem #843 (erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry "number: '843'"): informal_status = proved (2025-08-31), tags = ["number
theory"], oeis = ["N/A"].

LIMITATION, stated honestly up front: problem #843's metadata carries no OEIS
id at all ("N/A"), and the yaml entry gives no numeric sequence definition
either -- only a status and a "number theory" tag. There is therefore no
concrete sequence to derive a property from for this problem, and this script
cannot faithfully test problem #843 itself. Rather than fabricate a fake
"Erdos #843 sequence", this script honestly substitutes the smallest genuine,
finite, computable number-theory property consistent with the problem's own
tag: primality over a small finite range. This is a stand-in chosen for having
real mathematical content and a real quantum algorithm (Grover search), NOT a
literal value taken from problem #843's statement, since none exists in the
source data.

Classical property tested: "which n in {0, 1, ..., 15} (4 qubits) are prime?"
Ground truth is computed here from first principles (trial division), not
copied from any table:

    primes in [0, 15] = [2, 3, 5, 7, 11, 13]

Quantum approach: Grover's search algorithm over a 4-qubit register (N = 16
basis states). The oracle phase-flips exactly the computational basis states
whose integer value is prime (as computed classically above, and only those).
After the optimal number of Grover iterations for 6 marked items out of 16,
measuring the register should return a prime value with high probability.
We run the ideal AerSimulator, take the most frequently measured value, and
PASS iff it is a member of the classically-computed prime set AND the total
measured probability mass on prime outcomes exceeds the total mass on
non-prime outcomes (i.e. Grover amplification genuinely worked, not just a
lucky single shot).

Dependencies: qiskit, qiskit_aer, numpy only (already installed).
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed from first principles, no lookup table)
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16
CLASSICAL_PRIMES = [n for n in range(N) if is_prime(n)]
print(f"Classical primes in [0, {N - 1}]: {CLASSICAL_PRIMES}")
assert CLASSICAL_PRIMES == [2, 3, 5, 7, 11, 13], "sanity check on trial division failed"


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the prime basis states
# ---------------------------------------------------------------------------

def oracle_for_marked(marked_values, n_qubits):
    """Phase-flip |x> for each integer x in marked_values, via a multi-controlled Z
    applied to each target bitstring (X-gates to map the target pattern to all-ones,
    MCZ, X-gates back)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")  # MSB..LSB string
        # qubit i (0-indexed, little-endian) corresponds to bit (n_qubits-1-i)
        zero_qubits = [i for i in range(n_qubits) if bits[n_qubits - 1 - i] == "0"]
        for q in zero_qubits:
            qc.x(q)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked_values, n_qubits):
    m = len(marked_values)
    n_total = 2 ** n_qubits
    # optimal number of Grover iterations for m marked items out of n_total
    theta = math.asin(math.sqrt(m / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = oracle_for_marked(marked_values, n_qubits)
    diff = diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diff.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

def run_and_check():
    circuit, iterations = build_grover_circuit(CLASSICAL_PRIMES, N_QUBITS)
    print(f"Grover iterations used: {iterations}")

    backend = AerSimulator()
    transpiled = transpile(circuit, backend)
    shots = 4096
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    # counts keys are bitstrings MSB..LSB as printed by qiskit (creg order),
    # convert to integers
    value_counts = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        value_counts[value] = value_counts.get(value, 0) + c

    prime_mass = sum(value_counts.get(v, 0) for v in CLASSICAL_PRIMES)
    nonprime_mass = shots - prime_mass
    most_likely_value = max(value_counts, key=value_counts.get)

    print(f"Measured value counts (value: count): {sorted(value_counts.items())}")
    print(f"Total shots: {shots}, prime mass: {prime_mass}, non-prime mass: {nonprime_mass}")
    print(f"Most frequently measured value: {most_likely_value} "
          f"(classically prime: {most_likely_value in CLASSICAL_PRIMES})")

    quantum_says_prime = most_likely_value in CLASSICAL_PRIMES
    amplification_worked = prime_mass > nonprime_mass

    passed = quantum_says_prime and amplification_worked
    return passed, most_likely_value, prime_mass, nonprime_mass


if __name__ == "__main__":
    ok, top_value, prime_mass, nonprime_mass = run_and_check()
    if ok:
        print("PASS: Grover search amplified prime basis states "
              f"(prime mass {prime_mass} > non-prime mass {nonprime_mass}), "
              f"and the most likely outcome ({top_value}) is classically prime.")
    else:
        print("FAIL: Grover search did not amplify prime outcomes as expected.")
