"""
Erdos problem #463 (erdosproblems.com / manman4/erdosproblems data/problems.yaml)

Source metadata for problem 463 in data/problems.yaml:
    number: "463"
    status: open (as of 2025-08-31)
    oeis: ["possible"]
    tags: ["number theory", "primes"]

LIMITATION (read before trusting the "OEIS id used" below): the yaml entry's
`oeis` field is literally the string "possible" -- not an OEIS sequence id.
There is no A-number attached to problem 463 in the source data, so there is
no concrete integer sequence to target with a quantum circuit for this
specific open problem. Rather than fabricate an OEIS id or copy a value with
no real backing, this script honestly falls back to the problem's *tags*
("number theory", "primes") and builds a genuine, independently-verifiable
finite computable property from those: primality testing over a small
integer range, searched with Grover's algorithm.

Chosen property (classically defined and checked from first principles,
independent of any OEIS lookup):

    Over the 4-bit universe U = {0, 1, ..., 15}, let S = { n in U : n is
    prime }. We classically compute S by trial division (no library
    primality calls, no hardcoded prime list). We then build a Grover
    search whose oracle marks exactly the basis states |n> with n in S,
    and check that Grover amplification concentrates measurement
    probability on S.

Classical answer for U = {0,...,15} (computed by trial division below):
    S = {2, 3, 5, 7, 11, 13}   (6 primes out of 16 residues)

Quantum method: 4-qubit Grover search (index register n in [0,16)) with a
phase oracle built directly from the classical primality predicate (a
multi-controlled-Z gate is applied for each n in S, flipping the sign of
exactly that basis state), followed by the standard diffusion operator,
repeated the optimal number of iterations for |S|=6 out of N=16. The
circuit is run on the ideal AerSimulator with many shots; PASS requires
that the measured-outcome set with non-negligible probability mass equals
the classically-computed prime set S (i.e. the circuit's amplified
outcomes are exactly the primes, and non-primes are suppressed).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime_trial_division(n: int) -> bool:
    """Classical primality test by trial division, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_prime_set(n_bits: int):
    universe = range(2 ** n_bits)
    return sorted(n for n in universe if is_prime_trial_division(n))


def build_oracle(n_qubits: int, marked_values):
    """Phase oracle: flips the sign of each marked basis state |v>.

    For each marked integer v, apply X gates to the qubits where the bit of
    v is 0, then a multi-controlled Z (via H + MCX + H on the last qubit),
    then undo the X gates. This is a standard, direct construction (no
    external oracle libraries).
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    for v in marked_values:
        bits = [(v >> i) & 1 for i in range(n_qubits)]
        flip_qubits = [i for i, b in enumerate(bits) if b == 0]
        for q in flip_qubits:
            qc.x(q)
        # multi-controlled Z on all n_qubits (phase flip of |111...1>)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for q in flip_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits: int, marked_values, iterations: int):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    n_qubits = 4
    N = 2 ** n_qubits  # 16

    # --- Classical ground truth, derived from first principles ---
    primes = classical_prime_set(n_qubits)
    M = len(primes)
    print(f"Universe size N={N}, classically computed primes S={primes} (|S|={M})")

    if M == 0 or M == N:
        print("FAIL: degenerate marked set, cannot run meaningful Grover search")
        sys.exit(1)

    # Optimal number of Grover iterations for this N, M
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    # For this small N, exhaustively pick the iteration count (within a
    # generous range) that maximizes true theoretical success probability,
    # since the standard formula's rounding can land on a suboptimal step.
    best_iter, best_prob = iterations, -1.0
    for k in range(1, 6):
        p = math.sin((2 * k + 1) * theta) ** 2
        if p > best_prob:
            best_prob, best_iter = p, k
    iterations = best_iter
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(n_qubits, primes, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit ordering: classical register bit string is c[n-1]...c[0];
    # convert each measured bitstring back to the integer it encodes.
    outcome_counts = {}
    for bitstring, c in counts.items():
        n = int(bitstring, 2)
        outcome_counts[n] = outcome_counts.get(n, 0) + c

    total = sum(outcome_counts.values())
    assert total == shots

    # The set of outcomes carrying non-negligible probability mass should be
    # exactly the classically-computed prime set.
    threshold = 0.5 / N  # well below uniform (1/N), well above simulator noise floor
    amplified = sorted(
        n for n, c in outcome_counts.items() if (c / total) >= threshold
    )

    prime_mass = sum(outcome_counts.get(n, 0) for n in primes) / total
    print(f"Probability mass on classically-prime outcomes: {prime_mass:.4f}")
    print(f"Outcomes with non-negligible probability (measured): {amplified}")
    print(f"Classically-computed prime set (expected):           {primes}")

    verified = (amplified == primes) and (prime_mass > 0.9)

    if verified:
        print("PASS: Grover search amplified exactly the classically-computed "
              "prime residues in {0,...,15}, matching the classical answer.")
        sys.exit(0)
    else:
        print("FAIL: quantum result did not match the classical prime set.")
        sys.exit(1)


if __name__ == "__main__":
    main()
