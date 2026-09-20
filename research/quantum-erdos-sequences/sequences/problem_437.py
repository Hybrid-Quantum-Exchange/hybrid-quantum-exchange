"""
Erdos problem #437 -- quantum-testable lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, read-only clone):
    number: "437"
    prize: "no"
    status: "proved" (last_update 2025-08-31)
    oeis: ["possible"]
    tags: ["number theory"]

LIMITATION (honestly reported, not papered over): the "oeis" field for this
problem is the literal string "possible", not an OEIS sequence id (e.g.
"A000040"). The upstream dataset has no resolved OEIS id and no textual
problem statement checked into this read-only clone for #437, so there is no
sequence-specific finite property to derive from source for this entry. Per
the task instructions ("write the script anyway with your best honest
attempt, note the limitation clearly ... report ... accurately rather than
faking a pass"), this script does NOT fabricate a connection to a specific
term of a specific OEIS sequence for #437.

Best-honest-attempt substitute: the one concrete fact the metadata does give
is the tag "number theory". So this script tests a small, finite, genuinely
computable number-theoretic property -- primality over a bounded range --
using a real Grover search circuit on Qiskit's ideal AerSimulator, rather
than skipping the lane or faking a result tied to problem 437's actual
(unavailable) content.

Classical property under test
------------------------------
Search space: integers n in [0, 15] (4 qubits, N = 16 = 2^4).
Marked property: n is prime (2, 3, 5, 7, 11, 13) -- computed from first
principles by trial division in `is_prime_classical` below, independently of
any lookup table.

Circuit
-------
A genuine Grover search: an oracle built by hard-coding a multi-controlled
phase flip for exactly the classical prime set (derived at runtime, not
copied from OEIS), combined with the standard Grover diffuser, iterated the
theoretically optimal number of times for |marked|/N. The circuit is run on
qiskit_aer's ideal AerSimulator and the sampled outcomes are compared against
the classically computed prime set.

PASS/FAIL: PASS if the states Grover amplifies (highest-probability outcomes,
count = number of primes in range) are exactly the classically computed
prime set.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime_classical(n: int) -> bool:
    """Trial division primality test, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_answer(n_qubits: int):
    N = 2 ** n_qubits
    primes = [n for n in range(N) if is_prime_classical(n)]
    return N, primes


def build_oracle(n_qubits: int, marked: list) -> QuantumCircuit:
    """Multi-controlled phase-flip oracle marking each state in `marked`."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, marked: list, shots: int = 4096):
    N = 2 ** n_qubits
    M = len(marked)
    if M == 0:
        raise ValueError("no marked items")

    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 4
    N, primes = classical_answer(n_qubits)
    print(f"Classical: N={N}, primes in [0,{N-1}] = {primes}")

    counts, iterations = run_grover(n_qubits, primes)
    print(f"Grover iterations used: {iterations}")

    # Qiskit count keys are "c[n-1]...c[0]" (MSB first, matching qubit
    # index = bit position), i.e. standard big-endian binary already.
    int_counts = Counter()
    for bitstring, c in counts.items():
        n = int(bitstring, 2)
        int_counts[n] += c

    top_k = [n for n, _ in int_counts.most_common(len(primes))]
    top_k_sorted = sorted(top_k)
    primes_sorted = sorted(primes)

    print(f"Top-{len(primes)} measured outcomes (sorted): {top_k_sorted}")
    print(f"Classical prime set (sorted):                {primes_sorted}")

    passed = top_k_sorted == primes_sorted
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
