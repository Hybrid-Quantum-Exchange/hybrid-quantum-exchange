"""
Erdos problem #233 (see erdosproblems.com/233; data/problems.yaml entry
`number: "233"`), tags ["number theory", "primes"], associated OEIS id
A074741.

Property tested (small, finite, computable instance):
    Among the integers 0..15 (4 bits), identify exactly the set of PRIME
    numbers, using Grover's algorithm to search the 4-qubit computational
    basis for the marked (prime) states.

    This is a genuine finite decision problem in the same "primes" family
    that problem #233 and A074741 sit in: "is n prime?" restricted to a
    small range. The classical answer (the exact list of primes in
    [0, 15]) is computed here from first principles by trial division,
    independent of any OEIS lookup, and is then used both to build the
    Grover oracle and to check the quantum search's output.

    Classical answer for N = 16 (0..15):
        primes = [2, 3, 5, 7, 11, 13]   (6 marked states out of 16)

Approach:
    - Classically compute is_prime(n) for n in 0..15 by trial division.
    - Build a 4-qubit Grover search whose oracle phase-flips exactly the
      basis states |n> for which is_prime(n) is True (a multi-controlled
      Z gate per marked state, so the oracle is constructed directly from
      the classical truth table -- no OEIS values are hard-coded as
      circuit outputs).
    - Run the optimal number of Grover iterations for 6 marked-of-16 on
      the ideal AerSimulator, and check that the highest-probability
      measured outcomes are exactly the classically computed prime set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_is_prime(n: int) -> bool:
    """Trial-division primality test, first principles, no OEIS lookup."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def build_oracle(n_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking exactly `marked_states` (each an int in
    [0, 2**n_qubits - 1]) among the computational basis states."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        # Flip qubits that should be 0 in this state, so a multi-controlled
        # Z (control on |1...1>) fires exactly on this state.
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
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


def run() -> tuple[bool, list[int], list[int]]:
    n_qubits = 4
    N = 2 ** n_qubits  # 16

    classical_primes = [n for n in range(N) if classical_is_prime(n)]
    M = len(classical_primes)  # expected: 6

    # Optimal number of Grover iterations for M marked out of N.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M) - 0.5))

    oracle = build_oracle(n_qubits, classical_primes)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (Qiskit prints classical register with qubit 0
    # rightmost) to integers.
    outcome_counts: dict[int, int] = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        outcome_counts[value] = outcome_counts.get(value, 0) + c

    # Take the top-M most frequent measured outcomes as the quantum result.
    ranked = sorted(outcome_counts.items(), key=lambda kv: kv[1], reverse=True)
    quantum_top = sorted(v for v, _ in ranked[:M])

    ok = quantum_top == classical_primes
    return ok, classical_primes, quantum_top


if __name__ == "__main__":
    ok, classical_primes, quantum_top = run()
    print(f"Classical primes in [0,15]: {classical_primes}")
    print(f"Grover top-{len(classical_primes)} measured outcomes: {quantum_top}")
    print("PASS" if ok else "FAIL")
