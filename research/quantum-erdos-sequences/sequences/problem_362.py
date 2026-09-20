"""
Erdos problem #362 -- quantum-testable lane.

Source metadata (data/problems.yaml, erdosproblems.com dataset, entry
"number: '362'"):
    prize: no
    informal_status: proved (2025-08-31), formal_status: Lean (2026-08-24)
    oeis: ["possible"]
    tags: ["number theory"]

HONEST LIMITATION: the dataset's `oeis` field for problem 362 is the literal
string "possible", not a real OEIS sequence id (compare to neighbouring
entries such as #363, whose field is "N/A" -- these are placeholder/status
strings the erdosproblems.com scraper left in the `oeis` column, not A-numbers).
So there is no genuine OEIS sequence attached to this problem to pull a term
or membership test from, and no problem statement/proof file for #362 is
present in the read-only clone either -- only the tag "number theory" survives.

Given that, this script does NOT fabricate an OEIS value. Instead, since the
only real signal we have is the tag "number theory", it builds a genuine,
independently-checkable finite number-theoretic decision problem in the same
family many Erdos number-theory problems live in (prime distribution among
small integers) and solves it with a real Grover search circuit on
AerSimulator:

    Property tested: "n is prime", for n in the search space {0, 1, ..., 15}
    (4 qubits). The classical answer -- the exact subset of primes in that
    range -- is computed here in pure Python via trial division (first
    principles, no lookup table), independently of any OEIS data. Grover's
    algorithm is then run to amplify exactly those marked (prime) basis
    states, and the script checks that the states Grover boosts (the top
    len(primes) measurement outcomes by frequency) are exactly the classical
    prime set.

This is offered as the closest honest, genuinely quantum, genuinely
finite/computable analogue available for this entry -- not as a literal test
of "the OEIS sequence for problem 362", since no such sequence exists in the
source data.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator

N_QUBITS = 4
N = 2 ** N_QUBITS  # search space {0, ..., 15}


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(limit: int):
    return sorted(n for n in range(limit) if is_prime(n))


def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle: |x> -> -|x> for x in marked, identity otherwise."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")
        # Flip qubits where the target bit is 0, so the all-ones pattern
        # corresponds to |m>, apply a controlled-Z, then flip back.
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
            qc.append(mcz, list(range(n_qubits)))
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
        qc.append(mcz, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits: int, marked: list[int], iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    primes = classical_primes(N)  # classical ground truth, computed here
    k = len(primes)
    print(f"Search space: n in [0, {N - 1}], {N_QUBITS} qubits")
    print(f"Classical primes (trial division): {primes}  (count={k})")

    # Optimal Grover iteration count for k marked items out of N.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / k)))
    print(f"Grover iterations: {iterations}")

    qc = build_grover_circuit(N_QUBITS, primes, iterations)

    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    shots = 4096
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (qiskit little-endian, c[0] is qubit 0 = LSB) to ints.
    freq = {}
    for bitstring, c in counts.items():
        val = int(bitstring, 2)
        freq[val] = freq.get(val, 0) + c

    ranked = sorted(freq.items(), key=lambda kv: -kv[1])
    top_k = sorted(v for v, _ in ranked[:k])

    prime_mass = sum(c for v, c in freq.items() if v in primes)
    total_mass = sum(freq.values())
    prime_fraction = prime_mass / total_mass

    print(f"Top-{k} most frequent measured values: {top_k}")
    print(f"Fraction of shots landing on a prime: {prime_fraction:.3f}")

    quantum_result_matches_classical = (top_k == primes) and (prime_fraction > 0.8)

    if quantum_result_matches_classical:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
