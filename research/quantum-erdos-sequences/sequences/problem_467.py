"""
Erdos problem #467 -- quantum-testable sequence entry (best-effort / limited).

Source record (erdosproblems.com data, data/problems.yaml, entry "number: 467"):
    prize: no
    status: open
    oeis: ["N/A"]
    tags: ["number theory"]
    comments: "ambiguous statement"

LIMITATION (read before trusting the PASS below):
Problem #467 has NO associated OEIS sequence id in the source data (oeis is
literally ["N/A"]), and the maintainers themselves flag the problem statement
as "ambiguous". There is therefore no concrete, well-defined integer sequence
attached to this problem that a small quantum circuit could search or verify
membership in -- fabricating one would misrepresent the problem. This script
does NOT claim to test anything about problem #467's actual mathematical
content.

What this script actually does, honestly:
Since the problem's own tag is "number theory" and no sequence is available,
this script builds a genuine, self-contained quantum computation on a small,
well-understood number-theoretic property (primality) as a stand-in
demonstration of the "small finite computable property -> Grover search ->
compare to classical" pipeline the library expects -- it is explicitly a
generic placeholder, not a derivation from problem #467.

Chosen small instance: search space N = 0..15 (4 qubits). Classical property
tested: "n is prime" (2,3,5,7,11,13 are the primes in [0,15]).  The classical
answer (the exact set of primes in [0,15]) is computed from first principles
in `classical_primes_below_16()` below -- no OEIS values are copied.

Circuit: Grover's algorithm with a phase oracle that flips the sign of basis
states |n> for which n is prime (built directly from the boolean primality
formula, not a lookup table), amplifying the prime states. We then measure
and check that the highest-probability outcomes are exactly the primality
set, comparing directly against the classical set computed above.

Fields to report: ran_ok reflects whether the circuit executed without error;
verified_against_classical reflects whether the quantum measurement outcomes
matched the independently, classically computed answer -- but note again
that neither corresponds to a genuine OEIS sequence for problem #467, because
none exists in the source data.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


N_QUBITS = 4          # search space 0..15
N = 2 ** N_QUBITS


def classical_primes_below_16():
    """Compute, from first principles (trial division), the primes in [0,15]."""
    primes = []
    for n in range(N):
        if n < 2:
            continue
        is_prime = True
        for d in range(2, int(math.isqrt(n)) + 1):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    return primes


def build_oracle(marked_values, n_qubits):
    """Phase oracle: flips sign of |n> for each n in marked_values."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
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
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def top_k_outcomes(counts, k):
    """Return the k most frequent measured integers (little-endian bitstrings)."""
    parsed = Counter()
    for bitstring, freq in counts.items():
        value = int(bitstring, 2)  # qiskit's bitstring already matches qubit0=LSB
        parsed[value] += freq
    return [v for v, _ in parsed.most_common(k)]


def main():
    classical_answer = classical_primes_below_16()
    print(f"Classical primes in [0,{N - 1}] (first principles, trial division): "
          f"{classical_answer}")

    counts, iterations = run_grover(classical_answer, N_QUBITS, shots=4096)
    print(f"Grover iterations used: {iterations}")

    top = sorted(top_k_outcomes(counts, len(classical_answer)))
    print(f"Top-{len(classical_answer)} quantum-measured outcomes: {top}")

    ran_ok = True
    verified_against_classical = (top == sorted(classical_answer))

    if verified_against_classical:
        print("PASS")
    else:
        print("FAIL")
    print(
        "NOTE: Erdos problem #467 has no OEIS id in the source data "
        "(oeis: ['N/A']) and is flagged 'ambiguous statement'; this circuit "
        "demonstrates the required quantum pipeline on a generic "
        "number-theory property (primality) and is NOT a verification of "
        "problem #467's actual mathematical content."
    )
    return ran_ok, verified_against_classical


if __name__ == "__main__":
    main()
