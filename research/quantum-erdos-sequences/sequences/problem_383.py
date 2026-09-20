"""
Erdos problem #383 — quantum-testable sequence lane.

Source record: /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: '383'" (open, unformalized, tags: ["number theory"],
oeis: ["N/A"]).

LIMITATION (reported honestly, per instructions): Erdos problem #383 has
no associated OEIS sequence id in the source data (oeis: ["N/A"]). There
is therefore no specific integer sequence from this problem to build a
quantum-testable membership/search property around, and fabricating one
would misrepresent the problem. This script is a best-honest-attempt
fallback: since the problem's only real metadata is the tag "number
theory", it builds a genuine, small, finite, classically-checkable
number-theoretic search problem (primality search over a small range)
and verifies it with a real Grover's algorithm circuit on AerSimulator.
This choice is NOT derived from problem 383's actual open conjecture —
it is a placeholder quantum-computable task chosen because #383 itself
supplies none. verified_against_classical therefore certifies that the
Grover circuit correctly finds the classically-computed prime in the
search space, not that it says anything about Erdos problem #383's
actual (still-open) mathematical content.

Classical property tested:
    N = 8 (3-qubit search space, indices 0..7).
    Marked property: n is prime (classical trial division), among
    {0,...,7}. The primes in that range are {2, 3, 5, 7}.
    We pick a single specific target, n0 = 5, and build a Grover oracle
    that marks exactly |5> = |101>. The classical answer is verified by
    direct trial-division primality check computed in this script.

Circuit: standard 3-qubit Grover search (single marked state), diffuser,
optimal iteration count round(pi/4 * sqrt(N/M)) with N=8, M=1 -> 2
iterations, run on qiskit_aer's AerSimulator with 2048 shots. PASS if the
most frequently measured basis state equals the classically-marked state.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import sys
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_max: int):
    return [n for n in range(n_max) if is_prime(n)]


def build_oracle(n_qubits: int, target: int) -> QuantumCircuit:
    """Multi-controlled Z oracle marking the computational basis state
    |target> (target given as an integer, little-endian qubit order)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(target, f"0{n_qubits}b")[::-1]  # little-endian
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i in zero_positions:
        qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_search(n_qubits: int, target: int, iterations: int, shots: int = 2048):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, target)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    N_QUBITS = 3
    N = 2 ** N_QUBITS  # 8

    primes = classical_primes(N)
    assert primes == [2, 3, 5, 7], f"unexpected classical primes list: {primes}"

    target = 5
    assert target in primes, "chosen target must be classically prime"

    iterations = round((math.pi / 4) * math.sqrt(N / 1))  # M = 1 marked state
    counts = grover_search(N_QUBITS, target, iterations)

    # Most frequent measured bitstring -> integer (qiskit bit order is
    # big-endian string, little-endian qubit index; convert consistently).
    best_bitstring = max(counts, key=counts.get)
    measured_value = int(best_bitstring, 2)

    print("Erdos problem #383 quantum lane")
    print("Source: oeis=['N/A'] (no sequence available); tags=['number theory']")
    print(f"Classical primes in range [0,{N}): {primes}")
    print(f"Target marked state (classical): {target}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")
    print(f"Most frequent measured value: {measured_value}")

    success_prob = counts.get(best_bitstring, 0) / sum(counts.values())
    print(f"Success probability (top outcome): {success_prob:.3f}")

    ok = (measured_value == target) and (success_prob > 0.5)

    if ok:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
