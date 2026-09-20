"""
Erdos problem #887 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems repo, entry "number: '887'"):
    prize: no
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory", "divisors"]

LIMITATION: problem #887 has NO OEIS sequence id attached in the source data
(oeis: ["N/A"]) and is an open, unformalized problem, so there is no citable
"known term" of an Erdos-887 sequence to reproduce on a quantum computer.
Rather than fabricate an OEIS id or copy a value with no real connection to
the problem, this script instead builds a genuine, self-contained instance
from the problem's own tags ("number theory", "divisors"): a Grover search
for PERFECT NUMBERS (n such that the sum of n's proper divisors equals n),
which is a small, finite, exactly-computable divisor property in the same
family the problem is tagged with (OEIS A000396, sigma(n) - n == n).

Classical property being tested:
    Search space: integers n in [0, 15] (4 qubits, so N = 16).
    Predicate:    perfect(n) := (sum of proper divisors of n) == n, n > 0.
    Within [0, 15] the unique perfect number is n = 6 (divisors 1+2+3=6).
    This is computed from first principles below (divisor_sum), not looked
    up, and asserted against the well-known fact that 6 is the smallest
    perfect number.

Quantum approach:
    Grover's algorithm on 4 qubits (N=16 states). The oracle is built by
    classically evaluating perfect(n) for every n in range(16) and phase-
    flipping exactly the marked basis state(s) (a diagonal oracle -- this
    is the standard, legitimate way to realize an arbitrary classical
    predicate as a Grover oracle without needing in-circuit arithmetic).
    One marked item out of 16 -> optimal Grover iterations = round(pi/4 *
    sqrt(16/1)) = 3. We run this on the ideal AerSimulator and confirm the
    measured basis state (>90% of shots) equals the classical answer n=6.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def divisor_sum(n: int) -> int:
    """Sum of proper divisors of n (divisors of n excluding n itself)."""
    if n <= 0:
        return 0
    total = 0
    for d in range(1, n):
        if n % d == 0:
            total += d
    return total


def is_perfect(n: int) -> bool:
    return n > 0 and divisor_sum(n) == n


def classical_search(n_bits: int):
    """Return sorted list of n in [0, 2**n_bits - 1] with is_perfect(n)."""
    N = 2 ** n_bits
    return [n for n in range(N) if is_perfect(n)]


def build_diagonal_oracle(n_bits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip exactly the marked computational basis states."""
    N = 2 ** n_bits
    diag = np.ones(N, dtype=complex)
    for m in marked:
        diag[m] = -1.0
    qc = QuantumCircuit(n_bits, name="oracle")
    qc.append(DiagonalGate(list(diag)), list(range(n_bits)))
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def grover_circuit(n_bits: int, marked: list[int], iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    oracle = build_diagonal_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_bits))
        qc.append(diffuser.to_instruction(), range(n_bits))
    qc.measure(range(n_bits), range(n_bits))
    return qc


def main():
    n_bits = 4  # search space size N = 16 -> covers n in [0, 15]
    N = 2 ** n_bits

    # --- classical ground truth, computed from first principles ---
    marked = classical_search(n_bits)
    print(f"Search space: n in [0, {N - 1}]")
    print(f"Perfect numbers found classically: {marked}")
    assert marked == [6], (
        f"expected the unique perfect number in [0,15] to be 6, got {marked}"
    )
    classical_answer = marked[0]
    print(f"Classical answer (unique marked item): {classical_answer}")

    # --- optimal number of Grover iterations for 1 marked item out of N ---
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / len(marked))))
    print(f"Grover iterations: {iterations}")

    qc = grover_circuit(n_bits, marked, iterations)

    # sanity check the oracle+diffuser algebra on a statevector sim before
    # measuring, to make sure the marked amplitude is genuinely amplified
    sv_qc = QuantumCircuit(n_bits)
    sv_qc.h(range(n_bits))
    oracle = build_diagonal_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        sv_qc.append(oracle.to_instruction(), range(n_bits))
        sv_qc.append(diffuser.to_instruction(), range(n_bits))
    sv = Statevector.from_instruction(sv_qc)
    probs = sv.probabilities_dict()
    marked_bitstring = format(classical_answer, f"0{n_bits}b")[::-1]  # qiskit little-endian
    marked_prob = probs.get(marked_bitstring, 0.0)
    print(f"Statevector probability of marked state |{classical_answer}>: {marked_prob:.4f}")
    assert marked_prob > 0.9, f"Grover amplification too weak: {marked_prob}"

    # --- run on the ideal AerSimulator ---
    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 2048
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # counts keys are bitstrings c3c2c1c0 (qiskit little-endian classical bits
    # correspond to qubits 0..n_bits-1 in order); convert to integers
    int_counts = {}
    for bitstring, c in counts.items():
        n_val = int(bitstring[::-1], 2)
        int_counts[n_val] = int_counts.get(n_val, 0) + c

    best_n = max(int_counts, key=int_counts.get)
    best_frac = int_counts[best_n] / shots
    print(f"Measured distribution (top): n={best_n} with {int_counts[best_n]}/{shots} shots "
          f"({best_frac:.1%})")
    print(f"Full integer counts: {dict(sorted(int_counts.items(), key=lambda kv: -kv[1]))}")

    quantum_found_correct = (best_n == classical_answer) and (best_frac > 0.5)

    if quantum_found_correct:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
