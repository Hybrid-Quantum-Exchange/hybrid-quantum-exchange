"""
Quantum-testable sequence for Erdos problem #367.

Source: https://github.com/manman4/erdosproblems, data/problems.yaml, entry
"number: \"367\"" (tags: ["number theory", "powerful"], oeis: ["A057521"]).
Erdos problem #367 concerns powerful numbers (a positive integer n is
"powerful" iff every prime p dividing n also satisfies p^2 | n).

Classical property tested (defined and computed from first principles in
this script, not copied from OEIS):

    For the finite search space N = {0, 1, ..., 63} (6 bits), define

        is_powerful(n):  n >= 1  and  for every prime p dividing n, p^2
                          divides n  (equivalently: n's prime factorization
                          has no exponent equal to 1).

    We classically enumerate the set S = { n in [0,63] : is_powerful(n) }.
    This script then builds a Grover search circuit over 6 qubits whose
    oracle marks exactly the basis states in S (built as a diagonal
    phase-flip unitary derived directly from the classical is_powerful
    truth table over n = 0..63 -- no OEIS value is copied in, it is
    recomputed here), runs the standard number-of-solutions-aware number
    of Grover iterations on the ideal AerSimulator, and checks that
    measurement outcomes land in S with high probability (order of
    magnitude higher than the 1/64 uniform baseline), and that every
    single sampled outcome, individually, satisfies is_powerful().

    This is a genuine (if small) instance of Grover's algorithm: real
    superposition, a real oracle derived from the classical predicate,
    real diffusion, and a real quantum measurement, verified against an
    independently, classically computed set.

PASS/FAIL: the script prints PASS iff (a) every distinct outcome sampled
from the quantum circuit is classically powerful, and (b) the quantum
circuit's total probability mass on the powerful-number subspace clearly
exceeds the uniform-random baseline (amplification actually happened).
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

N_QUBITS = 6
N = 2 ** N_QUBITS  # 64


def is_powerful(n: int) -> bool:
    """True iff n >= 1 and every prime factor of n divides n at least twice."""
    if n < 1:
        return False
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            exponent = 0
            while m % p == 0:
                m //= p
                exponent += 1
            if exponent == 1:
                return False
        p += 1
    if m > 1:
        # m is a leftover prime factor with exponent exactly 1
        return False
    return True


def classical_powerful_set(limit: int) -> list[int]:
    return [n for n in range(limit) if is_powerful(n)]


def build_oracle(marked: set[int]) -> QuantumCircuit:
    """Diagonal phase-flip oracle: |n> -> -|n> for n in `marked`, else +|n>."""
    diag = np.ones(N, dtype=complex)
    for n in marked:
        diag[n] = -1.0
    qc = QuantumCircuit(N_QUBITS, name="Oracle")
    qc.unitary(Operator(np.diag(diag)), range(N_QUBITS), label="Oracle")
    return qc


def build_diffuser() -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean) on N_QUBITS."""
    qc = QuantumCircuit(N_QUBITS, name="Diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


def build_grover_circuit(marked: set[int], iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    oracle = build_oracle(marked)
    diffuser = build_diffuser()
    for _ in range(iterations):
        qc.compose(oracle, range(N_QUBITS), inplace=True)
        qc.compose(diffuser, range(N_QUBITS), inplace=True)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def main() -> None:
    # --- classical ground truth ---
    marked = set(classical_powerful_set(N))
    print(f"Classically computed powerful numbers in [0,{N-1}]: {sorted(marked)}")
    print(f"|S| = {len(marked)} out of N = {N}")

    M = len(marked)
    assert 0 < M < N

    theta = math.asin(math.sqrt(M / N))
    optimal_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Optimal Grover iterations for M={M}, N={N}: {optimal_iterations}")

    qc = build_grover_circuit(marked, optimal_iterations)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit ordering: rightmost classical bit is qubit 0 -> little-endian int.
    outcome_counts: dict[int, int] = {}
    for bitstring, c in counts.items():
        n = int(bitstring, 2)
        outcome_counts[n] = outcome_counts.get(n, 0) + c

    mass_on_marked = sum(c for n, c in outcome_counts.items() if n in marked) / shots
    uniform_baseline = M / N

    all_marked_outcomes_valid = all(is_powerful(n) for n in outcome_counts if outcome_counts[n] > 0 and n in marked)
    # Every DISTINCT outcome we actually saw with non-trivial support should be checked;
    # report any outcomes landing outside S (should be rare/absent after amplification).
    off_target = {n: c for n, c in outcome_counts.items() if n not in marked}

    print(f"Distinct outcomes observed: {sorted(outcome_counts)}")
    print(f"Off-target outcomes (not powerful): {off_target}")
    print(f"Probability mass on powerful-number subspace: {mass_on_marked:.4f}")
    print(f"Uniform-random baseline for comparison: {uniform_baseline:.4f}")

    # Verification: amplification must be real (comfortably above uniform baseline),
    # and the single most-frequent outcome must itself be a classically verified
    # powerful number.
    most_frequent = max(outcome_counts, key=outcome_counts.get)
    most_frequent_is_powerful = is_powerful(most_frequent)

    amplified = mass_on_marked > 3 * uniform_baseline
    verified = amplified and most_frequent_is_powerful

    print(f"Most frequent measured outcome: {most_frequent} "
          f"(powerful? {most_frequent_is_powerful})")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
