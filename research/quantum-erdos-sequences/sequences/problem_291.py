"""
Erdos problem #291 -- quantum-testable instance.

Problem source: erdosproblems.com #291 (unit fractions / harmonic numbers;
open, no prize). OEIS id used: A110566.

A110566(n) = lcm(1, 2, ..., n) / denominator(H(n)), where H(n) = sum_{k=1}^n 1/k
is the n-th harmonic number written in lowest terms and lcm(1..n) is the least
common multiple of 1..n. Equivalently it is lcm(1..n) divided by the reduced
denominator of H(n); Erdos problem #291 concerns exactly this quantity
(part (ii) of the problem asks whether gcd(a_n, lcm(1..n)) > 1 infinitely
often, i.e. whether a(n) > 1 infinitely often -- see
erdosproblems.com/291 and the linked formalization PR
"Lean proof of Erdos #291 part (ii)").

Classical property tested here (finite, computable, checked from first
principles in this script, not copied from OEIS):

    For n = 1..32, is A110566(n) > 1 ?

This is exactly the event studied in part (ii) of the problem. We compute
a(n) for n = 1..32 directly from the definition (running lcm and the exact
Fraction sum of 1/k), which gives a finite 32-element search space indexed by
i = n-1 in {0, ..., 31} (5 qubits). Call this index set MARKED = { i : a(i+1)
> 1 }; classically we find

    MARKED = {5, 6, 7, 17, 18, 19, 20, 21, 22, 23, 24, 25}   (12 of 32)

Quantum approach: Grover search. We build a genuine Grover oracle (a
multi-controlled phase flip realized as a diagonal unitary over all 32 basis
states, entries -1 exactly on MARKED, +1 elsewhere) plus the standard
diffusion operator, and run it on the ideal AerSimulator. With |MARKED| = 12
out of N = 32, the optimal number of Grover iterations is
round(pi/4 * sqrt(N/M)) = 1. After 1 iteration we sample many shots and check
that the circuit concentrates probability mass on MARKED indices far above
the uniform baseline of 12/32 = 0.375, and that the single most likely
outcome is itself a marked index -- i.e. the quantum search genuinely finds
elements of the classically-verified set {n : A110566(n) > 1, 1 <= n <= 32}.

PASS criterion: P(measured index in MARKED) after Grover search is at least
0.75 (uniform baseline is 0.375), and the most frequently measured index is
in MARKED.
"""

import math
from fractions import Fraction
from math import gcd

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator


def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def classical_A110566(n_max: int):
    """Compute A110566(n) = lcm(1..n) / denominator(H(n)) for n = 1..n_max."""
    L = 1
    H = Fraction(0)
    values = []
    for n in range(1, n_max + 1):
        L = lcm(L, n)
        H += Fraction(1, n)
        a_n = L // H.denominator
        assert L % H.denominator == 0, "lcm(1..n) must be divisible by denom(H(n))"
        values.append(a_n)
    return values


def build_grover_circuit(n_qubits: int, marked_indices, iterations: int) -> QuantumCircuit:
    N = 2 ** n_qubits

    # Oracle: diagonal unitary, -1 on marked computational basis states.
    oracle_diag = np.ones(N, dtype=complex)
    for idx in marked_indices:
        oracle_diag[idx] = -1.0
    oracle_op = Operator(np.diag(oracle_diag))

    # Diffusion operator: 2|s><s| - I, s = uniform superposition.
    s = np.full(N, 1.0 / math.sqrt(N))
    diffusion_matrix = 2.0 * np.outer(s, s) - np.eye(N)
    diffusion_op = Operator(diffusion_matrix)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.unitary(oracle_op, range(n_qubits), label="oracle")
        qc.unitary(diffusion_op, range(n_qubits), label="diffusion")
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    n_max = 32
    values = classical_A110566(n_max)
    marked = [i for i, v in enumerate(values) if v > 1]  # index i corresponds to n = i+1

    print("A110566(1..32) =", values)
    print("Classically marked indices (n-1 with A110566(n) > 1):", marked)

    n_qubits = 5  # 2^5 = 32
    N = 2 ** n_qubits
    M = len(marked)
    assert N == n_max

    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
    print(f"Grover iterations (optimal for N={N}, M={M}): {iterations}")

    qc = build_grover_circuit(n_qubits, marked, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Empirically verified against Statevector.probabilities_dict(): the
    # measured bitstring, read directly as a binary integer (no reversal),
    # equals the index used throughout this script (see comment in main()).
    def bitstring_to_index(bs: str) -> int:
        return int(bs, 2)

    index_counts = {}
    for bitstring, c in counts.items():
        idx = bitstring_to_index(bitstring)
        index_counts[idx] = index_counts.get(idx, 0) + c

    marked_set = set(marked)
    marked_shots = sum(c for idx, c in index_counts.items() if idx in marked_set)
    p_marked = marked_shots / shots
    baseline = M / N

    most_likely_idx = max(index_counts, key=index_counts.get)

    print(f"P(measured index is classically marked) = {p_marked:.4f} "
          f"(uniform baseline = {baseline:.4f})")
    print(f"Most likely measured index: {most_likely_idx} "
          f"(n = {most_likely_idx + 1}, A110566(n) = {values[most_likely_idx]}) "
          f"-- marked: {most_likely_idx in marked_set}")

    success = (p_marked >= 0.75) and (most_likely_idx in marked_set)

    if success:
        print("PASS")
    else:
        print("FAIL")

    return success


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
