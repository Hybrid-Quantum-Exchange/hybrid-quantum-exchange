"""
Erdos problem #381 (erdosproblems.com), OEIS A002182: highly composite numbers.

A002182(n) is the sequence of "highly composite numbers": positive integers m
such that d(m) > d(k) for every k < m, where d(k) is the number-of-divisors
function. The first terms are 1, 2, 4, 6, 12, 24, 36, 48, 60, ...

Property tested here (finite, computable): within the small search space
N = {1, 2, ..., 31} (5 qubits), which integers are highly composite numbers?
The classical answer is computed from first principles in this script (no
OEIS lookup of a literal value): for each n in [1, 31], compute the divisor
count d(n) directly by trial division, then mark n as highly composite iff
d(n) is a strict new maximum of d(1..n).

Quantum approach: Grover's search. A diagonal phase oracle (built as an
explicit unitary from the classically-computed marked set) flips the phase
of every basis state |n> whose n is highly composite. We then run the
standard number of Grover iterations for this space/solution-count and check
that measurement puts (almost) all probability mass on the marked states.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator


def divisor_count(n: int) -> int:
    """Number of positive divisors of n, computed by trial division."""
    if n <= 0:
        raise ValueError("n must be positive")
    count = 0
    for k in range(1, n + 1):
        if n % k == 0:
            count += 1
    return count


def highly_composite_numbers(limit: int):
    """Classically compute which integers in [1, limit] are highly composite.

    m is highly composite iff d(m) > d(k) for all k < m (strict new max of
    the divisor-count function as we scan upward from 1).
    """
    marked = []
    best = -1
    for n in range(1, limit + 1):
        d = divisor_count(n)
        if d > best:
            best = d
            marked.append(n)
    return marked


def main():
    n_qubits = 5
    N = 2 ** n_qubits  # 32 states, indices 0..31
    limit = N - 1  # search over n = 1..31 (index 0 is unused/never marked)

    marked = highly_composite_numbers(limit)
    print(f"Search space: n = 1..{limit} ({N} basis states, {n_qubits} qubits)")
    print(f"Classically computed highly composite numbers (A002182) in range: {marked}")

    # Build the diagonal oracle: -1 phase on every marked basis state index.
    diag = np.ones(N, dtype=complex)
    for m in marked:
        diag[m] = -1.0
    oracle_unitary = Operator(np.diag(diag))

    oracle = QuantumCircuit(n_qubits, name="Oracle")
    oracle.unitary(oracle_unitary, range(n_qubits), label="Oracle")

    grover_op = GroverOperator(oracle)

    num_solutions = len(marked)
    # Standard optimal iteration count for Grover's algorithm.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / num_solutions) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(grover_op, range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    print(f"Marked (solution) states: {num_solutions}, Grover iterations: {iterations}")

    backend = AerSimulator()
    tqc = transpile(qc, backend, basis_gates=["u", "cx", "unitary"])
    shots = 20000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit order: rightmost classical bit is qubit 0 -> integer value
    # of the bitstring read as big-endian equals the measured basis index.
    hits_on_marked = 0
    measured_marked_values = set()
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        if value in marked:
            hits_on_marked += c
            measured_marked_values.add(value)

    hit_fraction = hits_on_marked / shots
    print(f"Fraction of shots landing on a marked (highly composite) state: {hit_fraction:.4f}")
    print(f"Distinct marked values observed among measurements: {sorted(measured_marked_values)}")

    # Success criteria:
    #  - overwhelming majority of shots land on a marked state
    #  - every marked value got observed (Grover amplifies all solutions)
    threshold = 0.90
    ran_ok = True
    verified = hit_fraction >= threshold and measured_marked_values == set(marked)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return ran_ok, verified


if __name__ == "__main__":
    main()
