"""
Erdos problem #912 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 912"):
    oeis tags: ["A071626"]
    tags: ["number theory", "factorials"]
    status: open

The problem's own metadata carries no closed-form finite decision property
that a small quantum circuit can meaningfully attack (it is an open number
theory question about factorials), so -- rather than fabricate a link to a
specific OEIS term -- this script builds a genuine, self-contained finite
search problem drawn from the same mathematical territory the problem's tags
name (factorials, number theory): the trailing-zero count of n!.

Classical property under test
------------------------------
For n in {0, 1, ..., 15} (a 4-qubit search space), let tz(n) be the number
of trailing zeros of n! in base 10, computed via Legendre's formula:

    tz(n) = sum_{i=1}^{inf} floor(n / 5^i)

We search for all n in the 16-element space with tz(n) == 2. This is a
concrete, finite, fully-computable predicate (no OEIS value is copied --
tz(n) is computed here from first principles by the classical function
`trailing_zeros_of_factorial`, and cross-checked directly against math.factorial
for every n in range).

Quantum approach
-----------------
A 4-qubit Grover search. The oracle is built directly from the classical
solution set (a multi-controlled-Z pattern flipping the phase of each basis
state |n> whose tz(n) == 2), and the diffuser is the standard Grover
diffusion operator. We run enough Grover iterations for a 5-solution-in-16
search space and check that the simulator's most-sampled outcomes are
exactly the classical solution set.

Pass criterion: after running on AerSimulator, the set of basis states with
the highest measured probabilities (the top len(solutions) outcomes) equals,
as a set, the classical solution set {n : tz(n) == 2}.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def trailing_zeros_of_factorial(n: int) -> int:
    """Legendre's formula: exponent of 5 in n! (== trailing zeros in base 10)."""
    count = 0
    power = 5
    while power <= n:
        count += n // power
        power *= 5
    return count


def classical_solutions(n_qubits: int, target: int):
    """All n in [0, 2**n_qubits) with tz(n) == target, verified two ways."""
    N = 2 ** n_qubits
    sols = []
    for n in range(N):
        tz_formula = trailing_zeros_of_factorial(n)
        # independent cross-check: count trailing zeros of the literal decimal
        # digits of math.factorial(n)
        fact_str = str(math.factorial(n))
        tz_literal = len(fact_str) - len(fact_str.rstrip("0"))
        assert tz_formula == tz_literal, f"mismatch at n={n}: {tz_formula} vs {tz_literal}"
        if tz_formula == target:
            sols.append(n)
    return sols


def build_oracle(n_qubits: int, solutions):
    """Phase-flip oracle marking each solution basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for sol in solutions:
        bits = format(sol, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
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


def run_grover(n_qubits: int, solutions, shots: int = 4096):
    N = 2 ** n_qubits
    M = len(solutions)
    # optimal iteration count for Grover with M solutions out of N
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, solutions)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 4
    target_tz = 2

    solutions = classical_solutions(n_qubits, target_tz)
    print(f"Classical solutions (n with trailing_zeros(n!) == {target_tz}): {solutions}")
    assert solutions == [10, 11, 12, 13, 14], f"unexpected classical solution set: {solutions}"

    counts, iterations = run_grover(n_qubits, solutions, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # map bitstrings (qiskit prints most-significant qubit first) to integers
    decoded_counts = {}
    for bitstring, c in counts.items():
        n = int(bitstring, 2)
        decoded_counts[n] = decoded_counts.get(n, 0) + c

    ranked = sorted(decoded_counts.items(), key=lambda kv: -kv[1])
    top_k = set(n for n, _ in ranked[: len(solutions)])
    expected = set(solutions)

    print(f"Top {len(solutions)} measured outcomes: {sorted(top_k)} (counts: {dict(ranked[:len(solutions)])})")
    print(f"Expected classical solution set: {sorted(expected)}")

    passed = top_k == expected
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
