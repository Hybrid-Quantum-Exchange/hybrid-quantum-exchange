"""
Erdos problem #998 (as catalogued in the erdosproblems.com data set,
data/problems.yaml entry "number: 998").

Metadata for #998: prize = no, status = proved, tags = ["analysis",
"diophantine approximation"], oeis = ["N/A"]. The entry carries NO OEIS
sequence id -- it documents a proved analytic/diophantine-approximation
result, not an integer sequence. So the instruction "identify a small,
finite, computable property of the [OEIS] sequence" cannot be honestly
followed for this problem: there is no sequence to draw a property from.

HONEST LIMITATION: this script does not encode problem #998 itself (no
OEIS id exists to derive a verifiable finite property from, and the
formal statement is an analysis-flavoured approximation result, not
obviously reducible to a small finite/computable instance without
inventing math not actually in the problem). Rather than fabricate a
fake tie to #998, this script implements a genuine, honestly-labelled
example from the same *tag* the problem carries -- "diophantine
approximation" -- namely Dirichlet's approximation theorem:

    For irrational alpha and integer Q > 1, there exists an integer q
    with 1 <= q <= Q-1 such that the fractional part of q*alpha is
    within 1/Q of 0 or 1, i.e. |q*alpha - round(q*alpha)| < 1/Q.

Classical part (computed here from first principles, not copied from
any table): fix alpha = sqrt(2) and Q = 8, brute-force over q in
{1, ..., 7} (3 qubits) for the smallest q satisfying the inequality.

Quantum part: a real Grover search circuit over the 3-qubit register
{0,...,7} whose oracle marks exactly the q in {1,...,7} satisfying
Dirichlet's inequality for alpha=sqrt(2), Q=8 (oracle built from the
precomputed classical truth table, since the inequality itself is a
real-valued analytic condition with no small arithmetic circuit -- the
oracle is a bona fide marking oracle over the search space, and Grover's
algorithm genuinely amplifies the marked classical solutions found by
the classical precomputation). The circuit is run on the ideal
AerSimulator, and its most likely measured outcome is compared against
the classical answer.

Because this problem (#998) itself has no OEIS id and no directly
reducible finite instance, `verified_against_classical` and `ran_ok`
below describe THIS script's Dirichlet-approximation stand-in, not a
faithful encoding of the exact statement of Erdos problem #998.
"""

import math
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_dirichlet_solutions(alpha: float, Q: int):
    """Brute-force every q in {1,...,Q-1} and return those satisfying
    Dirichlet's approximation inequality |q*alpha - round(q*alpha)| < 1/Q.
    Also returns the smallest such q (guaranteed to exist by the theorem).
    """
    solutions = []
    for q in range(1, Q):
        frac_val = q * alpha - round(q * alpha)
        if abs(frac_val) < 1.0 / Q:
            solutions.append(q)
    if not solutions:
        raise RuntimeError("Dirichlet's theorem guarantees a solution; none found -- bug.")
    return solutions, min(solutions)


def build_oracle(n_qubits: int, marked_values):
    """Phase-flip oracle marking each value in `marked_values` (as an
    n_qubits-bit binary string, little-endian) by controlled-Z on the
    all-ones pattern after X-conjugating the 0-bits.
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = [(value >> i) & 1 for i in range(n_qubits)]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_search(n_qubits: int, marked_values, shots: int = 4096):
    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)

    N = 2 ** n_qubits
    M = len(marked_values)
    # optimal number of Grover iterations
    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    alpha = math.sqrt(2)
    Q = 8  # search space size -> 3 qubits, q in {0,...,7}; 0 excluded by theorem range
    n_qubits = 3

    solutions, smallest = classical_dirichlet_solutions(alpha, Q)
    print(f"alpha = sqrt(2), Q = {Q}")
    print(f"classical solutions q in 1..{Q-1} with |q*alpha - round(q*alpha)| < 1/{Q}: "
          f"{solutions}")
    print(f"classical smallest solution: q = {smallest}")

    counts = grover_search(n_qubits, solutions)
    print(f"quantum measurement counts: {counts}")

    # most frequent measured bitstring -> integer (qiskit bitstrings are big-endian
    # over the classical register in the order given to measure(), same order as
    # our little-endian oracle construction since we measured qubit i into bit i)
    best_bitstring = max(counts, key=counts.get)
    measured_value = int(best_bitstring[::-1], 2)

    print(f"quantum most-likely result: q = {measured_value}")

    # success criterion: the quantum-favoured value is among the true classical
    # solution set (Grover amplifies all marked states, so any of them is a
    # valid "found" answer), and its measured probability is markedly amplified
    # above the uniform baseline 1/Q.
    total_shots = sum(counts.values())
    marked_prob = sum(counts.get(format(v, f"0{n_qubits}b")[::-1], 0) for v in solutions) / total_shots
    baseline_prob = len(solutions) / Q

    is_marked = measured_value in solutions
    is_amplified = marked_prob > baseline_prob * 1.5  # comfortably above chance

    verified = is_marked and is_amplified
    print(f"P(marked) = {marked_prob:.3f}  vs classical baseline (no amplification) = {baseline_prob:.3f}")

    if verified:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
