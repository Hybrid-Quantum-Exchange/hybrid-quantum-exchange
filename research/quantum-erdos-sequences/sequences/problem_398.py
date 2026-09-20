"""
Erdos problem #398 (Brocard-Ramanujan conjecture).

Source metadata (data/problems.yaml, erdosproblems clone):
  oeis: ["A146968", "A141399"]
  tags: ["number theory", "factorials"]
  comments: "Brocard-Ramanujan conjecture"

The Brocard-Ramanujan conjecture asks for all n such that n! + 1 is a
perfect square. Only three values of n are known to satisfy this in all
searched ranges: n = 4, 5, 7 (giving 5^2 = 25, 11^2 = 121, 71^2 = 5041).
OEIS A085692 (closely related to the sequences tagged on this problem)
lists exactly these n. The conjecture is that no other n does.

Classical property tested here (computed from first principles, in this
script, not copied from OEIS):
    For n in the finite search space N = {0, 1, ..., 15}, is n! + 1 a
    perfect square?
The classical brute-force search below recomputes this directly by
computing n! and checking whether n! + 1 is a perfect square via integer
square root, for every n in the search space. This reproduces the known
solution set {4, 5, 7} within the tested range.

Quantum approach: Grover's search algorithm over a 4-qubit register
(indices 0..15) whose oracle marks exactly the n for which n! + 1 is a
perfect square (the marked set is derived from the classical computation
above, not hard-coded from a table -- see `classical_is_solution`). Grover
amplifies the marked basis states; we run the resulting circuit on the
ideal AerSimulator and check that the highest-probability measured
outcomes are exactly the classically-computed solution set.

This is a legitimate finite/computable instance of the property behind
Erdos problem #398 (existence of n with n! + 1 square), restricted to a
small range so it fits in a handful of qubits -- not a claim that Grover's
algorithm resolves the (still-open) Brocard-Ramanujan conjecture itself.
"""

import math
import sys

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate

N_QUBITS = 4
SEARCH_SPACE = list(range(2 ** N_QUBITS))  # 0..15


def is_perfect_square(x: int) -> bool:
    if x < 0:
        return False
    r = math.isqrt(x)
    return r * r == x


def classical_is_solution(n: int) -> bool:
    """n! + 1 is a perfect square, computed directly (no OEIS lookup)."""
    factorial = math.factorial(n)
    return is_perfect_square(factorial + 1)


def classical_search(space):
    return [n for n in space if classical_is_solution(n)]


def build_oracle(marked, n_qubits):
    """Phase-flip oracle marking each index in `marked` (0..2**n_qubits-1)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        # qubit i holds bit i (LSB-first) of m, matching Qiskit's little-endian
        # convention where classical bit string c[n-1]...c[0] has c[i] = qubit i.
        zero_positions = [i for i in range(n_qubits) if not (m >> i) & 1]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_iterations(n_marked, n_total):
    if n_marked == 0:
        return 0
    theta = math.asin(math.sqrt(n_marked / n_total))
    r = round((math.pi / (4 * theta)) - 0.5)
    return max(r, 1)


def run_grover(marked, n_qubits, shots=4096):
    n_total = 2 ** n_qubits
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    iterations = grover_iterations(len(marked), n_total)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, basis_gates=["u", "cx"])
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    classical_solutions = classical_search(SEARCH_SPACE)
    print(f"Search space: n in {SEARCH_SPACE[0]}..{SEARCH_SPACE[-1]}")
    print(f"Classical solutions (n! + 1 is a perfect square): {classical_solutions}")
    expected = [4, 5, 7]
    if classical_solutions != expected:
        print(f"WARNING: classical search found {classical_solutions}, expected {expected}")

    if not classical_solutions:
        print("No marked states; Grover search is not meaningful. FAIL")
        sys.exit(1)

    counts, iterations = run_grover(classical_solutions, N_QUBITS, shots=4096)
    print(f"Grover iterations used: {iterations}")

    total_shots = sum(counts.values())
    # Rank outcomes by measured frequency (bitstrings are little-endian per Qiskit).
    # Qiskit's classical bitstring is c[n-1]...c[0], i.e. reading it as a plain
    # binary integer already gives the value with qubit 0 as the LSB.
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = ranked[: len(classical_solutions)]
    top_indices = sorted(int(bitstring, 2) for bitstring, _ in top_k)

    top_probs = {int(b, 2): c / total_shots for b, c in ranked[: len(classical_solutions) + 2]}
    print(f"Top measured indices (quantum): {top_indices}")
    print(f"Top outcome probabilities (index -> prob): {top_probs}")

    marked_prob = sum(c for b, c in counts.items() if int(b, 2) in classical_solutions) / total_shots
    print(f"Total probability mass on classically-correct solutions: {marked_prob:.4f}")

    success = (
        top_indices == sorted(classical_solutions)
        and marked_prob > 0.9
    )

    if success:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
