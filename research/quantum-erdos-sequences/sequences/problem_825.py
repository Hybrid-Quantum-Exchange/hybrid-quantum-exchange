"""
Erdos problem #825 -- quantum-testable instance.

Erdos problem 825 concerns divisors / unit fractions and is linked (per the
problems.yaml metadata) to OEIS sequences A006037 (the "untouchable numbers":
positive integers that are not the sum of the proper divisors of any positive
integer) and A330244.

Classical property tested here (derived from first principles in this script,
not copied from OEIS):

    Let s(m) = sum of the proper divisors of m ("aliquot sum" / sigma(m) - m).
    A number n is TOUCHABLE if there exists some m with s(m) = n; it is
    UNTOUCHABLE (a member of A006037) otherwise. Membership testing for
    A006037 is exactly the decision problem "does there exist m in some
    search range R such that s(m) = n?" -- a finite search problem, which is
    what makes it quantum-testable with Grover's algorithm.

    Chosen finite instance: n = 4, search range m in {0, 1, ..., 31}
    (5 qubits). The script first computes s(m) for every m in that range
    classically, from first principles (by summing actual proper divisors,
    no OEIS lookup), and determines the full solution set
        S = { m in [0,31] : s(m) = 4 }.
    This is computed classically below and found to be the singleton
    S = {9} (since the proper divisors of 9 are {1, 3}, summing to 4, and no
    other m in [0,31] satisfies s(m) = 4). So n = 4 is TOUCHABLE within this
    range, witnessed uniquely by m = 9.

Quantum circuit: Grover's search algorithm over the 5-qubit register
representing m in [0,31]. The oracle is built directly from the classical
solution set S (a diagonal phase-flip on exactly the marked basis states),
and the standard Grover diffusion operator is applied for the optimal number
of iterations for |S| = 1 out of 32 states. The circuit is run on the ideal
AerSimulator; the most frequently measured basis state is compared against
the classically-computed unique solution m = 9.

Honesty note / limitation: this demonstrates the underlying decision problem
of A006037 membership (finite existence-of-witness search) on a small
instance, not a full proof that any particular number is untouchable (that
would require an unbounded search / analytic argument). The oracle itself is
literally constructed from the classically verified solution set, so success
here shows the quantum search correctly retrieves that classically-derived
witness, not that classical computation was bypassed.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def proper_divisor_sum(m: int) -> int:
    """s(m): sum of proper divisors of m, computed by trial division."""
    if m <= 1:
        return 0
    total = 0
    for d in range(1, m):
        if m % d == 0:
            total += d
    return total


def classical_solution_set(n: int, hi: int) -> list[int]:
    """All m in [0, hi] with proper_divisor_sum(m) == n, from first principles."""
    return [m for m in range(0, hi + 1) if proper_divisor_sum(m) == n]


N_QUBITS = 5
RANGE_HI = (1 << N_QUBITS) - 1  # 31
TARGET_N = 4


def build_oracle(marked_states: list[int], n_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle: multiplies |m> by -1 for each m in marked_states."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
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


def build_grover_circuit(marked_states: list[int], n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked_states, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main() -> None:
    # Step 1: classical computation of the solution set (first principles).
    solutions = classical_solution_set(TARGET_N, RANGE_HI)
    print(f"Classical search: s(m) for m in [0,{RANGE_HI}], target n = {TARGET_N}")
    print(f"Classical solution set S = {solutions}")

    if len(solutions) != 1:
        print(
            "FAIL: expected a unique witness in this instance for a clean "
            f"Grover demonstration, found {len(solutions)}."
        )
        return

    classical_answer = solutions[0]
    print(f"Classical witness m = {classical_answer} (i.e. n={TARGET_N} is touchable via m={classical_answer})")
    print(f"  check: proper divisors of {classical_answer} sum to "
          f"{proper_divisor_sum(classical_answer)}")

    # Step 2: Grover search for the witness.
    num_solutions = len(solutions)
    search_space = 1 << N_QUBITS
    iterations = max(1, round(math.pi / 4 * math.sqrt(search_space / num_solutions)))
    print(f"Grover iterations: {iterations} (search space size {search_space}, "
          f"{num_solutions} marked state)")

    qc = build_grover_circuit(solutions, N_QUBITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bitstrings are ordered c[n-1]...c[0] left-to-right; since qubit i
    # holds bit i (qubit 0 = LSB), that left-to-right string is already the
    # standard MSB-first binary representation of the integer value.
    best_bitstring = max(counts, key=counts.get)
    measured_value = int(best_bitstring, 2)
    top_probability = counts[best_bitstring] / shots

    print(f"Most frequent measured value: {measured_value} "
          f"(probability ~{top_probability:.3f} over {shots} shots)")

    quantum_answer = measured_value
    passed = quantum_answer == classical_answer

    print(f"Classical answer: {classical_answer}")
    print(f"Quantum answer:   {quantum_answer}")
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
