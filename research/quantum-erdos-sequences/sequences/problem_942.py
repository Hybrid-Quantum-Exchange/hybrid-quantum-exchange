"""
Erdos problem #942 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com mirror manman4/erdosproblems):
    number: "942"
    oeis: ["A119241", "A119242"]
    tags: ["number theory", "powerful"]

OEIS A001694 is the standard sequence of "powerful numbers": positive
integers n such that every prime p dividing n also satisfies p^2 | n
(equivalently n = a^2 * b^3 for some positive integers a, b). A119241 and
A119242 (the two OEIS ids attached to this Erdos problem) are both
powerful-number-adjacent sequences built on that same defining predicate
(representations of powerful numbers as a^2*b^3 with b squarefree). The
underlying finite, computable property common to all of them -- and the one
this script tests with a real quantum circuit -- is exactly the classical
membership test for A001694:

    PROPERTY: n is a "powerful number" iff n == 1, or every prime factor of
    n occurs with exponent >= 2 in n's factorization.

CLASSICAL INSTANCE (computed in this script, not copied from OEIS):
    We enumerate n = 1 .. 31 (5 bits, basis states |00000> .. |11111>,
    state |n> represents integer n, with n=0 unused/never marked) and
    classically determine, from first principles (trial division), which of
    those n are powerful numbers. That classically-computed marked set is
    the "database" that Grover's algorithm searches.

QUANTUM CIRCUIT: a genuine Grover search (superposition + phase oracle
built directly from the classically-computed marked bitstrings + diffuser)
over the 5-qubit space {0,...,31}, run on the ideal AerSimulator. Grover
amplifies the marked (powerful-number) basis states; we then check that
measurement overwhelmingly returns states in the classically-computed
powerful-number set, i.e. the quantum search recovers exactly the answer
the classical check above computed.

PASS/FAIL: PASS iff, after the optimal number of Grover iterations, the set
of measurement outcomes that together account for >= 95% of the shots is a
non-empty subset of the classically-computed powerful-number set (and at
least one classically-marked state is actually observed among the top
outcomes) -- i.e. the quantum search's high-probability answers agree with
the classical answer.
"""

import math
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


N_QUBITS = 5
N = 2 ** N_QUBITS  # 32; we use basis states 0..31, n=0 is never marked


def is_powerful(n: int) -> bool:
    """Classical, from-first-principles powerful-number test (trial division).

    n is powerful iff n == 1, or every prime factor of n has exponent >= 2
    in n's prime factorization.
    """
    if n <= 0:
        return False
    if n == 1:
        return True
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            exponent = 0
            while m % p == 0:
                m //= p
                exponent += 1
            if exponent < 2:
                return False
        p += 1
    if m > 1:
        # m is a leftover prime factor with exponent exactly 1
        return False
    return True


def classical_marked_set(limit: int) -> list:
    return [n for n in range(1, limit) if is_powerful(n)]


def bitstring_for(n: int, n_qubits: int) -> str:
    return format(n, f"0{n_qubits}b")


def apply_phase_oracle(qc: QuantumCircuit, marked_values: list, n_qubits: int) -> None:
    """Flip the sign of each marked basis state via a multi-controlled Z,
    built directly from the classically-computed marked bitstrings."""
    qubits = list(range(n_qubits))
    for value in marked_values:
        bits = bitstring_for(value, n_qubits)  # bits[0] = MSB -> qubit n-1 ... standard mapping below
        # Qiskit bit order: qubit 0 is least-significant. Build bit list accordingly.
        bits_lsb_first = format(value, f"0{n_qubits}b")[::-1]
        zero_positions = [i for i, b in enumerate(bits_lsb_first) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        if zero_positions:
            qc.x(zero_positions)


def apply_diffuser(qc: QuantumCircuit, n_qubits: int) -> None:
    qubits = list(range(n_qubits))
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(marked_values: list, n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        apply_phase_oracle(qc, marked_values, n_qubits)
        apply_diffuser(qc, n_qubits)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main() -> None:
    marked = classical_marked_set(N)
    print(f"Classically computed powerful numbers in [1, {N - 1}]: {marked}")

    num_marked = len(marked)
    theta = math.asin(math.sqrt(num_marked / N))
    optimal_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"N={N}, marked={num_marked}, optimal Grover iterations={optimal_iterations}")

    qc = build_grover_circuit(marked, N_QUBITS, optimal_iterations)

    simulator = AerSimulator()
    shots = 20000
    job = simulator.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Convert bitstrings (Qiskit prints MSB-first, qubit order c[n-1]...c[0])
    # back to integers.
    int_counts = {}
    for bitstring, count in counts.items():
        value = int(bitstring, 2)
        int_counts[value] = int_counts.get(value, 0) + count

    sorted_outcomes = sorted(int_counts.items(), key=lambda kv: -kv[1])
    print("Top measurement outcomes (value: count):")
    for value, count in sorted_outcomes[:10]:
        flag = "MARKED (powerful)" if value in marked else "unmarked"
        print(f"  {value:2d}: {count:5d}  [{flag}]")

    cumulative = 0
    top_values = []
    for value, count in sorted_outcomes:
        cumulative += count
        top_values.append(value)
        if cumulative >= 0.95 * shots:
            break

    all_top_marked = all(v in marked for v in top_values)
    any_marked_observed = any(v in marked for v in top_values)
    verified = all_top_marked and any_marked_observed

    print(f"Top outcomes covering >=95% of shots: {top_values}")
    print(f"All of those outcomes are classically-marked powerful numbers: {all_top_marked}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
