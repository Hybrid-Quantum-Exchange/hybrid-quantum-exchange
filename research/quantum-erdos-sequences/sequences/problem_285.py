"""
Erdos problem #285 -- quantum-testable instance.

OEIS: A030659, "smallest possible maximum denominator in a representation
of 1 as a sum of n distinct unit fractions."  A030659(3) = 6, via the
unique (up to order) representation

    1 = 1/2 + 1/3 + 1/6

Classical property being tested
--------------------------------
For n = 3, with the smallest denominator forced to be 2 (a triple of
distinct unit fractions 1/a < 1/b < 1/c summing to 1 must have a = 2,
since three distinct unit fractions all with denominator >= 3 sum to at
most 1/3 + 1/4 + 1/5 = 47/60 < 1), the remaining equation is

    1/b + 1/c = 1/2,   2 < b < c

Searching b over a small finite space (b = 3 .. 15, i.e. a 4-qubit
register), the classical script below checks, for each b, whether
c = 1 / (1/2 - 1/b) is an integer with c > b. This brute force is done
from first principles (no OEIS value is copied) and finds exactly one
solution in that range: b = 3, c = 6, matching A030659(3) = 6.

Quantum circuit
----------------
A Grover search over the 4-qubit register {0, ..., 15} is built whose
oracle marks exactly the classically-found solution b*. Grover
amplifies the marked basis state's amplitude; measuring the circuit on
the ideal AerSimulator should return b* = 3 as the overwhelmingly most
frequent outcome. This verifies that Grover's algorithm, run on this
concrete instance, converges on the same answer the classical search
established -- i.e. that the quantum search correctly locates the
denominator b that yields A030659(3) = 6.

PASS/FAIL is decided by comparing the most frequent measured value to
the classically-computed b*.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4          # search space: b in [0, 15]
SEARCH_LO = 3          # b must exceed a=2
SEARCH_HI = (1 << N_QUBITS) - 1


def classical_solutions():
    """Brute-force, from first principles, all b in [SEARCH_LO, SEARCH_HI]
    such that 1/2 + 1/b + 1/c = 1 for some integer c > b."""
    solutions = []
    for b in range(SEARCH_LO, SEARCH_HI + 1):
        remainder = 0.5 - 1.0 / b
        if remainder <= 0:
            continue
        # 1/c = remainder  =>  c = 1/remainder, check exact integrality
        # via integer arithmetic: c = 2b / (b - 2)
        num, den = 2 * b, b - 2
        if den <= 0:
            continue
        if num % den == 0:
            c = num // den
            if c > b:
                # sanity-check with fractions
                assert abs((1.0 / 2) + (1.0 / b) + (1.0 / c) - 1.0) < 1e-12
                solutions.append((b, c))
    return solutions


def build_oracle(marked_value, n_qubits):
    """Phase-flip oracle marking exactly |marked_value> in an n-qubit register."""
    oracle = QuantumCircuit(n_qubits, name="oracle")
    bits = format(marked_value, f"0{n_qubits}b")[::-1]  # little-endian
    zero_qubits = [i for i, bit in enumerate(bits) if bit == "0"]

    for q in zero_qubits:
        oracle.x(q)
    if n_qubits == 1:
        oracle.z(0)
    else:
        oracle.h(n_qubits - 1)
        oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        oracle.h(n_qubits - 1)
    for q in zero_qubits:
        oracle.x(q)
    return oracle


def build_diffuser(n_qubits):
    diffuser = QuantumCircuit(n_qubits, name="diffuser")
    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    if n_qubits == 1:
        diffuser.z(0)
    else:
        diffuser.h(n_qubits - 1)
        diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        diffuser.h(n_qubits - 1)
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))
    return diffuser


def grover_search(marked_value, n_qubits, shots=4096):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_value, n_qubits)
    diffuser = build_diffuser(n_qubits)

    # Optimal number of Grover iterations for a single marked item out of 2^n.
    N = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(N)))

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    solutions = classical_solutions()
    print(f"Classical search over b in [{SEARCH_LO}, {SEARCH_HI}]: "
          f"found solutions (b, c) = {solutions}")

    assert len(solutions) == 1, "expected a unique solution in this search range"
    b_star, c_star = solutions[0]
    assert (b_star, c_star) == (3, 6), "classical answer must match A030659(3) = 6"
    print(f"Classical answer: b* = {b_star}, giving max denominator c* = {c_star} "
          f"(A030659(3) = 6)")

    counts = grover_search(b_star, N_QUBITS)
    # counts keys are bitstrings in Qiskit's little-endian convention already
    # matching qubit order used in build_oracle/build_diffuser.
    int_counts = Counter()
    for bitstring, freq in counts.items():
        # Qiskit's bitstring is c[n-1]...c[0], i.e. qubit 0 is already the
        # least-significant (rightmost) character, so no reversal is needed.
        value = int(bitstring, 2)
        int_counts[value] += freq

    most_common_value, most_common_freq = int_counts.most_common(1)[0]
    total_shots = sum(int_counts.values())
    print(f"Grover search most frequent measured value: {most_common_value} "
          f"({most_common_freq}/{total_shots} shots)")
    print(f"Full measured distribution (value: count): {dict(sorted(int_counts.items()))}")

    quantum_ok = (most_common_value == b_star) and (most_common_freq / total_shots > 0.5)

    if quantum_ok:
        print("PASS: Grover search located b* matching the classical A030659(3) answer.")
    else:
        print("FAIL: Grover search did not converge on the classical answer.")

    return quantum_ok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
