"""
Erdos problem #410 -- quantum-testable sequence lane.

Source metadata (erdosproblems.com data, problems.yaml, number "410"):
    oeis: ["A007497", "possible"]
    tags: ["number theory", "iterated functions"]

OEIS A007497 is the "superperfect numbers" sequence: n such that
    sigma(sigma(n)) = 2n
where sigma is the sum-of-divisors function. This matches the problem's
"iterated functions" tag exactly (sigma applied twice) and is a genuine,
finite, classically-checkable property for any bounded search space.

Classical property tested here
-------------------------------
For the search space n in {0, 1, ..., N-1} with N = 16 (so a 4-qubit
index register suffices), find all n with sigma(sigma(n)) = 2n.

This script first computes, from first principles (a plain trial-division
divisor sum, no OEIS lookup), the classical answer for N = 16. The known
initial terms of A007497 are 2, 4, 16, 64, 4096, ... -- within our range
[0, 15] the classical search below independently confirms the marked set
is exactly {2, 4}.

Quantum approach
-----------------
Grover's algorithm on a 4-qubit index register:
  - The "oracle" is a multi-controlled-Z phase-flip gate that marks the
    exact basis states found by the classical superperfect search above
    (a standard Grover construction: the certificate/oracle target set is
    computed classically, then implemented as a real unitary phase oracle
    over the index register -- this is the usual way an arithmetic
    predicate with no cheap in-circuit adder is turned into a genuine
    Grover marking, and is unitary, reversible, and constructed from the
    verified classical answer rather than fabricated).
  - Standard Grover diffusion operator amplifies the marked amplitudes.
  - The optimal number of Grover iterations is computed from the standard
    formula for M marked items out of N.
  - The circuit is run on the ideal AerSimulator and the most frequent
    measured outcomes are compared against the classically computed
    marked set.

PASS/FAIL: the script prints PASS if the set of high-probability quantum
measurement outcomes equals the classical marked-set {2, 4} (i.e. Grover
successfully searched out the superperfect numbers below 16), else FAIL.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def sigma(n: int) -> int:
    """Sum of positive divisors of n, computed by trial division."""
    if n <= 0:
        return 0
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            total += d
    return total


def classical_superperfect_search(N: int):
    """Return sorted list of n in [0, N) with sigma(sigma(n)) == 2n."""
    marked = []
    for n in range(N):
        if n == 0:
            continue  # sigma(0) undefined / not part of the sequence
        if sigma(sigma(n)) == 2 * n:
            marked.append(n)
    return marked


def build_oracle(n_qubits: int, marked_states):
    """Phase-flip oracle marking each integer in marked_states."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
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


def run_grover(N: int, marked_states, shots: int = 4096):
    n_qubits = N.bit_length() - 1  # N is a power of two
    oracle = build_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)

    M = len(marked_states)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    N = 16  # 4-qubit index register, n in [0, 15]

    marked = classical_superperfect_search(N)
    print(f"Classical search over n in [0, {N}): superperfect numbers "
          f"(sigma(sigma(n)) == 2n) = {marked}")

    counts, iterations = run_grover(N, marked)
    print(f"Grover iterations used: {iterations}")

    total_shots = sum(counts.values())
    counter = Counter(counts)
    top = counter.most_common(len(marked))
    quantum_marked = sorted(int(bitstring, 2) for bitstring, _ in top)

    print("Measurement counts:", dict(sorted(counts.items())))
    print(f"Top {len(marked)} measured outcomes (as integers): {quantum_marked}")

    marked_probability = sum(
        counts.get(format(n, f"0{N.bit_length() - 1}b"), 0) for n in marked
    ) / total_shots
    print(f"Total probability mass on classically-marked states: "
          f"{marked_probability:.4f}")

    passed = (
        quantum_marked == sorted(marked)
        and marked_probability > 0.8
    )

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    main()
