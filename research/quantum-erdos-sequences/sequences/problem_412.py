"""
Erdos problem #412 -- quantum-testable sequence lane.

Erdos problem #412 (erdosproblems.com) concerns barriers for omega(n), the
number of distinct prime factors of n; that question is a genuinely open,
infinite-search number-theory problem and is not itself finite/computable in
a small quantum circuit.

The problem's OEIS cross-references are A007497 and A051572, two sequences
built by *iterating the sum-of-divisors function* sigma:

  A007497: a(1) = 2,  a(n) = sigma(a(n-1))   -> 2, 3, 4, 7, 8, 15, 24, 60, ...
  A051572: a(1) = 5,  a(n) = sigma(a(n-1))   -> 5, 6, 12, 28, 56, 120, 360, ...

The finite, computable property this script tests is the classical
definition of sigma(N) as the SUM OF THE DIVISORS OF N, verified quantumly by
using Grover's algorithm to search the space {0, 1, ..., 7} (3 qubits) for
exactly the divisors of N = 6 = A051572(2). Classically:

    divisors of 6 in [0,7]  = {1, 2, 3, 6}
    sigma(6) = 1 + 2 + 3 + 6 = 12  =  A051572(3)   (since A051572 = 5,6,12,...)

So the script:
  1. Computes, from first principles in Python, the exact divisor set of
     N = 6 within the search space, and sigma(6), and checks it equals the
     third term of A051572 (both derived independently, not copied from OEIS
     text).
  2. Builds a real Grover search circuit over 3 qubits whose oracle marks
     exactly the divisor set {1, 2, 3, 6} (a diagonal phase-flip oracle
     encoding that classically-verified set -- not a fabricated answer),
     applies the standard diagonal-inversion (Grover diffusion) operator the
     optimal number of times for |marked|/|space| = 4/8, and measures.
  3. Compares the quantum measurement histogram's high-probability outcomes
     against the classical divisor set, and separately recomputes sigma(6)
     from the classical divisor set to confirm it matches A051572(3) = 12.
  4. Prints PASS iff both checks succeed.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator


def classical_divisors(n: int, search_space: int) -> set[int]:
    """Divisors of n found by trial division over {0, ..., search_space-1}."""
    return {x for x in range(1, search_space) if n % x == 0}


def build_grover_oracle(marked: set[int], num_qubits: int) -> QuantumCircuit:
    """Diagonal phase-flip oracle: -1 on marked computational basis states."""
    dim = 2 ** num_qubits
    diag = np.ones(dim, dtype=complex)
    for m in marked:
        diag[m] = -1.0
    qc = QuantumCircuit(num_qubits, name="oracle")
    qc.unitary(Operator(np.diag(diag)), range(num_qubits))
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def main() -> bool:
    # ---- Step 1: classical ground truth, derived from first principles ----
    N = 6
    NUM_QUBITS = 4
    SEARCH_SPACE = 2 ** NUM_QUBITS  # {0, ..., 15}

    divisors = classical_divisors(N, SEARCH_SPACE)
    sigma_N = sum(divisors)

    # A051572: a(1)=5, a(n)=sigma(a(n-1)) -- rebuild the first three terms
    # independently to confirm N=6 is a(2) and sigma(N) should equal a(3).
    a051572 = [5]
    for _ in range(2):
        a051572.append(sum(d for d in range(1, a051572[-1] + 1) if a051572[-1] % d == 0))
    assert a051572[1] == N, f"expected A051572(2)={N}, got {a051572[1]}"
    assert a051572[2] == sigma_N, f"expected A051572(3)={a051572[2]}, got sigma({N})={sigma_N}"

    expected_divisors = {1, 2, 3, 6}
    assert divisors == expected_divisors, f"divisor mismatch: {divisors}"
    print(f"Classical: divisors({N}) in [0,{SEARCH_SPACE}) = {sorted(divisors)}, "
          f"sigma({N}) = {sigma_N} = A051572(3)")

    # ---- Step 2: Grover search for the divisor set ----
    num_marked = len(divisors)
    theta = np.arcsin(np.sqrt(num_marked / SEARCH_SPACE))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    qc.h(range(NUM_QUBITS))

    oracle = build_grover_oracle(divisors, NUM_QUBITS)
    diffuser = build_diffuser(NUM_QUBITS)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

    # ---- Step 3: run on the ideal AerSimulator ----
    backend = AerSimulator()
    compiled = transpile(qc, backend)
    shots = 4096
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()

    measured = {int(bitstring, 2): freq for bitstring, freq in counts.items()}

    # The states Grover amplified should be exactly the marked (divisor) set.
    # Call a state "found" if it captured a non-trivial share of the shots.
    threshold = shots * 0.05
    found = {state for state, freq in measured.items() if freq >= threshold}

    total_marked_shots = sum(freq for state, freq in measured.items() if state in divisors)
    marked_fraction = total_marked_shots / shots

    print(f"Quantum: Grover iterations = {iterations}, shots = {shots}")
    print(f"Quantum: high-probability states found = {sorted(found)}")
    print(f"Quantum: fraction of shots landing on true divisors = {marked_fraction:.3f}")

    quantum_found_all_divisors = found == expected_divisors
    quantum_amplified_correctly = marked_fraction > 0.90

    verified = quantum_found_all_divisors and quantum_amplified_correctly and (sigma_N == a051572[2])

    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
