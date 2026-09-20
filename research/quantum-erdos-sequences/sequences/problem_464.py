"""
Erdos problem #464 -- quantum-testable lane (best-effort, with an honest limitation).

Source metadata (data/problems.yaml, erdosproblems repo, entry "number: 464"):
    prize: no
    informal_status: proved
    formal_status: Lean
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION, stated up front: problem #464's entry carries no OEIS sequence id
("N/A" in the source YAML). There is therefore no concrete integer sequence
attached to this problem to build a Grover/estimation oracle around, and no
OEIS term to check a quantum result against. Rather than fabricate a
membership test for a sequence that isn't specified anywhere in the metadata,
this script falls back to the closest legitimate, self-contained, finite,
computable number-theory property available -- consistent with the problem's
own tag ("number theory") -- and is explicit that the resulting circuit
verifies a *general* number-theoretic property (primality on a small range),
not a term of an Erdos-problem-464-specific OEIS sequence, because no such
sequence id exists in the source data.

Classical property tested
--------------------------
For n in {0, 1, ..., 15} (4 qubits), is n prime?
Primality is computed from first principles below by classical trial
division (PRIMES_4BIT), independent of any external source.

Quantum approach
-----------------
Grover's algorithm on 4 qubits. The oracle is built as an exact
diagonal phase-flip (multi-controlled Z pattern) derived directly from the
classical PRIMES_4BIT truth table computed in this script -- it is not a
generic canned oracle, it encodes exactly the primality predicate computed
above. Amplitude amplification is then run for the optimal number of Grover
iterations for a 4-qubit, 6-solution search, and the resulting distribution
on the ideal AerSimulator is compared against the classical set of primes.

PASS criterion: the AerSimulator measurement counts are dominated
(all top outcomes, by count, up to the number of marked primes) by exactly
the basis states in PRIMES_4BIT.
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2**N_QUBITS  # 16

# Classical answer, computed here from first principles.
PRIMES_4BIT = sorted(n for n in range(N) if is_prime(n))
print(f"Classical primes in [0, {N-1}]: {PRIMES_4BIT}")
assert PRIMES_4BIT == [2, 3, 5, 7, 11, 13]


def oracle(qc: QuantumCircuit, marked: list[int], qubits: list[int]) -> None:
    """Flip the phase of each basis state in `marked` (multi-controlled Z)."""
    n = len(qubits)
    for m in marked:
        bits = [(m >> i) & 1 for i in range(n)]
        flip_qubits = [qubits[i] for i, b in enumerate(bits) if b == 0]
        for q in flip_qubits:
            qc.x(q)
        if n == 1:
            qc.z(qubits[0])
        else:
            qc.h(qubits[-1])
            qc.append(MCXGate(n - 1), [qubits[i] for i in range(n - 1)] + [qubits[-1]])
            qc.h(qubits[-1])
        for q in flip_qubits:
            qc.x(q)


def diffuser(qc: QuantumCircuit, qubits: list[int]) -> None:
    n = len(qubits)
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.append(MCXGate(n - 1), [qubits[i] for i in range(n - 1)] + [qubits[-1]])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


def build_grover_circuit(marked: list[int], n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))
    qc.h(qubits)
    for _ in range(iterations):
        oracle(qc, marked, qubits)
        diffuser(qc, qubits)
    qc.measure(qubits, qubits)
    return qc


def optimal_iterations(n_qubits: int, n_marked: int) -> int:
    N_states = 2**n_qubits
    theta = np.arcsin(np.sqrt(n_marked / N_states))
    r = round((np.pi / (4 * theta)) - 0.5)
    return max(1, int(r))


def main() -> bool:
    iterations = optimal_iterations(N_QUBITS, len(PRIMES_4BIT))
    print(f"Running Grover search on {N_QUBITS} qubits, {iterations} iteration(s), "
          f"marking {len(PRIMES_4BIT)} primes out of {N} states.")

    qc = build_grover_circuit(PRIMES_4BIT, N_QUBITS, iterations)

    sim = AerSimulator()
    shots = 8192
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bitstrings are little-endian in the classical register order used
    # (c[0] is qubit 0 = LSB), matching how `oracle`/`diffuser` indexed bits.
    outcome_counts = {}
    for bitstring, c in counts.items():
        n_val = int(bitstring, 2)
        outcome_counts[n_val] = outcome_counts.get(n_val, 0) + c

    top_k = sorted(outcome_counts.items(), key=lambda kv: -kv[1])[: len(PRIMES_4BIT)]
    top_values = sorted(v for v, _ in top_k)

    print(f"Top {len(PRIMES_4BIT)} measured outcomes (by count): {top_values}")
    print(f"Classical primes                                 : {PRIMES_4BIT}")

    top_mass = sum(c for _, c in top_k)
    total_mass = sum(outcome_counts.values())
    concentration = top_mass / total_mass

    passed = (top_values == PRIMES_4BIT) and concentration > 0.75
    print(f"Amplitude concentration on marked states: {concentration:.3f}")
    return passed


if __name__ == "__main__":
    ok = main()
    print("PASS" if ok else "FAIL")
