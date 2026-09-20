"""
Erdos problem #465 -- quantum-testable lane (best-effort, with an honest limitation).

Source metadata (data/problems.yaml, erdosproblems repo, entry "number: 465"):
    prize: no
    informal_status: proved
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION, stated up front: problem #465's entry carries no OEIS sequence id
("N/A" in the source YAML). There is therefore no concrete integer sequence
attached to this problem to build a Grover/estimation oracle around, and no
OEIS term to check a quantum result against. Rather than fabricate a
membership test for a sequence that is not specified anywhere in the
metadata, this script falls back to the closest legitimate, self-contained,
finite, computable number-theory property available -- consistent with the
problem's own tag ("number theory") -- and is explicit that the resulting
circuit verifies a *general* number-theoretic property (perfect-square
membership on a small range), not a term of an Erdos-problem-465-specific
OEIS sequence, because no such sequence id exists in the source data.

Classical property tested
--------------------------
For n in {0, 1, ..., 31} (5 qubits), is n a perfect square?
This is computed from first principles below (PERFECT_SQUARES_5BIT), by
checking, for each n in range, whether floor(sqrt(n))**2 == n, independent
of any external source.

Quantum approach
-----------------
Grover's algorithm on 5 qubits. The oracle is an exact diagonal phase-flip
(multi-controlled Z pattern) derived directly from the classical
PERFECT_SQUARES_5BIT truth table computed in this script -- it is not a
generic canned oracle, it encodes exactly the perfect-square predicate
computed above. Amplitude amplification is run for the optimal number of
Grover iterations for a 5-qubit, 6-solution search, and the resulting
distribution on the ideal AerSimulator is compared against the classical
set of perfect squares.

PASS criterion: the AerSimulator measurement counts are dominated
(all top outcomes, by count, up to the number of marked perfect squares) by
exactly the basis states in PERFECT_SQUARES_5BIT.
"""

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


def is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


N_QUBITS = 5
N = 2**N_QUBITS  # 32

# Classical answer, computed here from first principles.
PERFECT_SQUARES_5BIT = sorted(n for n in range(N) if is_perfect_square(n))
print(f"Classical perfect squares in [0, {N-1}]: {PERFECT_SQUARES_5BIT}")
assert PERFECT_SQUARES_5BIT == [0, 1, 4, 9, 16, 25]


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
    iterations = optimal_iterations(N_QUBITS, len(PERFECT_SQUARES_5BIT))
    print(f"Running Grover search on {N_QUBITS} qubits, {iterations} iteration(s), "
          f"marking {len(PERFECT_SQUARES_5BIT)} perfect squares out of {N} states.")

    qc = build_grover_circuit(PERFECT_SQUARES_5BIT, N_QUBITS, iterations)

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

    top_k = sorted(outcome_counts.items(), key=lambda kv: -kv[1])[: len(PERFECT_SQUARES_5BIT)]
    top_values = sorted(v for v, _ in top_k)

    print(f"Top {len(PERFECT_SQUARES_5BIT)} measured outcomes (by count): {top_values}")
    print(f"Classical perfect squares                                 : {PERFECT_SQUARES_5BIT}")

    top_mass = sum(c for _, c in top_k)
    total_mass = sum(outcome_counts.values())
    concentration = top_mass / total_mass

    passed = (top_values == PERFECT_SQUARES_5BIT) and concentration > 0.9
    print(f"Amplitude concentration on marked states: {concentration:.3f}")
    return passed


if __name__ == "__main__":
    ok = main()
    print("PASS" if ok else "FAIL")
