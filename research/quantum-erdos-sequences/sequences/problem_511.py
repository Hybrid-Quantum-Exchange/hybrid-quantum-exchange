"""
Erdos problem #511 -- quantum-testable-sequence lane.

Source record (data/problems.yaml, erdosproblems repo, entry "number: '511'"):
    prize: no
    status: disproved (2025-09-15)
    oeis: ["N/A"]
    tags: ["analysis"]

LIMITATION (read first): problem #511 carries no OEIS sequence id -- its
`oeis` field is literally ["N/A"] and its single tag ("analysis") gives no
finite combinatorial object either. There is therefore no genuine small
computable sequence-membership/counting/divisibility property belonging to
*this* problem that a quantum circuit could test, and this script does not
pretend otherwise: it does not invent an OEIS id or a fake "term" of a
nonexistent sequence. Per the task instructions for this case, what follows
is the best-effort honest substitute: a real, self-contained, small Grover
search circuit that computes a genuine, independently-checkable arithmetic
property (finding the unique x in a small finite universe satisfying a
stated numeric predicate), run on the ideal AerSimulator and checked against
a from-scratch classical computation. It is NOT a test of an Erdos-511
sequence -- there isn't one to test -- and this script reports that
accurately rather than claiming false verification against problem #511's
mathematics.

Substitute property actually computed (real math, not fabricated):
    Over the universe U = {0, 1, ..., 15} (4 qubits), find the unique x
    such that x is the only element of U whose value equals 3*x mod 16
    having x != 0, i.e. solve 3*x = x (mod 16) for x in [1, 15].
    3x = x (mod 16)  =>  2x = 0 (mod 16)  =>  x in {0, 8}.
    Restricting to x != 0 gives the unique nonzero solution x = 8.
    This is computed classically from first principles (brute force over
    all 16 values, no shortcut, no OEIS lookup) and then located with a
    real 4-qubit Grover search whose oracle marks exactly that x.

Run: python3 problem_511.py
Prints PASS if the quantum (Grover) result matches the classical answer,
FAIL otherwise.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np


N_QUBITS = 4
N = 2 ** N_QUBITS  # universe size, 16


def classical_answer():
    """Brute-force, from-scratch classical solution of 3x = x (mod 16), x != 0."""
    solutions = [x for x in range(N) if (3 * x) % N == x and x != 0]
    assert len(solutions) == 1, f"expected a unique nonzero solution, got {solutions}"
    return solutions[0]


def build_oracle(target: int, n_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle marking the computational basis state |target>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(target, f"0{n_qubits}b")[::-1]  # little-endian qubit order
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(target: int, n_qubits: int) -> QuantumCircuit:
    oracle = build_oracle(target, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    # optimal number of Grover iterations for 1 marked item out of 2^n_qubits
    n_items = 2 ** n_qubits
    iterations = max(1, round((np.pi / 4) * np.sqrt(n_items)))

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run_grover(target: int, n_qubits: int, shots: int = 2048) -> int:
    qc = build_grover_circuit(target, n_qubits)
    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    # most frequent measured bitstring, little-endian -> integer
    # Qiskit's returned bitstring already has qubit 0 as the rightmost
    # (least-significant) character, matching standard binary notation.
    best_bitstring = max(counts, key=counts.get)
    measured = int(best_bitstring, 2)
    return measured, counts


def main():
    classical = classical_answer()
    print(f"Classical answer (brute force over 0..{N - 1}): x = {classical}")

    measured, counts = run_grover(classical, N_QUBITS)
    total_shots = sum(counts.values())
    hit_fraction = counts.get(format(classical, f"0{N_QUBITS}b"), 0) / total_shots

    print(f"Quantum (Grover, ideal AerSimulator) most frequent result: x = {measured}")
    print(f"Fraction of shots landing on the classical answer: {hit_fraction:.3f}")
    print(f"Counts (top 5): {sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")

    verified = (measured == classical) and (hit_fraction > 0.5)

    print()
    print("NOTE: Erdos problem #511 has no OEIS sequence (oeis: ['N/A']), so this")
    print("circuit verifies a standalone arithmetic search property, not a term of")
    print("any sequence belonging to problem #511. See module docstring.")
    print()
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
