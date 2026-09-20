"""
Erdos problem #982 -- quantum-testable sequence lane.

Erdos problem #982 (see erdosproblems.com/982, tags: geometry, convex,
distances) is linked in the problems.yaml metadata to OEIS sequence
A004526, "the quarter-squares sequence" a.k.a. the simplest form of it:
    A004526(n) = floor(n / 2)   for n = 0, 1, 2, 3, ...
    (0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, ...)

The problem's own statement is a hard open geometric question about convex
sets and distances, well beyond what a few-qubit circuit can decide. What
*is* small, finite, and genuinely computable is the defining arithmetic
property of the OEIS sequence attached to it: membership in the graph of
A004526, i.e.

    CLASSICAL PROPERTY TESTED:
        For n in {0, 1, ..., 15} (4 bits), find all n such that
        floor(n / 2) == TARGET, for a chosen small TARGET (here TARGET = 5).

    floor(n / 2) is exactly "n with its least-significant bit dropped",
    i.e. the top 3 bits of a 4-bit n. So floor(n/2) == 5 (binary 101) iff
    n's top 3 bits equal 101, i.e. n in {10, 11} (binary 1010, 1011).

This is computed from first principles classically in `classical_A004526`
and `classical_search`, and then verified with a real Grover search
circuit built in Qiskit: the oracle marks exactly the basis states |n>
(4 qubits) whose top 3 bits equal the binary encoding of TARGET, which is
precisely the set {n : floor(n/2) == TARGET}. Grover amplification is run
for the optimal number of iterations and the circuit is measured on the
ideal AerSimulator; the most frequent measured outcomes must equal the
classically-computed solution set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_BITS = 4          # n ranges over 0..15
TARGET = 5           # floor(n/2) == TARGET is the property under test


def classical_A004526(n: int) -> int:
    """A004526(n) = floor(n / 2), computed directly (no OEIS lookup)."""
    return n // 2


def classical_search(target: int, n_bits: int):
    """Brute-force, from first principles, all n in [0, 2**n_bits) with
    floor(n/2) == target. This is the ground truth the quantum circuit
    is checked against."""
    space = range(2 ** n_bits)
    return sorted(n for n in space if classical_A004526(n) == target)


def build_oracle(n_bits: int, target: int) -> QuantumCircuit:
    """Phase oracle marking basis states |n> (n_bits qubits, little-endian
    Qiskit convention: qubit 0 is the least-significant bit) for which
    floor(n / 2) == target, i.e. bits [1 .. n_bits-1] of n equal the
    binary expansion of target (using n_bits - 1 bits). Qubit 0 (the
    dropped LSB) is a don't-care, exactly matching floor division by 2.
    """
    qc = QuantumCircuit(n_bits, name="oracle")
    high_bits = n_bits - 1
    target_bits = [(target >> i) & 1 for i in range(high_bits)]  # LSB-first over qubits 1..n_bits-1

    # Flip qubits whose target bit is 0, so that "all high qubits == |1>"
    # exactly captures "high bits == target".
    flip_qubits = [1 + i for i in range(high_bits) if target_bits[i] == 0]
    for q in flip_qubits:
        qc.x(q)

    high_qubit_indices = list(range(1, n_bits))
    if len(high_qubit_indices) == 1:
        qc.z(high_qubit_indices[0])
    else:
        # Multi-controlled Z on the high qubits: H, MCX, H trick on the
        # last high qubit, controlled by the rest.
        target_q = high_qubit_indices[-1]
        controls = high_qubit_indices[:-1]
        qc.h(target_q)
        qc.append(MCXGate(len(controls)), controls + [target_q])
        qc.h(target_q)

    for q in flip_qubits:
        qc.x(q)

    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    else:
        target_q = n_bits - 1
        controls = list(range(n_bits - 1))
        qc.h(target_q)
        qc.append(MCXGate(len(controls)), controls + [target_q])
        qc.h(target_q)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def build_grover_circuit(n_bits: int, target: int, n_solutions: int, n_iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, target)
    diffuser = build_diffuser(n_bits)

    for _ in range(n_iterations):
        qc.append(oracle.to_gate(), range(n_bits))
        qc.append(diffuser.to_gate(), range(n_bits))

    qc.measure(range(n_bits), range(n_bits))
    return qc


def main():
    classical_solutions = classical_search(TARGET, N_BITS)
    n_solutions = len(classical_solutions)
    space_size = 2 ** N_BITS

    print(f"Erdos problem #982 -> OEIS A004526 (floor(n/2)), N_BITS={N_BITS}, TARGET={TARGET}")
    print(f"Classical search space: n in [0, {space_size})")
    print(f"Classical solutions (floor(n/2) == {TARGET}): {classical_solutions}")

    # Optimal Grover iteration count for n_solutions out of space_size.
    n_iterations = max(1, round((math.pi / 4) * math.sqrt(space_size / n_solutions)))
    print(f"Grover iterations used: {n_iterations}")

    qc = build_grover_circuit(N_BITS, TARGET, n_solutions, n_iterations)

    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    shots = 4096
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bitstrings are big-endian in the printed key (qubit n-1 ... qubit 0),
    # which matches standard binary integer reading, since qubit 0 is bit 0.
    int_counts = Counter()
    for bitstring, c in counts.items():
        n_val = int(bitstring, 2)
        int_counts[n_val] += c

    # Take the top `n_solutions` most frequent measured values as the
    # quantum circuit's answer.
    most_common = [n for n, _ in int_counts.most_common(n_solutions)]
    quantum_solutions = sorted(most_common)

    total_solution_shots = sum(c for n, c in int_counts.items() if n in classical_solutions)
    success_rate = total_solution_shots / shots

    print(f"Measured counts (top): {int_counts.most_common(6)}")
    print(f"Quantum-found solutions (top {n_solutions} most frequent): {quantum_solutions}")
    print(f"Fraction of shots landing on a true solution: {success_rate:.3f}")

    verified = (quantum_solutions == classical_solutions) and (success_rate > 0.8)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
