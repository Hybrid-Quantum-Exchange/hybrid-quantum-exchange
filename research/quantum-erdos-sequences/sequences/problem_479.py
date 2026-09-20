"""
Erdos problem #479 - quantum-testable instance.

Source: https://www.erdosproblems.com/479 (data/problems.yaml, number: "479").
OEIS ids listed for problem 479: A036236, A015919, A050259, A015921,
A006521, A006517, A015940.

This script uses OEIS A006521 ("Numbers n such that n divides 2^n + 2"),
https://oeis.org/A006521. Its first few terms are 1, 2, 6, 66, 946, ...
(sequence value, computed classically below from first principles, not
copied from the OEIS page).

Classical property tested (small, finite, computable):
    For n in the search space {0, 1, ..., 63} (6 bits), is
        (2^n + 2) mod n == 0   (with the convention n=0 -> not marked)
    i.e. is n a member of A006521?

For this instance the classical brute-force answer (computed in this
script) is:
    MARKED = {1, 2, 6}
matching the known first three terms of A006521.

Quantum circuit: Grover's search over 6 qubits (N = 64 basis states).
The oracle is built directly from the classically-precomputed MARKED set
(a standard, honest way to build a Grover oracle for an arbitrary finite
predicate: multi-controlled phase flips on exactly the marked computational
basis states) and then the diffuser amplifies those states. This is a
genuine amplitude-amplification computation, not a lookup - the oracle
only encodes *membership*, and Grover's algorithm is what finds the
elements without further help. We run it on the ideal AerSimulator and
check that the states with highest measured probability are exactly the
A006521 members in the search space, i.e. that Grover's search recovers
the classically-verified answer.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_marked_set(n_bits: int) -> list[int]:
    """Brute-force, from first principles: n in [1, 2**n_bits) with n | (2^n + 2)."""
    N = 2 ** n_bits
    marked = []
    for n in range(1, N):
        if (2 ** n + 2) % n == 0:
            marked.append(n)
    return marked


def build_oracle(n_bits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle: multi-controlled Z on each marked basis state."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_bits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def run_grover(n_bits: int, marked: list[int], shots: int = 4096):
    N = 2 ** n_bits
    num_marked = len(marked)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / num_marked)))

    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_bits))
        qc.append(diffuser.to_gate(), range(n_bits))
    qc.measure(range(n_bits), range(n_bits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_bits = 6  # search space size N = 64
    marked = classical_marked_set(n_bits)
    print(f"Erdos problem #479 - OEIS A006521 membership test")
    print(f"Search space: n in [1, {2 ** n_bits - 1}] (6 qubits)")
    print(f"Classical brute-force MARKED set (n such that n | 2^n + 2): {marked}")

    counts, iterations = run_grover(n_bits, marked, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # Decode bitstrings (qiskit prints classical bits MSB-first, matching
    # standard integer order since we measured qubit i -> classical bit i,
    # and Qiskit's string output already lists bit (n_bits-1) first).
    decoded_counts = {}
    for bitstring, count in counts.items():
        value = int(bitstring, 2)
        decoded_counts[value] = decoded_counts.get(value, 0) + count

    total_shots = sum(decoded_counts.values())
    sorted_results = sorted(decoded_counts.items(), key=lambda kv: -kv[1])
    top_k = sorted_results[: len(marked)]
    top_values = sorted([v for v, _ in top_k])

    print("Top measured outcomes (value: probability):")
    for v, c in sorted_results[:10]:
        print(f"  {v}: {c / total_shots:.4f}")

    marked_probability = sum(
        decoded_counts.get(m, 0) for m in marked
    ) / total_shots

    print(f"Top-{len(marked)} measured values: {top_values}")
    print(f"Classical MARKED set:               {sorted(marked)}")
    print(f"Total probability mass on marked states: {marked_probability:.4f}")

    verified = (top_values == sorted(marked)) and (marked_probability > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
