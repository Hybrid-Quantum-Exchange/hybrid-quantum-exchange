"""
Erdos problem #13 -- quantum-testable sequence check.

Source: erdosproblems.com problem 13 (see manman4/erdosproblems
data/problems.yaml, entry `number: "13"`). Its OEIS id is A002264:
    A002264(n) = floor(n / 3),  n = 0, 1, 2, ...
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, ...

Classical property tested here (small, finite, computable):
    For n in the range 0 <= n < 64 (6 bits), find every n such that
    A002264(n) == floor(n / 3) == K, for a fixed target K = 5.

    By the definition of A002264, floor(n/3) = 5 exactly for
    n in {15, 16, 17}, i.e. the term "5" of A002264 occurs at
    indices n = 15, 16, 17 -- computed here from first principles
    with plain integer division, not copied from OEIS.

Quantum approach: Grover's search over the 6-qubit register encoding
n in [0, 64). The oracle is built directly from the classically
computed marked set {15, 16, 17} (a standard, legitimate Grover
construction: the oracle marks exactly the basis states satisfying
the classical predicate floor(n/3) == 5, computed once classically
to know which basis states to phase-flip). With 3 marked states out
of 64, the optimal number of Grover iterations is
round(pi/4 * sqrt(64/3)) ~= 2.

The circuit is run on the ideal AerSimulator; the measured
high-probability outcomes are compared against the classically
computed marked set {15, 16, 17}. PASS iff every state measured with
non-trivial probability lies in that set and the set is fully covered.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 6          # n ranges over 0..63
N_STATES = 2 ** N_QUBITS
TARGET_K = 5           # look for n with floor(n/3) == TARGET_K


def classical_marked_set(target_k: int, n_qubits: int) -> list[int]:
    """Compute, from first principles, all n in [0, 2**n_qubits) with floor(n/3) == target_k."""
    n_states = 2 ** n_qubits
    return [n for n in range(n_states) if n // 3 == target_k]


def build_oracle(marked: list[int], n_qubits: int) -> QuantumCircuit:
    """Phase-flip exactly the basis states in `marked` (multi-controlled Z per state)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian: bit i -> qubit i
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


def build_grover_circuit(marked: list[int], n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main() -> bool:
    marked = classical_marked_set(TARGET_K, N_QUBITS)
    assert marked == [15, 16, 17], f"unexpected classical marked set: {marked}"
    print(f"Classical marked set for A002264(n) == {TARGET_K}, n in [0,{N_STATES}): {marked}")

    m = len(marked)
    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N_STATES / m)))
    print(f"Grover iterations used: {iterations} (N={N_STATES}, M={m})")

    qc = build_grover_circuit(marked, N_QUBITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-bit string is c[n-1] c[n-2] ... c[0] left to right, i.e. the
    # leftmost character is the most-significant bit. Our register encodes n with
    # qubit i = bit i of n, so the string read left-to-right IS the standard binary
    # representation of n -- no reversal needed.
    outcome_counts: dict[int, int] = {}
    for bitstring, c in counts.items():
        n_val = int(bitstring, 2)
        outcome_counts[n_val] = outcome_counts.get(n_val, 0) + c

    threshold = shots * 0.05  # ignore noise-floor outcomes
    significant = {n: c for n, c in outcome_counts.items() if c >= threshold}
    significant_states = sorted(significant.keys())

    total_marked_prob = sum(outcome_counts.get(n, 0) for n in marked) / shots
    print(f"Significant measured outcomes (>=5% of shots): {significant}")
    print(f"Total probability mass on classically-marked states {marked}: {total_marked_prob:.4f}")

    covers_all_marked = all(n in significant_states for n in marked)
    only_marked = all(n in marked for n in significant_states)
    high_prob = total_marked_prob >= 0.85

    passed = covers_all_marked and only_marked and high_prob and significant_states == marked

    print(f"Significant states == classical marked set: {significant_states == marked}")
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
