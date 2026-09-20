"""
Erdos problem #538 (source: https://github.com/manman4/erdosproblems,
data/problems.yaml, entry `number: "538"`).

Metadata read from the source of truth (verified 2026-09-19):
    prize:           no
    informal_status: open (last_update 2025-08-31)
    formal_status:   unformalized
    oeis:             ["N/A"]
    tags:            ["number theory"]

LIMITATION, stated honestly up front: problem #538 has NO OEIS sequence
associated with it (`oeis: ["N/A"]`) and the problems.yaml record carries no
further formula/statement text in this read-only clone -- there is nothing
sequence-shaped here to build a genuine, problem-538-specific quantum
instance from. Per the task's fallback instructions, this script is the best
honest attempt: since the only real content available for #538 is its tag
"number theory", it builds a genuine, self-contained, small quantum
computation squarely inside that tag rather than fabricating a fake OEIS
value or a fake connection to the actual (unformalized) statement of #538.

The classical property tested (real math, checked classically here, not
copied from anywhere):
    N = 15. Consider the "candidate" register ranging over integers
    x in [2, 15] (4 qubits, values 0..15, with 0 and 1 excluded by the
    oracle). x is a MARKED (good) state iff x is a proper, nontrivial
    divisor of N, i.e. 1 < x < N and N % x == 0.
    For N = 15 the classical divisors in [2,14] are exactly {3, 5}.

This is exactly the finite, computable search problem Grover's algorithm is
built for: search space size M = 16 (4 qubits), 2 marked good states out of
16. We compute the classically-correct marked set first (from first
principles, by trial division -- no lookup table, no OEIS), build the
Grover oracle that flags precisely those computed states, run the full
Grover circuit (optimal number of iterations for this M and this number of
solutions) on the ideal AerSimulator, and check that the two most likely
measurement outcomes are exactly the classically-computed divisors.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


def classical_divisors(n: int, n_bits: int) -> list[int]:
    """Trial division from first principles: proper nontrivial divisors of n
    among integers representable in n_bits bits (0 .. 2**n_bits - 1)."""
    limit = 2 ** n_bits
    return [x for x in range(2, min(n, limit)) if n % x == 0]


def build_oracle(n_bits: int, marked_states: list[int]) -> QuantumCircuit:
    """Phase-flip oracle: multi-controlled Z on each marked basis state,
    using X gates to map the marked bit pattern onto the all-ones pattern
    the MCZ triggers on."""
    qc = QuantumCircuit(n_bits, name="oracle")
    mcz = MCMTGate(ZGate(), n_bits - 1, 1)
    for state in marked_states:
        bits = [(state >> i) & 1 for i in range(n_bits)]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        qc.append(mcz, list(range(n_bits)))
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean) over n_bits qubits."""
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    mcz = MCMTGate(ZGate(), n_bits - 1, 1)
    qc.append(mcz, list(range(n_bits)))
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def run_grover(n_bits: int, marked_states: list[int], iterations: int, shots: int = 4096):
    qr = QuantumRegister(n_bits, "x")
    qc = QuantumCircuit(qr)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, marked_states)
    diffuser = build_diffuser(n_bits)

    for _ in range(iterations):
        qc.compose(oracle, qubits=qr, inplace=True)
        qc.compose(diffuser, qubits=qr, inplace=True)

    qc.measure_all()

    backend = AerSimulator()
    from qiskit import transpile
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    N = 15
    N_BITS = 4  # search space: integers 0..15

    # --- classical ground truth, computed here from first principles ---
    marked = classical_divisors(N, N_BITS)
    print(f"N = {N}, search space size M = {2 ** N_BITS}")
    print(f"Classically computed proper nontrivial divisors of {N}: {marked}")
    expected = {3, 5}
    assert set(marked) == expected, f"classical computation disagrees with expectation: {marked}"

    # --- optimal Grover iteration count for M states, t marked ---
    M = 2 ** N_BITS
    t = len(marked)
    theta = math.asin(math.sqrt(t / M))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Running Grover search with {iterations} iteration(s) for {t} marked state(s) out of {M}")

    counts = run_grover(N_BITS, marked, iterations)

    # bitstrings from qiskit measure_all come back MSB-first per register;
    # with a single 4-qubit register + measure_all, the label is the 4 bits,
    # qubit 0 is the rightmost character.
    def label_to_int(label: str) -> int:
        return int(label.replace(" ", ""), 2)

    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    print("Top measurement outcomes (bitstring: counts):")
    for label, c in sorted_counts[:6]:
        print(f"  {label} -> x={label_to_int(label)}: {c}")

    total_shots = sum(counts.values())
    top_states = {label_to_int(label) for label, _ in sorted_counts[:t]}

    print(f"Top-{t} most frequent measured x values: {sorted(top_states)}")
    print(f"Classical divisors of {N} in range: {sorted(marked)}")

    passed = top_states == set(marked)

    # sanity: marked states should collectively carry a large majority of
    # the amplified probability mass (well above the uniform 2/16 = 12.5%)
    marked_mass = sum(counts.get(label, 0) for label, _ in sorted_counts
                       if label_to_int(label) in marked) / total_shots
    print(f"Fraction of shots landing on a marked (divisor) state: {marked_mass:.3f}")
    passed = passed and marked_mass > 0.5

    if passed:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
