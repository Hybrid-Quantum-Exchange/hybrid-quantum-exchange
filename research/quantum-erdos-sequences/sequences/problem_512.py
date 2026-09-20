"""
Erdos problem #512 — quantum-testable-sequence lane.

Source metadata (from data/problems.yaml in the manman4/erdosproblems clone,
block "number: \"512\"", 2026-09-19 read):
    comments: "Littlewood's conjecture"
    tags: ["analysis"]
    oeis: ["N/A"]
    informal_status: proved (Lean), last_update 2026-06-22

LIMITATION (reported honestly, not papered over): problem #512 has NO OEIS
sequence id attached (oeis: ["N/A"]). Littlewood's conjecture is a statement
about liminf_{n->inf} n * ||n*alpha|| * ||n*beta|| = 0 for all real alpha,
beta — a continuous, infinitary statement over irrationals. There is no
finite/computable OEIS sequence here to build a genuine small quantum
circuit around, and inventing one would violate the "do not fabricate a
property" instruction. So per the task's own fallback clause, this script
is the best honest attempt: it does NOT test anything about problem #512's
actual mathematical content. It builds a real, genuine Grover search circuit
(oracle + diffuser, run on AerSimulator) for an unrelated but well-defined,
small, finite, classically-checkable arithmetic property -- "find the
integers n in [0, 7] with gcd(n, 6) == 1" -- purely so this lane still
exercises a real quantum circuit end to end. This substitute target is
clearly NOT derived from problem #512's sequence (there isn't one), and the
PASS/FAIL below should be read as "the generic Grover circuit machinery
works", not as any verification of Erdos problem #512.

verified_against_classical for this lane should be interpreted as: the
quantum circuit's output distribution matches the classical brute-force
answer for the substitute toy problem, NOT as evidence about Littlewood's
conjecture or any OEIS sequence (none exists for this problem).
"""

import itertools
from math import gcd, pi, sqrt

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_answer(n_bits: int) -> list[int]:
    """Brute-force integers n in [0, 2**n_bits - 1] with gcd(n, 6) == 1."""
    N = 2 ** n_bits
    return [n for n in range(N) if gcd(n, 6) == 1]


def build_oracle(n_bits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each n in `marked` (multi-controlled Z)."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_bits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def run_grover(n_bits: int, marked: list[int], shots: int = 4096):
    N = 2 ** n_bits
    M = len(marked)
    assert 0 < M < N

    # Optimal number of Grover iterations.
    theta = np.arcsin(sqrt(M / N))
    iterations = max(1, int(round((pi / 4) / theta - 0.5)))
    # For small N/M the round-off above can pick a suboptimal iteration
    # count; do an exact search over a small range for the value that
    # maximizes sin^2((2k+1)*theta), the true success probability formula.
    candidates = range(1, 6)
    iterations = max(
        candidates, key=lambda k: np.sin((2 * k + 1) * theta) ** 2
    )

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_bits), range(n_bits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_bits = 3  # search space N = 8, small enough for exact simulation
    marked = classical_answer(n_bits)
    print(f"Classical brute force over [0, {2**n_bits - 1}]:")
    print(f"  n with gcd(n, 6) == 1: {marked}")

    counts, iterations = run_grover(n_bits, marked, shots=4096)
    print(f"Grover iterations used: {iterations}")
    print("Measurement counts (bitstring -> count):")
    for bitstring, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {bitstring}: {c}")

    total_shots = sum(counts.values())
    marked_strs = {format(m, f"0{n_bits}b") for m in marked}
    marked_mass = sum(c for bs, c in counts.items() if bs in marked_strs)
    marked_fraction = marked_mass / total_shots

    # Success criterion: the measured distribution should be heavily
    # concentrated (> 90%) on the classically-marked states, and no
    # non-marked state should individually out-count every marked state.
    top_bitstring = max(counts, key=counts.get)
    top_is_marked = top_bitstring in marked_strs

    ok = marked_fraction > 0.90 and top_is_marked

    print(f"Fraction of shots landing on classically-marked states: {marked_fraction:.4f}")
    print(f"Top measured bitstring marked-correct: {top_is_marked}")

    if ok:
        print("PASS")
    else:
        print("FAIL")

    return ok


if __name__ == "__main__":
    import sys

    ok = main()
    sys.exit(0 if ok else 1)
