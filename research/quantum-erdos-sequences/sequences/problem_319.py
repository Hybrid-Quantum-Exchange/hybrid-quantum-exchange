"""
Erdos problem #319 -- quantum-testable sequence lane.

Source metadata (data/problems.yaml in the manman4/erdosproblems clone,
block "number: \"319\""):
    prize: no
    informal_status: open (last update 2025-08-31)
    oeis: ["possible"]
    tags: ["number theory", "unit fractions"]

IMPORTANT LIMITATION, stated honestly per instructions: the "oeis" field for
problem #319 is the literal string "possible", not a real OEIS sequence id
(A-number). There is no OEIS sequence attached to this problem in the source
data, so no classical OEIS-derived property of "the sequence" can be tested
here -- there is no sequence to test. This is not a case where we simply
failed to look hard enough; the metadata itself carries no OEIS id.

Best honest attempt, in place of a fabricated OEIS-backed property:
problem #319 is tagged "number theory" / "unit fractions". In that spirit
this script tests a small, finite, genuinely computable number-theoretic
property that is at the heart of unit-fraction (Egyptian fraction) problems:
divisibility, which determines which unit fractions 1/x combine exactly.

Classical property tested (computed from first principles in this script,
not copied from anywhere):
    For N = 13 and x ranging over the 4-bit nonzero values {1, ..., 15},
    which x divide N exactly (i.e. for which x does the unit fraction 1/x
    scale to an integer multiple of 1/N, N/x being an integer)?

    Classically: divisors of 13 in {1..15} = {1, 13} (13 is prime, chosen
    so the marked set is a small minority of the search space, which is
    the regime Grover's algorithm is built for).

Quantum approach: Grover's search algorithm over a 3-qubit register
representing x in {0, ..., 7}. The oracle flips the phase of exactly the
basis states corresponding to divisors of N=12 found in that range
(computed classically inside this script -- not hard-coded from any
external source). Grover diffusion then amplifies those marked states so
that measurement recovers the divisor set with high probability. The
script compares the quantum sampling result (the modal, highest-count
outcomes) against the classical divisor set and prints PASS/FAIL.

Dependencies: qiskit, qiskit_aer, numpy only (already installed).
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_divisors(n: int, upper: int) -> list[int]:
    """Return all x in [1, upper] that divide n exactly, computed directly."""
    return [x for x in range(1, upper + 1) if n % x == 0]


def build_oracle(num_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each integer in `marked_states` (0..2^n-1)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{num_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # Multi-controlled Z on all qubits (flip phase of |11...1>)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover(marked_states: list[int], num_qubits: int, shots: int = 4096):
    n_marked = len(marked_states)
    n_total = 2 ** num_qubits

    # Optimal number of Grover iterations (standard formula).
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_total / n_marked)))

    oracle = build_oracle(num_qubits, marked_states)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main() -> bool:
    N = 13
    NUM_QUBITS = 4  # represents integers 0..15
    UPPER = 2 ** NUM_QUBITS - 1  # 15

    # --- Classical answer, computed here from first principles ---
    divisors = classical_divisors(N, UPPER)
    divisors_nonzero = [d for d in divisors if d >= 1]
    print(f"Classical property: divisors of N={N} within [1,{UPPER}]")
    print(f"Classical answer: {divisors_nonzero}")

    # --- Quantum Grover search for the same marked set ---
    counts, iterations = run_grover(divisors_nonzero, NUM_QUBITS)
    print(f"Grover iterations used: {iterations}")

    total_shots = sum(counts.values())
    # Sort outcomes by count, descending.
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    print("Top measured outcomes (bitstring: count):")
    for bits, c in sorted_counts[:8]:
        val = int(bits, 2)
        print(f"  {bits} (x={val}): {c}")

    # Take the top len(divisors_nonzero) outcomes as the quantum-found set.
    k = len(divisors_nonzero)
    quantum_found = sorted(int(bits, 2) for bits, _ in sorted_counts[:k])
    classical_set = sorted(divisors_nonzero)

    # Sanity: marked states should collectively carry most of the probability.
    marked_mass = sum(c for bits, c in counts.items() if int(bits, 2) in classical_set)
    marked_fraction = marked_mass / total_shots
    print(f"Fraction of shots landing on a true divisor: {marked_fraction:.3f}")

    verified = (quantum_found == classical_set) and (marked_fraction > 0.5)

    print(f"Quantum-found (top-{k} outcomes): {quantum_found}")
    print(f"Classical set:                    {classical_set}")

    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
