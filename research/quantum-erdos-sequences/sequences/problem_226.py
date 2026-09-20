"""
Erdos problem #226 -- quantum-testable lane (best-honest-effort fallback).

LIMITATION (read first): Erdos problem #226, as recorded in
manman4/erdosproblems/data/problems.yaml (entry "number: \"226\"", tags:
["analysis"]), lists oeis: ["N/A"]. There is no OEIS sequence attached to
this problem in the source data, and the data file carries no statement
text -- only status/metadata fields (prize, informal_status, formal_status,
tags). With no OEIS id and no problem statement to derive a finite,
sequence-specific computable property from, the task's own instructions say
to write the script anyway with a best honest attempt, note the limitation,
and report accurately rather than fabricate a property with no real
mathematical content.

So this script does NOT test anything specific to problem #226's actual
mathematical content (there is none finite/computable available to us here).
Instead, as the closest honest substitute, it runs a genuine, independently
checkable finite quantum computation: Grover's algorithm searching the
search space N = 16 (4 qubits) for the marked set S = {primes in [0, 15]}
= {2, 3, 5, 7, 11, 13}, a small, finite, exactly-computable classical
property (primality of a 4-bit integer). The classical answer is computed
from first principles by trial division in this script, independently of
any table lookup, and compared against the empirical distribution produced
by the real Grover circuit on Qiskit's ideal AerSimulator.

verified_against_classical: this script DOES verify a real quantum result
against an independently computed classical answer, but that answer is NOT
derived from problem #226's own sequence (none exists in the source data).
ran_ok reflects the script itself; the problem-specific claim does not hold.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N_QUBITS = 4
N = 2 ** N_QUBITS  # 16


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


MARKED = sorted(x for x in range(N) if is_prime(x))
assert MARKED == [2, 3, 5, 7, 11, 13], MARKED
M = len(MARKED)


# ---------------------------------------------------------------------------
# 2. Grover oracle and diffuser for this marked set, built from primitives.
# ---------------------------------------------------------------------------

def build_oracle(marked: list[int], n_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle: multi-controlled Z on each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
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


def optimal_grover_iterations(n_total: int, n_marked: int) -> int:
    theta = math.asin(math.sqrt(n_marked / n_total))
    iters = round((math.pi / (4 * theta)) - 0.5)
    return max(1, iters)


def build_grover_circuit(marked: list[int], n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

def main() -> bool:
    iterations = optimal_grover_iterations(N, M)
    circuit = build_grover_circuit(MARKED, N_QUBITS, iterations)

    sim = AerSimulator(method="statevector")
    compiled = transpile(circuit, sim)
    shots = 4096
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's counts keys already parse directly as the integer whose bit i
    # corresponds to qubit i (verified against Statevector.probabilities_dict
    # for this same oracle construction), so no bit-reversal is needed.
    measured_ints = {}
    for bitstring, count in counts.items():
        value = int(bitstring, 2)
        measured_ints[value] = measured_ints.get(value, 0) + count

    marked_hits = sum(c for v, c in measured_ints.items() if v in MARKED)
    marked_fraction = marked_hits / shots

    top_values = sorted(measured_ints.items(), key=lambda kv: -kv[1])[:M]
    top_set = {v for v, _ in top_values}

    print(f"Erdos problem 226: no OEIS id available (oeis: ['N/A']) -- fallback lane.")
    print(f"Search space N={N} (4 qubits); classical marked set (primes): {MARKED}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measured counts (as integers): {dict(sorted(measured_ints.items()))}")
    print(f"Fraction of shots landing on a marked (prime) state: {marked_fraction:.3f}")
    print(f"Top-{M} most frequent measured values: {sorted(top_set)}")

    # Success criteria: amplification worked (most probability mass is on
    # marked states) AND the top-M most frequent outcomes are exactly the
    # classically computed marked set.
    amplified = marked_fraction > 0.80
    exact_match = top_set == set(MARKED)
    passed = amplified and exact_match

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
