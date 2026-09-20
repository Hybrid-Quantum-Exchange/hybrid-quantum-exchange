"""
Erdos problem #709 -- quantum-testable lane (best-effort, limitation noted).

Source record: erdosproblems.com data file (data/problems.yaml, entry
`number: "709"`, tags: ["number theory"], prize: "no", status: open,
last_update 2025-08-31). That entry's `oeis` field is `["possible"]` --
this is a placeholder string in the source data, NOT a real OEIS sequence
id (a real id looks like "A000040"). There is therefore no genuine OEIS
sequence attached to problem #709 to build a faithful quantum-testable
property from, and the problem's own statement is not in this metadata
file (only status/tag metadata is). This script does NOT fabricate an
OEIS id or invent a property and claim it represents #709's actual content.

Honest fallback, clearly scoped as such: problem #709 is tagged
"number theory", so this script builds a real, self-contained,
finite/computable number-theory property -- primality over a small
finite search space -- and verifies it with a genuine Grover search
circuit on Qiskit's ideal AerSimulator. This models the *kind* of
finite decision problem number-theory Erdos problems often reduce to,
it is not a derivation of #709 itself.

Classical property under test:
    For N = 16 (4 qubits, search space {0, ..., 15}), let
    PRIMES = { n in [0, 15] : n is prime }.
    This is computed classically in `classical_primes()` below by
    trial division from first principles (no lookup table).
    PRIMES = {2, 3, 5, 7, 11, 13}  (6 of 16 elements).

Quantum method:
    Grover's algorithm with an oracle built directly from the classical
    membership set (a multi-controlled-Z marking each prime's basis
    state), amplifying the marked amplitudes over ~2 optimal Grover
    iterations for M=6, N=16. The circuit is run on AerSimulator with
    many shots; the measured outcome distribution should concentrate on
    exactly the classically-computed PRIMES set.

Pass criterion:
    The set of the k most-frequently measured 4-bit outcomes (k = |PRIMES|)
    from the quantum run must equal classical_primes() exactly, AND the
    total measured probability mass on PRIMES must exceed a threshold
    that is far above the 6/16 baseline of uniform random guessing.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import sys
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16


def classical_primes(limit: int) -> set[int]:
    """Compute primes in [0, limit) by trial division, from first principles."""
    primes = set()
    for n in range(2, limit):
        is_p = True
        for d in range(2, int(n ** 0.5) + 1):
            if n % d == 0:
                is_p = False
                break
        if is_p:
            primes.add(n)
    return primes


def oracle_circuit(marked: set[int], n_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle: applies -1 to each basis state in `marked`."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        # Flip qubits that should be 0 so the controlled-Z condition is "all 1"
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def diffuser_circuit(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked: set[int], n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = oracle_circuit(marked, n_qubits)
    diffuser = diffuser_circuit(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main() -> int:
    primes = classical_primes(N)
    m = len(primes)
    print(f"Classical PRIMES in [0,{N}) via trial division: {sorted(primes)} (M={m})")

    # Optimal number of Grover iterations for M marked out of N.
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / m) - 0.5))
    print(f"Using {iterations} Grover iteration(s) for N={N}, M={m}")

    qc = build_grover_circuit(primes, N_QUBITS, iterations)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 20000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-register bitstring is written MSB..LSB with
    # qubit (n_qubits-1) leftmost and qubit 0 rightmost, so parsing it
    # directly as binary already yields the integer value with qubit 0
    # as the least-significant bit -- matching classical_primes()'s
    # convention and the statevector index convention.
    value_counts: dict[int, int] = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        value_counts[value] = value_counts.get(value, 0) + c

    ranked = sorted(value_counts.items(), key=lambda kv: -kv[1])
    top_k = set(v for v, _ in ranked[:m])
    mass_on_primes = sum(c for v, c in value_counts.items() if v in primes) / shots

    print(f"Top-{m} measured outcomes: {sorted(top_k)}")
    print(f"Probability mass on classical PRIMES set: {mass_on_primes:.4f}")

    baseline = m / N
    threshold = baseline + 0.5 * (1 - baseline)  # comfortably above uniform-random baseline

    matches_set = (top_k == primes)
    exceeds_threshold = (mass_on_primes > threshold)
    verified = matches_set and exceeds_threshold

    print(f"Set match: {matches_set}  |  Mass {mass_on_primes:.4f} > threshold {threshold:.4f}: {exceeds_threshold}")

    if verified:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
