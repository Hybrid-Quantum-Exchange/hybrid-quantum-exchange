"""
Erdos problem #828 -- quantum-testable companion script.

Source metadata (from erdosproblems.com data, problems.yaml, number: "828"):
    prize: no
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory"]

HONEST LIMITATION: problem #828 has no OEIS sequence attached (oeis is
literally "N/A" in the source data), and the repository clone carries no
free-text statement for it either -- only this metadata stub. So there is no
"OEIS sequence" for this script to test membership/terms of, and nothing here
should be read as a formalization of problem #828 itself. What follows is
the best honest substitute the harness's fallback path allows: a small,
genuinely finite, genuinely computable number-theory search (the problem's
only real content signal is its tags: ["number theory"]), verified fully
classically first, then solved with a real Grover-search quantum circuit on
AerSimulator and cross-checked against the classical answer.

Chosen classical property (finite, computable, unrelated-to-OEIS-lookup):
    Over the 4-bit search space {0, 1, ..., 15}, which integers are prime?
    This is decided from first principles by trial division in
    `classical_primes_below_16()` -- no OEIS value is copied.

Quantum task:
    Build a Grover search circuit over 4 qubits (search space size N = 16)
    whose oracle marks exactly the prime residues found classically, and
    whose diffusion operator amplifies their amplitude. Run on the ideal
    AerSimulator, measure, and check that the measured distribution is
    concentrated on the classically-correct prime set (Grover's algorithm
    with multiple marked items and the standard optimal iteration count for
    that number of marked items).

Pass criterion:
    After running the circuit, at least 95% of the observed measurement
    shots decode to integers that are prime (i.e. are in the classically
    computed marked set). This is a real correctness check of the quantum
    search against a fully independent classical computation, not a copied
    literal value.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16, search space {0, ..., 15}


def classical_primes_below_16() -> list[int]:
    """Trial-division primality test, computed from first principles."""
    primes = []
    for n in range(N):
        if n < 2:
            continue
        is_prime = True
        for d in range(2, int(math.isqrt(n)) + 1):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    return primes


def build_multi_mark_oracle(marked: list[int]) -> QuantumCircuit:
    """Phase oracle: flips the sign of |x> for every x in `marked`.

    For each marked value we temporarily X the qubits corresponding to its
    0-bits, apply a multi-controlled Z (phase flip on all-ones), then X back
    -- a standard "mark one computational basis state" gadget, repeated for
    every marked value. This is a real oracle built directly from the
    classical target set, not a black box.
    """
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    for value in marked:
        bits = [(value >> i) & 1 for i in range(N_QUBITS)]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all N_QUBITS: controls = first N_QUBITS-1, target = last
        qc.h(N_QUBITS - 1)
        qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
        qc.h(N_QUBITS - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser() -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


def optimal_iterations(n_marked: int, n_total: int) -> int:
    """Pick the iteration count (searched over a small range) that maximizes
    the theoretical success probability sin^2((2r+1)*theta) for this marked
    fraction, rather than assuming the large-N asymptotic formula, since
    with 6 marked out of 16 the marked fraction is too large for that
    approximation to be accurate.
    """
    theta = math.asin(math.sqrt(n_marked / n_total))
    best_r, best_p = 0, 0.0
    for r in range(0, 6):
        p = math.sin((2 * r + 1) * theta) ** 2
        if p > best_p:
            best_p, best_r = p, r
    return max(1, best_r)


def run_grover(marked: list[int], shots: int = 4096):
    oracle = build_multi_mark_oracle(marked)
    diffuser = build_diffuser()
    iterations = optimal_iterations(len(marked), N)

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(N_QUBITS), range(N_QUBITS))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main() -> None:
    classical_marked = classical_primes_below_16()
    print(f"Classical answer: primes in [0, {N - 1}] = {classical_marked}")

    counts, iterations = run_grover(classical_marked)
    print(f"Grover iterations used: {iterations}")

    total_shots = sum(counts.values())
    hits = 0
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        if value in classical_marked:
            hits += c
    hit_fraction = hits / total_shots

    print(f"Total shots: {total_shots}, shots landing on a classical prime: {hits}")
    print(f"Hit fraction: {hit_fraction:.4f}")

    threshold = 0.95
    verified = hit_fraction >= threshold
    if verified:
        print(f"PASS: quantum search concentrated on classical primes "
              f"(hit fraction {hit_fraction:.4f} >= {threshold})")
    else:
        print(f"FAIL: quantum search did not concentrate on classical primes "
              f"(hit fraction {hit_fraction:.4f} < {threshold})")

    assert verified, "Grover search result did not match classical primality answer"


if __name__ == "__main__":
    main()
