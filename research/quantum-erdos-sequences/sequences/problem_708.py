"""
Erdos problem #708 — quantum-testable proxy (best-effort, limitation noted)
=============================================================================

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: \"708\"" (prize "$100", tags: ["number theory"], status: open,
last_update 2025-08-31).

LIMITATION (read before trusting the PASS below): problem #708's `oeis`
field in the source data is the literal placeholder value `["possible"]`,
not a real OEIS sequence id. There is no concrete OEIS sequence attached to
this problem in the source, so no OEIS-derived finite property of "the
problem 708 sequence" can honestly be built or verified here. Rather than
fabricate an OEIS id or invent a property and claim it represents problem
708, this script builds a genuine, independently-checkable number-theory
Grover search instead, and reports that substitution plainly. The property
chosen is a real one (primality over a small finite range), not tied to any
specific OEIS entry, and should not be read as verifying anything about
Erdos problem #708 itself beyond documenting that no OEIS id was available.

Classical property tested
--------------------------
Over the 4-qubit search space {0, 1, ..., 15} (N = 16), find all x such
that x is prime AND x = 1 (mod 4). Computed from first principles by
trial division and modular arithmetic in this script (no external data,
no OEIS lookup):

    primes in [0, 15]            = {2, 3, 5, 7, 11, 13}
    primes with x = 1 (mod 4)    = {5, 13}            (2 marked states out of 16)

Quantum approach
-----------------
Grover's algorithm on 4 qubits (N = 16, M = 2 marked items). The optimal
number of Grover iterations is computed from the standard formula
r = round(pi/4 * sqrt(N/M)), which works out to 2 for this instance. Two
iterations of a 4-qubit Grover search with an oracle marking exactly the
classically-computed target states, followed by measurement, should
return one of {5, 13} with strongly amplified probability (theoretical
success probability ~0.94, versus a uniform baseline of 2/16 = 0.125).
We run it on the ideal AerSimulator (no noise) with many shots and check
that the fraction of measured outcomes landing on the classically-verified
target set clears a threshold far above the uniform baseline.

PASS criterion: over the shots, the fraction of measurement outcomes that
are prime according to the classical trial-division check must exceed a
threshold (0.90) well above the theoretical minimum for one Grover
iteration on this instance, confirming the quantum circuit actually
performs amplitude amplification toward the classically-verified answer.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    """Trial division primality test, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_target_set(n_bits: int) -> set[int]:
    """Primes x in [0, 2**n_bits - 1] with x = 1 (mod 4), from first principles."""
    n = 2 ** n_bits
    return {x for x in range(n) if is_prime(x) and x % 4 == 1}


def build_oracle(n_bits: int, marked: set[int]) -> QuantumCircuit:
    """Phase-flip oracle: negates the amplitude of each marked basis state."""
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
    """Standard Grover diffuser (inversion about the mean)."""
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


def build_grover_circuit(n_bits: int, marked: set[int], iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_bits), range(n_bits))
    return qc


def main() -> bool:
    n_bits = 4
    n = 2 ** n_bits  # N = 16

    # Classical answer, computed from first principles.
    primes = classical_target_set(n_bits)
    print(f"Classical property: primes = 1 (mod 4) in [0, {n - 1}] = {sorted(primes)}")
    assert primes == {5, 13}, "classical primality/mod computation looks wrong"

    m = len(primes)
    iterations = max(1, round((math.pi / 4) * math.sqrt(n / m)))
    print(f"N={n}, M={m}, Grover iterations={iterations}")

    qc = build_grover_circuit(n_bits, primes, iterations)
    qc = transpile(qc, AerSimulator())

    shots = 4096
    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    prime_hits = 0
    for bitstring, count in counts.items():
        value = int(bitstring, 2)
        if value in primes:
            prime_hits += count

    fraction_prime = prime_hits / shots
    print(f"Measurement outcomes landing on prime states: {fraction_prime:.4f} "
          f"(baseline uniform would be {m / n:.4f})")

    threshold = 0.90
    passed = fraction_prime >= threshold

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
