"""
Erdos problem #440 -- quantum-testable instance.

LIMITATION (read first): problem #440's entry in erdosproblems/data/problems.yaml
has `oeis: ["N/A"]` -- no OEIS sequence is associated with this problem, and no
per-problem description file exists in that repository either. Its only other
metadata is `tags: ["number theory"]` and `informal_status: solved`. There is
therefore no genuine "OEIS sequence membership/term" property to derive for
this problem specifically, and fabricating one would violate the task's own
rule against inventing unfounded properties.

Rather than fake a property tied to problem 440, this script honestly falls
back to the one concrete, finite, computable number-theory property that the
problem's tag does support in general: PRIMALITY on a small finite domain.
This is a best-effort placeholder instance, not a claim that it encodes
anything specific proved/disproved by Erdos problem 440.

Classical property tested: for N = 16 (4-bit integers 0..15), which integers
n are prime? Ground truth is computed here from first principles (trial
division), independently of any table lookup.

Quantum approach: Grover's algorithm. We build a quantum oracle over 4 qubits
that marks exactly the primes in [0, 15] (2, 3, 5, 7, 11, 13 -- 6 marked
states out of 16), phase-flipping those basis states via multi-controlled Z
gates addressing their exact bit patterns, then run the standard number of
Grover diffusion iterations for M=6 marked items out of N=16 on the ideal
AerSimulator. The measurement distribution is checked against the classical
list of primes: PASS if the sampled outcomes are overwhelmingly concentrated
on the classically-computed prime set.
"""

import math
from itertools import product

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator


def classical_primes(n_max: int):
    """Trial-division primality test, computed here (no lookup)."""
    primes = []
    for n in range(2, n_max):
        is_p = True
        for d in range(2, int(math.isqrt(n)) + 1):
            if n % d == 0:
                is_p = False
                break
        if is_p:
            primes.append(n)
    return primes


def build_oracle(qc: QuantumCircuit, qubits, marked_values, n_bits):
    """Phase-flip each basis state in marked_values (list of ints < 2**n_bits)."""
    for value in marked_values:
        bits = format(value, f"0{n_bits}b")  # MSB..LSB string
        # X on qubits whose target bit is 0, so the all-ones pattern lines up
        # with `value`. qubits[0] is the least-significant qubit.
        flip_qubits = []
        for i, q in enumerate(qubits):
            bit = bits[n_bits - 1 - i]
            if bit == "0":
                flip_qubits.append(q)
        for q in flip_qubits:
            qc.x(q)
        if n_bits == 1:
            qc.z(qubits[0])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for q in flip_qubits:
            qc.x(q)


def build_diffuser(qc: QuantumCircuit, qubits, n_bits):
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


def run_grover(marked_values, n_bits, shots=4096):
    qr = QuantumRegister(n_bits, "q")
    cr = ClassicalRegister(n_bits, "c")
    qc = QuantumCircuit(qr, cr)

    for q in qr:
        qc.h(q)

    n_total = 2 ** n_bits
    m = len(marked_values)
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_total / m)))

    for _ in range(iterations):
        build_oracle(qc, list(qr), marked_values, n_bits)
        build_diffuser(qc, list(qr), n_bits)

    qc.measure(qr, cr)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_bits = 4
    n_max = 2 ** n_bits  # domain 0..15

    primes = classical_primes(n_max)
    print(f"Classical primes in [0, {n_max - 1}]: {primes}")

    counts, iterations = run_grover(primes, n_bits)
    print(f"Grover iterations used: {iterations}")

    shots_total = sum(counts.values())
    # Qiskit bitstrings are MSB..LSB matching qubit order q[n-1]..q[0].
    hit_shots = 0
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        if value in primes:
            hit_shots += c

    hit_fraction = hit_shots / shots_total
    print(f"Fraction of shots landing on a prime value: {hit_fraction:.4f}")

    # With the correct number of Grover iterations for M=6, N=16, the
    # theoretical success probability (sin^2((2k+1)*theta), theta =
    # asin(sqrt(M/N))) peaks at ~0.83 for the nearest-integer iteration
    # count, since M/N does not land on a value giving near-certainty.
    # Demand a threshold comfortably below that peak but far above the
    # uniform-random baseline of M/N = 0.375.
    threshold = 0.75
    passed = hit_fraction >= threshold

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
