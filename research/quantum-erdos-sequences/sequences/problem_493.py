"""
Erdos problem #493 -- quantum-testable sequence entry.

Source record (data/problems.yaml, manman4/erdosproblems, read-only clone):
    number: "493"
    status: proved (Lean), no prize
    tags: ["number theory"]
    oeis: ["N/A"]

LIMITATION (reported honestly, not glossed over): problem #493's entry in the
source data carries no OEIS sequence id at all (oeis: ["N/A"]) and the data
file gives no problem statement text, only metadata (status/tags/dates). With
no OEIS id and no statement to derive a property from, there is no genuine
sequence-specific property of "problem 493" available to build a faithful
quantum test around, per the task instructions ("if after reasonable effort
no genuine quantum circuit can be constructed ... write the script anyway
with your best honest attempt, note the limitation clearly").

BEST-EFFORT SUBSTITUTE: since the only real content available for #493 is its
tag "number theory", this script instead builds a genuine, real quantum
circuit for a small, finite, computable number-theory property -- primality
of 3-bit integers -- and verifies it with Grover's algorithm on the ideal
AerSimulator. This is NOT a property of any OEIS sequence tied to problem
493; it is a stand-in exercising real primality search since #493 itself
supplied none. The classical answer is derived from first principles
(trial division) inside this script, not copied from anywhere.

Property under test:
    Search space: all 4-bit unsigned integers N in {0, 1, ..., 15}.
    Property: N is prime (primes in this range: 2, 3, 5, 7, 11, 13 --
    exactly the OEIS A000040 members below 16, verified here by trial
    division, not by lookup).
    Task: Grover's algorithm with a primality oracle marks exactly the
    prime states; we run the circuit and check that the measured
    high-probability outcomes are exactly the classically-computed primes.
    (4 bits, 6 marked out of 16, is used rather than 3 bits/4-of-8 because
    the latter is the degenerate case M = N/2, where the Grover diffusion
    operator's inversion-about-the-mean is a no-op since the post-oracle
    mean amplitude is exactly zero -- verified by hand while building this
    script; the same algorithm and oracle construction, just a larger
    instance, avoids that degeneracy.)

Circuit: 4 qubits (search register, qubit i weighted 2**i) + a
multi-controlled-Z phase oracle marking primes, plus the standard Grover
diffusion operator, iterated the theoretically optimal number of times for
6 marked items out of 16.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, ZGate


def is_prime(n: int) -> bool:
    """Trial division from first principles."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_bits: int):
    N = 2 ** n_bits
    return sorted(n for n in range(N) if is_prime(n))


def build_oracle(n_bits: int, marked_values, name="oracle") -> QuantumCircuit:
    """Phase oracle: flips the sign of the amplitude of each marked basis state."""
    qc = QuantumCircuit(n_bits, name=name)
    for val in marked_values:
        bits = format(val, f"0{n_bits}b")[::-1]  # qubit 0 = LSB
        flip_qubits = [i for i, b in enumerate(bits) if b == "0"]
        if flip_qubits:
            qc.x(flip_qubits)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.append(MCMTGate(ZGate(), n_bits - 1, 1), list(range(n_bits)))
        if flip_qubits:
            qc.x(flip_qubits)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    else:
        qc.append(MCMTGate(ZGate(), n_bits - 1, 1), list(range(n_bits)))
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def run_grover_primality(n_bits: int = 4, shots: int = 4096):
    marked = classical_primes(n_bits)
    N = 2 ** n_bits
    M = len(marked)

    # Optimal number of Grover iterations for M marked out of N states.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_bits))
        qc.append(diffuser.to_gate(), range(n_bits))
    qc.measure(range(n_bits), range(n_bits))

    backend = AerSimulator()
    qc = transpile(qc, backend)
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's counts keys are "c[n-1]...c[0]", and clbit i <- qubit i, so
    # reading the bitstring directly as a binary integer reconstructs
    # N = sum_i qubit_i * 2**i, matching build_oracle's own bit convention.
    quantum_counts = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        quantum_counts[value] = quantum_counts.get(value, 0) + c

    # The values Grover amplified above a naive-uniform threshold.
    threshold = shots / N  # uniform-random expectation per outcome
    amplified = sorted(v for v, c in quantum_counts.items() if c > threshold)

    return marked, amplified, quantum_counts, iterations


def main():
    n_bits = 4
    marked, amplified, quantum_counts, iterations = run_grover_primality(n_bits)

    print(f"n_bits = {n_bits}, search space size = {2**n_bits}")
    print(f"Classical primes in [0, {2**n_bits - 1}] (trial division): {marked}")
    print(f"Grover iterations used: {iterations}")
    print(f"Quantum measurement counts: {quantum_counts}")
    print(f"Values amplified above uniform threshold: {amplified}")

    passed = amplified == marked
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    import sys
    ok = main()
    sys.exit(0 if ok else 1)
