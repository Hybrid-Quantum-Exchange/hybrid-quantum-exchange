"""
Erdos problem #649 -- quantum-testable sequence entry (best-effort fallback).

Source metadata (erdosproblems/data/problems.yaml, entry "number: \"649\""):
    prize: no
    informal_status: disproved (Lean, 2026-02-07)
    oeis: ["possible"]
    tags: ["number theory"]

LIMITATION (read before trusting the PASS below): the yaml's `oeis` field for
problem 649 is the literal string "possible" -- not an OEIS A-number. There is
no real OEIS sequence id attached to this problem in the source data, so there
is no genuine "sequence" to derive a finite computable property from, and the
requested "OEIS sequence property" quantum circuit cannot honestly be built
for problem #649 as asked. Rather than fabricate a fake OEIS id or pretend a
literal value was "derived" from a sequence that isn't there, this script
falls back to the one concrete, checkable classical claim available: problem
649's tag is "number theory" and its status is "disproved". As a small,
honest, finite computable number-theory property in that spirit, this script
uses Grover search to find the primes in {0, ..., 15} (4 qubits), which is
verified against a first-principles classical trial-division primality test
computed in this script. This is NOT a property of an OEIS sequence tied to
problem 649 specifically -- it is a best-honest-attempt placeholder given the
missing OEIS id, and is reported as such (no real OEIS id used).

Classical computation (trial division, computed here, not copied):
    Primes in [0, 15]: 2, 3, 5, 7, 11, 13  (6 primes out of 16 integers)

Quantum computation:
    A 4-qubit Grover search with an oracle marking exactly the primes in
    [0, 15], run on the ideal AerSimulator, iterated the optimal number of
    times for 6 marked states out of 16. The script checks that the most
    frequently measured basis states after Grover iterations are all primes
    (i.e. Grover search recovers members of the classically-computed set),
    which is the finite computable property actually being tested here.

PASS/FAIL is determined by comparing the set of quantum-favored outcomes to
the classical prime set -- not by copying a literal value from OEIS.
"""

import sys
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


def classical_primes(n_max: int):
    """Trial-division primality test computed from first principles."""
    primes = []
    for k in range(n_max + 1):
        if k < 2:
            continue
        is_prime = True
        d = 2
        while d * d <= k:
            if k % d == 0:
                is_prime = False
                break
            d += 1
        if is_prime:
            primes.append(k)
    return primes


def build_oracle(n_qubits: int, marked_values: list) -> QuantumCircuit:
    """Phase oracle flipping the sign of each marked basis state |v>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover_prime_search(n_max: int, shots: int = 4096):
    n_qubits = int(np.ceil(np.log2(n_max + 1)))
    N = 2 ** n_qubits

    primes = classical_primes(n_max)
    marked = [p for p in primes if p < N]
    M = len(marked)

    # Optimal number of Grover iterations for M marked out of N states.
    theta = np.arcsin(np.sqrt(M / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qreg = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(qreg)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure_all()

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

    return primes, marked, counts, iterations


def main():
    n_max = 15  # 4 qubits: search space {0, ..., 15}
    primes, marked, counts, iterations = run_grover_prime_search(n_max)

    print(f"Classical primes in [0, {n_max}]: {primes}")
    print(f"Marked (in-range) values used as oracle targets: {marked}")
    print(f"Grover iterations used: {iterations}")

    # Sort measured outcomes by frequency, take the top-M (M = number marked).
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    total_shots = sum(counts.values())
    top_m = sorted_counts[: len(marked)]

    print("Top measured outcomes (bitstring: count):")
    for bitstring, count in top_m:
        value = int(bitstring, 2)
        print(f"  {bitstring} -> {value}  (count {count}/{total_shots})")

    top_values = {int(bitstring, 2) for bitstring, _ in top_m}
    classical_set = set(marked)

    # Require that the quantum-favored outcomes are exactly the classical
    # prime set, and that together they carry a clear majority of the shots
    # (well above uniform-random baseline of M/N).
    shots_on_top = sum(count for _, count in top_m)
    total = sum(counts.values())
    fraction = shots_on_top / total
    baseline = len(marked) / (2 ** int(np.ceil(np.log2(n_max + 1))))

    correct_set = top_values == classical_set
    amplified = fraction > baseline * 1.5

    verified = correct_set and amplified

    print(f"Quantum top-{len(marked)} set matches classical prime set: {correct_set}")
    print(f"Fraction of shots landing on classical primes: {fraction:.3f} "
          f"(uniform baseline {baseline:.3f})")

    if verified:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
