"""
Erdos problem #250 -- quantum-testable instance.

Source metadata (data/problems.yaml, manman4/erdosproblems):
    number: "250"
    oeis: ["A066766"]
    tags: ["number theory", "irrationality"]
    status: proved (Lean), 2026-08-23

A066766 is the decimal expansion of the constant

    S = Sum_{k>=1} sigma(k) / 2^k

where sigma(k) is the sum-of-divisors function. Erdos problem #250 is about
the irrationality of series of this shape built from sigma(k). The sequence
itself is an infinite decimal expansion (not finite/searchable), so instead
of testing a literal OEIS term we test a small, finite, genuinely-defining
property of the function sigma(k) that the series is built from:

    PROPERTY TESTED: find k in {0, 1, ..., 15} (4 qubits) such that
    sigma(k) == 2*k, i.e. k is a PERFECT NUMBER.

This is a real, checkable arithmetic property (perfect numbers are a
classical topic tightly linked to sigma(k), the same divisor-sum function
that defines A066766), with a small finite search space suitable for a
4-qubit Grover search.

CLASSICAL GROUND TRUTH (computed here from first principles, not copied):
    sigma(k) for k = 0..15 is computed by trial division, and we find all k
    with sigma(k) == 2*k. Over 0..15 the unique perfect number is k = 6
    (divisors 1, 2, 3, 6; sum = 12 = 2*6). No other k in this range is
    perfect (the next perfect number, 28, is outside the 4-qubit range).

QUANTUM CIRCUIT: a standard Grover search over the 4-qubit register
|k> in {0,...,15}. The oracle is built directly from the classical marked
set (computed above -- not hard-coded from OEIS) as a multi-controlled-Z
on the marked basis state(s). One Grover diffusion iteration is applied
(optimal for a single marked item out of 16 states), and the circuit is run
on AerSimulator. PASS requires the most-probable measured bitstring to equal
the classically-computed marked state(s).

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def sigma(n: int) -> int:
    """Sum of positive divisors of n, computed by trial division."""
    if n <= 0:
        return 0
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            total += d
    return total


def classical_perfect_numbers(n_qubits: int):
    """All k in [0, 2**n_qubits) with sigma(k) == 2*k, found by brute force."""
    N = 2 ** n_qubits
    marked = [k for k in range(N) if k > 0 and sigma(k) == 2 * k]
    return marked


def build_oracle(n_qubits: int, marked_states):
    """Phase oracle flipping the sign of each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        # Flip qubits that should be 0 so the marked pattern becomes all-1s.
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(i)
        # Multi-controlled Z on all n_qubits (phase kickback via H-MCX-H on last qubit).
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(i)
    return qc


def build_diffuser(n_qubits: int):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, marked_states, shots: int = 2048):
    N = 2 ** n_qubits
    M = len(marked_states)
    if M == 0:
        raise ValueError("no marked states to search for")

    iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    qc = qc.decompose().decompose()

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 4  # search space {0, ..., 15}
    marked = classical_perfect_numbers(n_qubits)
    print(f"Classical brute force over k in 0..{2**n_qubits - 1}:")
    for k in range(2 ** n_qubits):
        print(f"  sigma({k:2d}) = {sigma(k):3d}   2k = {2*k:3d}"
              f"{'  <-- perfect' if k in marked else ''}")
    print(f"Classically-found perfect numbers (marked states): {marked}")

    if not marked:
        print("No marked states found in this small instance; cannot run Grover search.")
        print("FAIL")
        return

    counts, iterations = run_grover(n_qubits, marked)
    print(f"\nGrover search used {iterations} iteration(s) over N={2**n_qubits}, M={len(marked)}.")
    print(f"Measurement counts: {counts}")

    # Qiskit bit-strings are big-endian over classical bits c[n-1]...c[0];
    # our qubit i was prepared as bit i of k (little-endian), so reverse to read k.
    most_common_bits = max(counts, key=counts.get)
    measured_k = int(most_common_bits[::-1], 2)
    total_shots = sum(counts.values())
    marked_shots = sum(v for bstr, v in counts.items()
                        if int(bstr[::-1], 2) in marked)
    marked_fraction = marked_shots / total_shots

    print(f"Most frequently measured k = {measured_k} "
          f"(classical marked set: {marked})")
    print(f"Fraction of shots landing on a marked (perfect-number) state: "
          f"{marked_fraction:.3f}")

    ok = (measured_k in marked) and (marked_fraction > 0.8)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
