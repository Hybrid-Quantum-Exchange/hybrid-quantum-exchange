"""
Erdos problem #243 (erdosproblems.com/243) — quantum-testable instance.

OEIS: A000058 — Sylvester's sequence, a(0)=2, a(n+1) = a(n)^2 - a(n) + 1
(equivalently a(n+1) = a(0)*a(1)*...*a(n) + 1). Problem #243 concerns the
irrationality/behaviour of the sum of reciprocals of this sequence; tags:
["number theory", "irrationality"].

Classical property tested (finite, computable):
    Sylvester's sequence has the well-known "greedy Egyptian fraction"
    property that a(n) - 1 = a(0)*a(1)*...*a(n-1) (the product of all prior
    terms), i.e. each term is highly non-random modulo small numbers. Here
    we test a small, genuinely searchable instance of that structure:

    Search space: n in {0, 1, ..., 7} (3 qubits).
    Target predicate: a(n) mod 3 == 0.

    a(0)=2, a(1)=3, a(2)=7, a(3)=43, a(4)=1807, a(5)=3263443,
    a(6)=10650056950807, a(7)=113423713055421844361000443.

    a(n) mod 3 for n=0..7: [2, 0, 1, 1, 1, 1, 1, 1]
    (all computed classically in this script from the recurrence, using
    Python's arbitrary-precision integers — no OEIS values are copied
    verbatim; they are re-derived here.)

    So the unique index n in {0,...,7} with a(n) mod 3 == 0 is n = 1
    (since a(1) = 3, and for n >= 1 every later term is 1 mod 3 because the
    recurrence a(n+1) = a(n)^2 - a(n) + 1 preserves residue 1 mod 3 once a
    term is 1 mod 3).

Quantum circuit: a genuine Grover search over the 3-qubit index register
{0,...,7}, whose oracle marks exactly the classically-precomputed unique
solution {1} (binary 001, little-endian). Grover's algorithm is run for the
optimal number of iterations for |solutions|=1 out of N=8, and the
measurement distribution is checked against the classical solution.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, XGate
from qiskit import transpile
from qiskit_aer import AerSimulator


def sylvester_terms(k):
    """Return a(0..k-1) of OEIS A000058 via a(n+1) = a(n)^2 - a(n) + 1."""
    terms = [2]
    for _ in range(k - 1):
        prev = terms[-1]
        terms.append(prev * prev - prev + 1)
    return terms


def classical_solution_set(n_index_bits, modulus, residue):
    """Indices n in [0, 2**n_index_bits) with a(n) mod modulus == residue."""
    n_vals = 2 ** n_index_bits
    terms = sylvester_terms(n_vals)
    solutions = [n for n, val in enumerate(terms) if val % modulus == residue]
    return solutions, terms


def build_oracle(n_qubits, solutions):
    """Phase-flip oracle marking each solution index (little-endian bits)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for sol in solutions:
        bits = format(sol, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        # Multi-controlled Z on all n_qubits (phase flip when all qubits |1>)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCMTGate(XGate(), n_qubits - 1, 1), list(range(n_qubits)))
            qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCMTGate(XGate(), n_qubits - 1, 1), list(range(n_qubits)))
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits, solutions, shots=4096):
    n_vals = 2 ** n_qubits
    m = len(solutions)
    # Optimal number of Grover iterations for m solutions out of n_vals.
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_vals / m) - 0.5))

    oracle = build_oracle(n_qubits, solutions)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    transpiled = transpile(qc, sim)
    result = sim.run(transpiled, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_index_bits = 3  # search space size N = 8
    modulus = 3
    residue = 0

    solutions, terms = classical_solution_set(n_index_bits, modulus, residue)
    print(f"Sylvester's sequence (OEIS A000058), first {2**n_index_bits} terms:")
    print(terms)
    print(f"a(n) mod {modulus} for n=0..{2**n_index_bits - 1}: "
          f"{[t % modulus for t in terms]}")
    print(f"Classical solution set (a(n) mod {modulus} == {residue}): {solutions}")

    counts, iterations = run_grover(n_index_bits, solutions, shots=4096)
    print(f"\nGrover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    # Little-endian bitstrings from Qiskit -> integer index.
    shots_total = sum(counts.values())
    measured_solution_shots = 0
    top_indices = []
    for bitstring, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        # Qiskit's classical bitstring lists c_{n-1}...c_0 (MSB first), with
        # c_i the measurement of qubit i, so int(bitstring, 2) already
        # reconstructs index = sum_i q_i * 2**i.
        index = int(bitstring, 2)
        top_indices.append((index, count))
        if index in solutions:
            measured_solution_shots += count

    success_fraction = measured_solution_shots / shots_total
    print(f"Top measured indices (index, count): {top_indices[:5]}")
    print(f"Fraction of shots landing on a classical solution index: "
          f"{success_fraction:.4f}")

    # Verification: amplified solution states should dominate measurements.
    verified = success_fraction > 0.90

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
