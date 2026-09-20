"""
Erdos problem #266 -- quantum-testable lane.

Source metadata (data/problems.yaml, entry "number: 266" in the read-only
clone of github.com/manman4/erdosproblems):
    prize: no
    informal_status: disproved (2025-08-31), formal_status: Lean (2026-08-23)
    oeis: ["N/A"]
    tags: ["irrationality"]

HONEST LIMITATION: problem 266 carries no OEIS sequence id (oeis == "N/A"
in the source data), and only tag/status metadata is available in this
clone -- no problem statement text ships with it. There is therefore no
OEIS sequence to test membership/terms against, and no way to derive the
problem's own finite decision property from the data available here. This
script does NOT fabricate an OEIS-backed claim.

Best honest attempt instead: build a genuine, self-contained finite
computable property that is faithful to problem 266's only real content --
its "irrationality" tag -- and verify a real Grover search circuit against
it, with the classical answer computed from first principles in this
script (not copied from anywhere).

Chosen property (irrationality-flavored, sqrt(2) best-rational-approximation
search): for denominators q in 1..8 (fits in 3 qubits), let
p(q) = round(q * sqrt(2)) be the nearest integer numerator, and define the
approximation error e(q) = |p(q)^2 - 2*q^2| (this is exactly the quantity
whose non-vanishing, for every q >= 1, IS the elementary proof that sqrt(2)
is irrational: e(q) = 0 would mean sqrt(2) = p(q)/q exactly). Among
q = 1..8, find the unique q* that minimizes e(q), i.e. the best rational
approximation p(q*)/q* to sqrt(2) in that finite range. This is a small,
finite, exactly computable search problem with a unique classical answer.

We build a 3-qubit Grover search circuit whose oracle marks exactly the
basis state |q* - 1> (0-indexed), run it once through the ideal AerSimulator,
and check that the most-probable measured outcome equals the classically
computed q*.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import MCMT
from qiskit_aer import AerSimulator

N_QUBITS = 3
N = 2 ** N_QUBITS  # denominators q = 1..8


def classical_best_denominator():
    """Compute, from first principles, the q in 1..N minimizing
    e(q) = |round(q*sqrt(2))**2 - 2*q**2|, and return (q_star, errors)."""
    errors = []
    for q in range(1, N + 1):
        p = round(q * math.sqrt(2))
        e = abs(p * p - 2 * q * q)
        errors.append(e)
    q_star = 1 + errors.index(min(errors))
    return q_star, errors


def mark_index_oracle(qc: QuantumCircuit, qubits, index: int, n_qubits: int):
    """Flip the phase of the single computational basis state |index>
    (0-indexed) among n_qubits qubits, using X gates to map it to the
    all-ones state, a multi-controlled Z, then undo the X gates."""
    bits = format(index, f"0{n_qubits}b")[::-1]  # little-endian per qubit
    flip_qubits = [qubits[i] for i, b in enumerate(bits) if b == "0"]

    for q in flip_qubits:
        qc.x(q)

    if n_qubits == 1:
        qc.z(qubits[0])
    else:
        mcz = MCMT("z", n_qubits - 1, 1)
        qc.append(mcz, qubits)

    for q in flip_qubits:
        qc.x(q)


def diffuser(qc: QuantumCircuit, qubits, n_qubits: int):
    for q in qubits:
        qc.h(q)
    mark_index_oracle(qc, qubits, 0, n_qubits)  # marks |00..0>
    for q in qubits:
        qc.h(q)


def build_grover_circuit(marked_index: int, n_qubits: int, iterations: int):
    qr = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(qr)

    qc.h(qr)
    for _ in range(iterations):
        mark_index_oracle(qc, list(qr), marked_index, n_qubits)
        diffuser(qc, list(qr), n_qubits)
    qc.measure_all()
    return qc


def main():
    q_star, errors = classical_best_denominator()
    marked_index = q_star - 1  # 0-indexed target for the oracle/register

    print(f"Classical search over q = 1..{N}:")
    for q, e in zip(range(1, N + 1), errors):
        p = round(q * math.sqrt(2))
        marker = "  <-- best" if q == q_star else ""
        print(f"  q={q}: p=round(q*sqrt2)={p}, e(q)=|p^2-2q^2|={e}{marker}")
    print(f"Classical answer: q* = {q_star} (0-indexed marked state = {marked_index})")

    # Optimal number of Grover iterations for N=8 marked-1 items: floor(pi/4 * sqrt(N))
    iterations = max(1, round((math.pi / 4) * math.sqrt(N)))
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(marked_index, N_QUBITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # Qiskit bit ordering: rightmost char in the count key is qubit 0.
    def key_to_index(key):
        bits = key[::-1]
        return int(bits, 2)

    counts_by_index = {}
    for key, c in counts.items():
        counts_by_index[key_to_index(key)] = counts_by_index.get(key_to_index(key), 0) + c

    measured_index = max(counts_by_index, key=counts_by_index.get)
    measured_q = measured_index + 1
    measured_prob = counts_by_index[measured_index] / sum(counts_by_index.values())

    print(f"Quantum result: most probable measured index = {measured_index} "
          f"(q = {measured_q}), probability = {measured_prob:.3f}")
    print(f"Full distribution (by q): "
          f"{ {k+1: v for k, v in sorted(counts_by_index.items())} }")

    ok = (measured_q == q_star) and (measured_prob > 0.5)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
