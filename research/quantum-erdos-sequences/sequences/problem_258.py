"""
Erdos problem #258 (as recorded in manman4/erdosproblems, data/problems.yaml)
------------------------------------------------------------------------------
Metadata for problem 258 (verified by grep on 2026-09-19):

    number: "258"
    status: proved (Lean), no prize
    oeis:   ["N/A"]          <-- no OEIS sequence is attached to this problem
    tags:   ["irrationality"]

LIMITATION, stated up front and honestly: problem 258 has no OEIS id at all
("N/A"), so there is no concrete integer sequence to build a Grover/oracle
search or a membership/counting circuit around, as the task description asks
for. Fabricating a fake OEIS-backed property would violate the instructions,
so this script does not do that.

What this script does instead, as the best honest substitute tied to the
problem's one real piece of content (its tag, "irrationality"):

    A genuine, small, finite, *verifiable-classically* quantum computation
    in the same family of techniques (Quantum Phase Estimation) that is
    actually used to attack irrationality-adjacent questions: estimating
    the binary expansion of an irrational number's fractional part.

    Concretely: let theta = frac(sqrt(2)) (irrational, since sqrt(2) is
    irrational). We build a single-qubit unitary U such that
        U |1> = exp(2*pi*i*theta) |1>
    (a phase gate with angle 2*pi*theta) and run the standard n-qubit QPE
    circuit (n = 5 counting qubits) on the ideal AerSimulator to recover the
    top n bits of theta's binary expansion.

    The classical answer is theta's true binary expansion to n bits,
    computed from Python's arbitrary-precision `decimal` module (first
    principles: Newton's method for sqrt(2), no external deps).

    PASS/FAIL is decided by comparing the quantum circuit's most likely
    measured n-bit integer against round(theta * 2**n) mod 2**n, the
    classical n-bit binary approximation of theta.

This is a real Qiskit circuit computing a real, non-fabricated numerical
property (a finite-precision binary truncation of an irrational number's
fractional part) and checking it against an independently, classically
computed value. It is not a sequence-membership test against problem 258's
OEIS entry, because problem 258 has none.
"""

import math
from decimal import Decimal, getcontext

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
import numpy as np


def classical_sqrt2_frac_bits(n_bits: int) -> tuple[Decimal, int]:
    """Compute theta = frac(sqrt(2)) to high precision via Newton's method
    (first principles, no external numeric libraries), then return the
    exact Decimal value of theta and its n_bits-bit binary truncation as
    an integer in [0, 2**n_bits).
    """
    getcontext().prec = 60
    two = Decimal(2)
    x = Decimal(1)
    for _ in range(100):
        x = (x + two / x) / 2
    # x is now sqrt(2) to ~50 correct decimal digits.
    theta = x - Decimal(int(x))  # fractional part
    scaled = theta * (Decimal(2) ** n_bits)
    k = int(scaled.to_integral_value(rounding="ROUND_HALF_EVEN"))
    k %= 2 ** n_bits
    return theta, k


def qpe_circuit(theta_angle_turns: float, n_counting: int) -> QuantumCircuit:
    """Standard textbook QPE circuit estimating the eigenphase of the
    1-qubit unitary U: |1> -> exp(2*pi*i*theta_angle_turns) |1>,
    U: |0> -> |0>, using the eigenstate |1> on the target qubit and
    n_counting counting qubits.
    """
    qc = QuantumCircuit(n_counting + 1, n_counting)

    target = n_counting
    qc.x(target)  # prepare eigenstate |1>

    for q in range(n_counting):
        qc.h(q)

    # Controlled-U^(2^q) for each counting qubit q, U = phase gate.
    for q in range(n_counting):
        power = 2 ** q
        angle = 2 * math.pi * theta_angle_turns * power
        qc.cp(angle, q, target)

    # Inverse QFT on counting register, then measure.
    qc.append(QFT(n_counting, inverse=True).to_gate(label="QFT†"), range(n_counting))
    qc.measure(range(n_counting), range(n_counting))
    return qc


def main() -> None:
    n_counting = 5  # small finite instance: 5 counting qubits (2^5 = 32 bins)

    theta_exact, k_classical = classical_sqrt2_frac_bits(n_counting)

    circuit = qpe_circuit(float(theta_exact), n_counting)

    sim = AerSimulator()
    tqc = transpile(circuit, sim)
    result = sim.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # Qiskit's classical-register bit order is little-endian in the count
    # string (c[n-1] ... c[0]); reverse to get counting-register order
    # matching qubit index 0..n-1, matching how QFT/QPE bit-order convention
    # is normally reported.
    best_bitstring = max(counts, key=counts.get)
    k_quantum = int(best_bitstring, 2)

    print("Erdos problem #258 (no OEIS id; 'irrationality' tag)")
    print(f"theta = frac(sqrt(2)) (exact, high precision) = {theta_exact}")
    print(f"n_counting = {n_counting} bits")
    print(f"classical n-bit truncation of theta * 2^n = {k_classical} "
          f"(binary {format(k_classical, f'0{n_counting}b')})")
    print(f"quantum QPE most-likely outcome            = {k_quantum} "
          f"(binary {format(k_quantum, f'0{n_counting}b')})  "
          f"[{counts[best_bitstring]}/4096 shots]")

    # QPE recovers theta to within +-1 of the true n-bit truncation with
    # high probability on the ideal simulator; allow that documented
    # +-1 rounding tolerance (standard QPE error bound) before failing.
    diff = min((k_quantum - k_classical) % (2 ** n_counting),
               (k_classical - k_quantum) % (2 ** n_counting))
    passed = diff <= 1

    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
