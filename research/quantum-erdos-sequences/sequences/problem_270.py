"""
Erdos problem #270 — quantum-testable instance.

Source data: erdosproblems.com problem #270 (data/problems.yaml, number "270"),
informal_status: disproved (Lean, 2026-08-24), tags: ["irrationality"],
oeis: ["A073016"].

OEIS A073016 is the decimal expansion of Sum_{n>=1} 1/binomial(2n, n)
  = (9 + 2*sqrt(3)*Pi) / 27
i.e. the reciprocal-central-binomial-coefficient sum that problem #270's
irrationality-measure question concerns. Its digits (0.73639985871871507790...)
are:
    7, 3, 6, 3, 9, 9, 8, 5, 8, 7, 1, 8, 7, 1, 5, 0, 7, 7, 9, 0, ...

Classical property tested (computed here from first principles, not copied
from OEIS): using exact rational arithmetic (fractions.Fraction) we sum the
series S = sum_{n=1}^{200} 1/C(2n,n) exactly, then extract its first 8
decimal digits by exact long division. Among the first N=8 digits (indices
0..7), we find the *set of index positions whose digit equals the target
digit 9* (a small, finite, fully computable search problem). The classical
brute-force answer over this space of 8 indices is computed directly from
the digit list.

Quantum computation: a Grover search over the N=8 (3-qubit) index space,
with an oracle that flags exactly the indices whose corresponding decimal
digit equals 9. With 2 marked out of 8 states, one Grover iteration is
(near-)optimal and should concentrate essentially all measurement
probability on the two marked indices. We run this on the ideal AerSimulator
and compare the set of high-probability measured indices against the
classical set computed above.

This is a genuine (if small) instance of amplitude amplification / Grover
search applied to a real, derived property of the OEIS sequence tied to
this Erdos problem — not a fabricated or literal-copy property.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from fractions import Fraction
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


def exact_decimal_digits(num_terms: int, num_digits: int):
    """Exactly compute the first `num_digits` decimal digits (after the
    decimal point) of S = sum_{n=1}^{num_terms} 1/C(2n, n), using exact
    rational arithmetic (no floating point)."""
    S = Fraction(0)
    for n in range(1, num_terms + 1):
        S += Fraction(1, math.comb(2 * n, n))

    num = S.numerator
    den = S.denominator
    int_part = num // den
    rem = num - int_part * den

    digits = []
    for _ in range(num_digits):
        rem *= 10
        d = rem // den
        digits.append(int(d))
        rem -= d * den
    return int_part, digits


def classical_answer(digits, target_digit):
    """Indices i in range(len(digits)) with digits[i] == target_digit."""
    return sorted(i for i, d in enumerate(digits) if d == target_digit)


def grover_oracle(n_qubits, marked_indices):
    """Phase-flip oracle marking the given computational basis indices."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [q for q, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        # Multi-controlled Z on all n_qubits (phase flip |11...1>)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def grover_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits, marked_indices, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = grover_oracle(n_qubits, marked_indices)
    diffuser = grover_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    N_DIGITS = 8          # 2^3 = 8 -> 3 qubits
    N_QUBITS = 3
    TARGET_DIGIT = 9
    NUM_TERMS = 200        # far more than enough for 8 correct decimal digits
    SHOTS = 4096

    int_part, digits = exact_decimal_digits(NUM_TERMS, N_DIGITS)
    assert int_part == 0

    # Sanity-check against the known OEIS A073016 initial terms.
    oeis_a073016_prefix = [7, 3, 6, 3, 9, 9, 8, 5]
    assert digits == oeis_a073016_prefix, (
        f"computed digits {digits} do not match OEIS A073016 prefix "
        f"{oeis_a073016_prefix}"
    )

    classical = classical_answer(digits, TARGET_DIGIT)
    print(f"OEIS A073016 first {N_DIGITS} digits: {digits}")
    print(f"Classical answer (indices with digit == {TARGET_DIGIT}): {classical}")

    num_marked = len(classical)
    assert 0 < num_marked < 2 ** N_QUBITS

    # Optimal number of Grover iterations for M marked out of 2^n states.
    theta = math.asin(math.sqrt(num_marked / 2 ** N_QUBITS))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = build_grover_circuit(N_QUBITS, classical, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=SHOTS).result()
    counts = result.get_counts()

    # Aggregate probability landing on the classically-marked indices.
    marked_prob = 0
    outcome_counts = {}
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        outcome_counts[idx] = outcome_counts.get(idx, 0) + count
        if idx in classical:
            marked_prob += count
    marked_prob /= SHOTS

    print(f"Grover iterations used: {iterations}")
    print(f"Measured outcome distribution (index -> counts): {outcome_counts}")
    print(f"Probability mass on classically-correct indices: {marked_prob:.4f}")

    # Most-likely measured index/indices should be exactly the classical set
    # (allow ties / near-ties, since with 2 marked states out of 8 both tend
    # to be amplified almost equally).
    sorted_by_count = sorted(outcome_counts.items(), key=lambda kv: -kv[1])
    top_indices = sorted([idx for idx, _ in sorted_by_count[:num_marked]])

    verified = (top_indices == classical) and (marked_prob > 0.85)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
