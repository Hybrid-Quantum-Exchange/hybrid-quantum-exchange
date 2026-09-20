"""
Erdos problem #404 (erdosproblems.com), quantum-testable lane.

Source metadata (from manman4/erdosproblems data/problems.yaml, entry
"number: \"404\""): prize="no", status="open", oeis=["N/A"],
tags=["number theory", "factorials"].

LIMITATION, stated honestly: problem #404 carries no OEIS sequence id
("N/A"). There is therefore no specific OEIS sequence to build a quantum
membership/search test around. To still produce a genuine, non-fabricated
quantum circuit that is thematically tied to the problem's own tags
("number theory", "factorials"), this script tests a small, fully
computable factorial-number-theory property and does NOT claim it is
"the" sequence behind problem #404 (there isn't one, per the source data).

Classical property under test
------------------------------
For n in the finite range 1..15, let tz(n) = the number of trailing zero
digits of n! in base 10 (equivalently, the exponent of 5 in the prime
factorization of n!, by Legendre's formula, since factors of 2 are never
the bottleneck for n! in this range).

    tz(n) = sum_{k>=1} floor(n / 5^k)

Property tested: "n is the unique integer in 1..15 with tz(n) = 3."

This is computed from first principles in `trailing_zeros_of_factorial`
below (no OEIS lookup, no hard-coded answer) over the full range 1..15,
which gives:

    tz(1..4)   = 0
    tz(5..9)   = 1
    tz(10..14) = 2
    tz(15)     = 3   <-- unique solution

So the classical answer is n = 15, and it is the *only* marked item in
the search space {1, ..., 15} (represented as 4-qubit basis states
0000..1111, with state |1111> = 15 the unique target -- 1..14 are
represented directly as their 4-bit binary value, and 0 is present in the
space but is never a solution since tz(0) = 0 != 3).

Quantum circuit
----------------
A standard single-solution Grover search over 4 qubits (16-item search
space): a phase oracle that flags exactly |1111> (an MCZ gate, since the
unique classical solution 15 = 1111 in binary), a diffusion operator, and
floor(pi/4 * sqrt(16/1)) = 3 Grover iterations (optimal for 1 marked item
out of 16). We simulate the ideal circuit on AerSimulator, measure, and
check that the most frequent outcome equals the classically-derived
answer 15.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Classical part: compute tz(n) for n in 1..15 from first principles, and
# find the unique n with tz(n) == 3. No OEIS values are hard-coded.
# ---------------------------------------------------------------------------

def factorial(n: int) -> int:
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def trailing_zeros_of_factorial(n: int) -> int:
    """Number of trailing base-10 zeros of n!, via Legendre's formula
    applied to the prime 5 (the bottleneck prime for n! trailing zeros)."""
    count = 0
    power = 5
    while power <= n:
        count += n // power
        power *= 5
    return count


def trailing_zeros_by_brute_force(n: int) -> int:
    """Cross-check: compute n! directly and count trailing zeros in its
    base-10 representation, independent of the Legendre-formula method."""
    val = factorial(n)
    if val == 0:
        return 0
    count = 0
    while val % 10 == 0:
        count += 1
        val //= 10
    return count


N_QUBITS = 4
SEARCH_SPACE = list(range(0, 2 ** N_QUBITS))  # 0..15

tz_values = {}
for n in SEARCH_SPACE:
    if n == 0:
        tz_values[n] = 0  # 0! = 1, zero trailing zeros; not part of 1..15 range of interest
        continue
    formula = trailing_zeros_of_factorial(n)
    brute = trailing_zeros_by_brute_force(n)
    assert formula == brute, f"cross-check failed for n={n}: {formula} vs {brute}"
    tz_values[n] = formula

TARGET_TZ = 3
solutions = [n for n in range(1, 16) if tz_values[n] == TARGET_TZ]
assert len(solutions) == 1, f"expected a unique solution, got {solutions}"
CLASSICAL_ANSWER = solutions[0]
print(f"Classical property: unique n in 1..15 with tz(n!) == {TARGET_TZ}")
print(f"tz(n) table (n=1..15): {[tz_values[n] for n in range(1, 16)]}")
print(f"Classical answer: n = {CLASSICAL_ANSWER} (binary {CLASSICAL_ANSWER:04b})")


# ---------------------------------------------------------------------------
# Quantum part: Grover search for the unique marked basis state.
# ---------------------------------------------------------------------------

def build_oracle(marked: int, n_qubits: int) -> QuantumCircuit:
    """Phase oracle flipping the sign of |marked> only, via an
    X-sandwiched multi-controlled Z (MCZ) gate."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(marked, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_search(marked: int, n_qubits: int, shots: int = 2048):
    n_items = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_items / 1)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def most_frequent_outcome(counts: dict) -> int:
    best_bitstring = max(counts, key=counts.get)
    # Qiskit bit order: rightmost char is qubit 0.
    return int(best_bitstring[::-1], 2)


def main():
    counts, iterations = grover_search(CLASSICAL_ANSWER, N_QUBITS)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    quantum_answer = most_frequent_outcome(counts)
    total_shots = sum(counts.values())
    marked_bitstring = format(CLASSICAL_ANSWER, f"0{N_QUBITS}b")[::-1]
    marked_probability = counts.get(marked_bitstring, 0) / total_shots

    print(f"Most frequent measured value: {quantum_answer}")
    print(f"Probability mass on classical answer ({CLASSICAL_ANSWER}): {marked_probability:.3f}")

    ok = (quantum_answer == CLASSICAL_ANSWER) and (marked_probability > 0.5)
    if ok:
        print("PASS")
    else:
        print("FAIL")
    return ok


if __name__ == "__main__":
    main()
