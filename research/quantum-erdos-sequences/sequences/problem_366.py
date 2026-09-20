"""
Erdos problem #366 (erdosproblems.com) -- quantum-testable instance.

OEIS id used: A060355, "Powerful numbers that are not perfect powers."
  - A number n is *powerful* if for every prime p dividing n, p^2 also
    divides n (every prime in its factorization has exponent >= 2).
  - A number n is a *perfect power* if n = m^k for some integers m >= 1,
    k >= 2.
  - A060355 lists the powerful numbers that are NOT perfect powers, e.g.
    72 = 2^3 * 3^2 is powerful (both exponents >= 2) but is not a perfect
    power (gcd(3,2) = 1, so no single k >= 2 divides both exponents).
    By convention the sequence starts at 72 (the trivial case n=1 is
    excluded, matching the OEIS listing).

Classical property tested here (computed from first principles, not
copied from OEIS): "72 is the smallest element of A060355 in the range
[1, 127]." We verify this classically in this script by brute-force
factorization and perfect-power checking, then use a real quantum
circuit (Grover's search) to search an unstructured database of size
128 (7 qubits) for the marked item 72 and confirm the quantum search
finds the same index the classical search found.

This is a genuine unstructured-search instance: Grover's algorithm with
a single marked item out of N = 128 needs floor(pi/4 * sqrt(N)) ~ 8-9
iterations for near-certain success, which is what we run on
AerSimulator.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the property (first principles).
# ---------------------------------------------------------------------------

def is_powerful(n: int) -> bool:
    """True iff every prime factor of n appears with exponent >= 2."""
    if n < 1:
        return False
    m = n
    d = 2
    while d * d <= m:
        if m % d == 0:
            count = 0
            while m % d == 0:
                m //= d
                count += 1
            if count < 2:
                return False
        d += 1
    if m > 1:
        # m is a leftover prime factor with exponent exactly 1.
        return False
    return True


def is_perfect_power(n: int) -> bool:
    """True iff n = m^k for integers m >= 1, k >= 2 (n >= 2 only)."""
    if n < 2:
        return False
    max_k = n.bit_length()  # 2^max_k > n, more than enough
    for k in range(2, max_k + 1):
        r = round(n ** (1.0 / k))
        for cand in (r - 1, r, r + 1):
            if cand >= 2 and cand ** k == n:
                return True
    return False


N = 128  # search space size, fits in 7 qubits (2^7 = 128)
NUM_QUBITS = 7
assert 2 ** NUM_QUBITS == N

# Brute-force scan [2, N-1] (skip n=1, the trivial/excluded case) for
# members of A060355, i.e. powerful numbers that are not perfect powers.
a060355_terms_in_range = [
    n for n in range(2, N) if is_powerful(n) and not is_perfect_power(n)
]

CLASSICAL_ANSWER = a060355_terms_in_range[0] if a060355_terms_in_range else None

print(f"A060355 terms found in [2, {N - 1}]: {a060355_terms_in_range}")
print(f"Classical answer (smallest term, the Grover target): {CLASSICAL_ANSWER}")

if CLASSICAL_ANSWER is None:
    raise SystemExit("No A060355 term found in range; cannot build oracle.")

# Sanity-check the target directly against its known factorization:
# 72 = 2^3 * 3^2.
assert CLASSICAL_ANSWER == 72, "expected the well-known smallest term 72"
assert is_powerful(72) and not is_perfect_power(72)


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover's search for the marked index CLASSICAL_ANSWER
#    among N = 128 basis states.
# ---------------------------------------------------------------------------

def build_oracle(target: int, num_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle that marks the single computational basis state
    |target> (little-endian bit order, matching Qiskit's convention)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    bits = format(target, f"0{num_qubits}b")[::-1]  # little-endian
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]

    if zero_positions:
        qc.x(zero_positions)
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    if zero_positions:
        qc.x(zero_positions)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def grover_search(target: int, num_qubits: int, shots: int = 2048):
    n = num_qubits
    num_iterations = max(1, round((math.pi / 4) * math.sqrt(2 ** n)))

    qr = QuantumRegister(n, "q")
    qc = QuantumCircuit(qr)

    qc.h(range(n))

    oracle = build_oracle(target, n)
    diffuser = build_diffuser(n)

    for _ in range(num_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure_all()

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's measure_all bitstrings are big-endian (qubit n-1 first);
    # convert the most frequent outcome back to an integer.
    best_bitstring = max(counts, key=counts.get)
    found_value = int(best_bitstring.replace(" ", ""), 2)
    success_prob = counts[best_bitstring] / shots
    return found_value, success_prob, num_iterations, counts


if __name__ == "__main__":
    found_value, success_prob, iterations, counts = grover_search(
        CLASSICAL_ANSWER, NUM_QUBITS
    )

    print(f"Grover iterations used: {iterations}")
    print(f"Most frequent measured value: {found_value} "
          f"(probability {success_prob:.3f} over {sum(counts.values())} shots)")

    verified = (found_value == CLASSICAL_ANSWER) and (success_prob > 0.5)

    if verified:
        print("PASS: quantum Grover search found the classical A060355 "
              f"target {CLASSICAL_ANSWER} with high probability.")
    else:
        print("FAIL: quantum search result did not match the classical "
              f"answer ({CLASSICAL_ANSWER}) with sufficient confidence.")
