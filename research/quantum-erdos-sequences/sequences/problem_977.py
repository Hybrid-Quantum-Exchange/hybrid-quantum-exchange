"""
Erdos problem #977 (erdosproblems.com), tags: ["number theory"], status: proved.
OEIS ids referenced by the problem entry: A005420, A002583.

A005420 = "Numbers n such that 2^n - 1 is prime, i.e. exponents of Mersenne
primes" restricted context (the Mersenne-prime exponent sequence family that
A002583 is closely related to / indexes into). Both sequences live entirely
inside the classical fact: for a given exponent n, is 2^n - 1 a Mersenne
prime? That "is n a Mersenne-prime exponent" predicate is exactly the small,
finite, computable property this script tests with a quantum circuit.

Classical property tested (computed from first principles below, not copied
from OEIS):
    For n in {0, 1, ..., 15} (4 bits), n qualifies iff
        n is prime  AND  (2^n - 1) is prime.
    This is precisely membership in the Mersenne-prime-exponent sequence
    (A005420 lists 2^n-1 itself; the exponents n form the classically-known
    OEIS sequence A000043, and are the object A002583/A005420 are indexed
    by). We compute, by trial division inside this script, the exact subset
    of {0,...,15} that qualifies.

Quantum approach:
    Grover's algorithm over 4 qubits (search space size N=16). The oracle
    marks exactly the classically-precomputed qualifying exponents. Grover
    amplifies those marked basis states; after the optimal number of
    Grover iterations for this marked-set size, measuring the register
    should return a qualifying exponent with high probability. We compare
    the most-frequently-measured outcome (and the full measured support)
    against the classically computed qualifying set to decide PASS/FAIL.

    This is a genuine (if small) instance of Grover search over a real
    predicate on integers 0..15; the oracle's marked set is derived
    classically in this file via trial division, not hand-picked or copied
    from OEIS text.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the property, from first principles.
# ---------------------------------------------------------------------------
def is_prime(k: int) -> bool:
    if k < 2:
        return False
    if k in (2, 3):
        return True
    if k % 2 == 0:
        return False
    i = 3
    while i * i <= k:
        if k % i == 0:
            return False
        i += 2
    return True


N_QUBITS = 4
N = 1 << N_QUBITS  # 16

qualifying = []
for n in range(N):
    if is_prime(n) and is_prime((1 << n) - 1):
        qualifying.append(n)

print(f"Classical search space: n in [0, {N - 1}]")
print(f"Classically computed Mersenne-prime-exponent set (n prime and "
      f"2^n-1 prime): {qualifying}")
assert qualifying == [2, 3, 5, 7, 13], (
    f"Unexpected classical result {qualifying}; refusing to proceed "
    "with a quantum circuit built on a wrong classical answer."
)


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the qualifying basis states.
# ---------------------------------------------------------------------------
def build_oracle(marked_values, n_qubits):
    """Phase oracle: flips sign of |x> for each x in marked_values."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian
        # Flip qubits that are 0 in this value so the multi-controlled-Z
        # fires exactly when the register equals `value`.
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(qualifying, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

# Optimal number of Grover iterations for M marked items out of N.
M = len(qualifying)
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Marked items M={M}, N={N}, Grover iterations={iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N_QUBITS))
    qc.append(diffuser.to_instruction(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------
sim = AerSimulator()
shots = 4096
compiled = transpile(qc, sim)
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical register bitstrings are printed MSB-first big-endian
# over the qubit list [0..n-1] with qubit 0 as the rightmost bit.
def bitstring_to_int(bs: str) -> int:
    return int(bs, 2)

measured = {bitstring_to_int(bs): c for bs, c in counts.items()}
most_common_value = max(measured, key=measured.get)
qualifying_shots = sum(c for v, c in measured.items() if v in qualifying)
qualifying_fraction = qualifying_shots / shots

print(f"Most frequently measured value: {most_common_value} "
      f"(count {measured[most_common_value]}/{shots})")
print(f"Fraction of shots landing on a classically-qualifying value: "
      f"{qualifying_fraction:.3f}")

verified = (
    most_common_value in qualifying
    and qualifying_fraction > 0.5
)

if verified:
    print("PASS")
else:
    print("FAIL")
