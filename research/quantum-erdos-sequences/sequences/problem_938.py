"""
Erdos problem #938 -- quantum-testable instance.

Problem #938 (erdosproblems.com/938) concerns powerful numbers, and its
metadata lists OEIS sequence A001694 (with A076446 as a related sequence).
A001694 is the sequence of POWERFUL NUMBERS: positive integers n such that
for every prime p dividing n, p^2 also divides n (equivalently, n = a^2 * b^3
for some positive integers a, b). The sequence begins:
    1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 72, 81, 100, 108, 121, 125, ...

Classical property tested here
-------------------------------
Search space: integers n in [0, 15] (4 bits, N = 16).
Property: "n is a powerful number" (n >= 1 and every prime factor of n
appears with exponent >= 2).

This script:
  1. Computes, from first principles (trial-division factorization, no
     lookup of the OEIS b-file), the exact classical set of powerful
     numbers in [0, 15]: {1, 4, 8, 9}.
  2. Builds a Grover search circuit (4 qubits, oracle + diffuser, optimal
     number of iterations for 4 marked items out of 16) whose oracle marks
     exactly the powerful numbers in that range, using a phase oracle built
     from a multi-controlled-Z gate per marked basis state.
  3. Runs the circuit on the ideal AerSimulator and checks that the most
     frequently measured basis states are exactly the classical powerful
     numbers, i.e. that Grover search over "is this a powerful number?"
     converges on the classically-correct answer set.
  4. Prints PASS or FAIL based on that comparison.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, trial division).
# ---------------------------------------------------------------------------

def is_powerful(n: int) -> bool:
    """True iff n >= 1 and every prime factor of n has exponent >= 2."""
    if n < 1:
        return False
    if n == 1:
        return True
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            exponent = 0
            while m % p == 0:
                m //= p
                exponent += 1
            if exponent < 2:
                return False
        p += 1
    if m > 1:
        # m is a leftover prime factor with exponent exactly 1.
        return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # search space size: 0..15

classical_powerful = sorted(n for n in range(N) if is_powerful(n))
print(f"Classical powerful numbers in [0, {N - 1}]: {classical_powerful}")
assert classical_powerful == [1, 4, 8, 9], "classical computation changed unexpectedly"


# ---------------------------------------------------------------------------
# 2. Grover search circuit.
# ---------------------------------------------------------------------------

def bits_of(n: int, width: int):
    return [(n >> i) & 1 for i in range(width)]


def add_oracle(qc: QuantumCircuit, marked, qubits):
    """Phase oracle: flips the sign of |n> for each n in `marked`."""
    width = len(qubits)
    for n in marked:
        b = bits_of(n, width)
        # Map |n> onto |11...1> via X on zero-bits, apply multi-controlled Z,
        # then undo the X's.
        for i, bit in enumerate(b):
            if bit == 0:
                qc.x(qubits[i])
        if width == 1:
            qc.z(qubits[0])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for i, bit in enumerate(b):
            if bit == 0:
                qc.x(qubits[i])


def add_diffuser(qc: QuantumCircuit, qubits):
    width = len(qubits)
    for q in qubits:
        qc.h(q)
        qc.x(q)
    if width == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


marked_count = len(classical_powerful)
# Optimal number of Grover iterations for M marked items out of N.
theta = math.asin(math.sqrt(marked_count / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))
qc.h(qubits)

for _ in range(iterations):
    add_oracle(qc, classical_powerful, qubits)
    add_diffuser(qc, qubits)

qc.measure(qubits, qubits)

print(f"Grover iterations used: {iterations} (marked={marked_count}, N={N})")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 20000
job = backend.run(transpiled, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit's bitstring convention already has the leftmost character as the
# highest-index (qu/cl)bit, i.e. clbit i = qubit i is bit position i of the
# integer -- so interpreting the reported bitstring directly as binary
# yields the integer value n.
freq = {}
for bitstring, count in counts.items():
    n = int(bitstring, 2)
    freq[n] = freq.get(n, 0) + count

sorted_by_freq = sorted(freq.items(), key=lambda kv: -kv[1])
print("Top measured outcomes (value: count):")
for n, c in sorted_by_freq[:8]:
    tag = " <- powerful" if n in classical_powerful else ""
    print(f"  {n:2d}: {c:5d}{tag}")

# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

top_k = sorted_by_freq[:marked_count]
quantum_top_set = set(n for n, _ in top_k)
classical_set = set(classical_powerful)

# Also require each marked outcome to be amplified well above the
# uniform-random baseline (1/N of shots), confirming genuine amplification
# rather than a coincidental measurement.
baseline = shots / N
amplified_enough = all(freq.get(n, 0) > 2 * baseline for n in classical_powerful)

verified = (quantum_top_set == classical_set) and amplified_enough

print()
print(f"Classical answer set : {sorted(classical_set)}")
print(f"Quantum top-{marked_count} outcomes: {sorted(quantum_top_set)}")
print(f"Amplified above baseline ({baseline:.0f} shots/16): {amplified_enough}")

if verified:
    print("PASS")
else:
    print("FAIL")
