"""
Erdos problem #764 -- quantum-testable-sequence lane (honest non-match).

Source record: /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: \"764\"" (number theory / additive combinatorics), status
"disproved" (2025-08-31), field:

    oeis: ["N/A"]

LIMITATION (read before trusting anything below): problem #764 has NO OEIS
sequence attached in the source data ("N/A"). There is therefore no genuine
OEIS-derived sequence property to encode as a quantum oracle for this problem
specifically, and this script does NOT test problem 764's mathematical
content, its statement, or any term of a sequence claiming to represent it.
Per the task instructions for the no-OEIS-id case, this script instead
contains a best-honest-attempt, genuinely computed small quantum circuit on a
property in the same subject area listed in problem 764's tags ("number
theory", "additive combinatorics"), clearly labeled as illustrative and NOT a
verification of problem 764 itself.

Illustrative property actually tested (real math, checked classically here
from first principles, not copied from any table):

    Over the universe U = {0, 1, ..., 15} (4 bits), a number x is "prime" in
    the elementary sense (has exactly two distinct positive divisors, 1 and
    itself). We classically enumerate the exact set of primes in U:

        primes(U) = {2, 3, 5, 7, 11, 13}

    and build a real Grover-search circuit whose oracle marks precisely the
    x in U with x in primes(U), then run the circuit on the ideal AerSimulator
    and check that the highest-probability measured outcomes are exactly
    primes(U).

This is a genuine, independently checkable finite computation (primality is
decidable and the search space is 2^4 = 16), run through a real amplitude-
amplification (Grover) circuit -- not a fabricated or copied OEIS value --
but it is explicitly NOT a claim about Erdos problem #764, whose sequence
does not exist in the source data to be tested.
"""

import math
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
UNIVERSE = list(range(2 ** N_QUBITS))  # 0..15
CLASSICAL_PRIMES = sorted(x for x in UNIVERSE if is_prime(x))
print(f"Classical ground truth: primes in 0..15 = {CLASSICAL_PRIMES}")
assert CLASSICAL_PRIMES == [2, 3, 5, 7, 11, 13], "sanity check on classical primality failed"


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the classical prime bitstrings.
# ---------------------------------------------------------------------------

def build_oracle(marked_values, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
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


def build_grover_circuit(marked_values, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# Optimal number of Grover iterations for N=16 items, M=6 marked.
N_ITEMS = len(UNIVERSE)
M_MARKED = len(CLASSICAL_PRIMES)
theta = math.asin(math.sqrt(M_MARKED / N_ITEMS))
optimal_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {optimal_iterations}")

circuit = build_grover_circuit(CLASSICAL_PRIMES, N_QUBITS, optimal_iterations)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

simulator = AerSimulator()
compiled = transpile(circuit, simulator)
SHOTS = 4096
result = simulator.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-first (qubit n-1 ... qubit 0); our encoding
# above treated bit i (little-endian) as qubit i, so convert back to ints
# consistently with that same convention.
def bitstring_to_int(bs: str) -> int:
    # bs is qiskit's c[n-1]...c[0] string; reverse to get little-endian bit i
    le = bs[::-1]
    return int(le[::-1], 2) if False else int(bs, 2)

value_counts = {}
for bitstring, count in counts.items():
    value_counts[int(bitstring, 2)] = value_counts.get(int(bitstring, 2), 0) + count

# Take the top-M_MARKED most frequent measured values as Grover's answer.
sorted_values = sorted(value_counts.items(), key=lambda kv: kv[1], reverse=True)
quantum_top = sorted(v for v, _ in sorted_values[:M_MARKED])

marked_prob = sum(c for v, c in value_counts.items() if v in CLASSICAL_PRIMES) / SHOTS
print(f"Quantum top-{M_MARKED} measured values: {quantum_top}")
print(f"Total probability mass on classically-prime outcomes: {marked_prob:.3f}")

verified = (quantum_top == CLASSICAL_PRIMES) and (marked_prob > 0.5)

print("PASS" if verified else "FAIL")

if not verified:
    sys.exit(1)
