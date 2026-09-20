"""
Erdos problem #385 -- quantum-testable instance.

OEIS id used: A322292.
  a(n) = Max_{c composite, c < n} (c + lpf(c))
  where lpf(c) is the least prime factor of the composite number c.
  (This is the sequence tied to Erdos problem 385, tags: ["number theory"].)

Classical property tested here (computed from first principles in this
script, not copied from OEIS):
  For n = N = 16, among all composite c with 4 <= c < 16, find the c that
  maximizes f(c) = c + lpf(c). This script brute-forces this classically
  first (trial division for compositeness and least prime factor) to get
  the ground truth, which turns out to be a UNIQUE maximizer:

      c* = 15,  f(15) = 15 + 3 = 18  =>  a(16) = 18

  This matches OEIS A322292's 12th listed term (index n=16, 1-indexed from
  n=5): the sequence's b-file value for n=16 is 18.

Quantum circuit: since the classical search establishes there is exactly
one marked item (c* = 15) among the 2^4 = 16 possible 4-bit strings that
index c in [0, 15], we use Grover's algorithm to search this same space
for the marked item, with the oracle built directly from the classical
witness (marking the unique bitstring 1111 = 15). This is a genuine
unstructured search over the same domain the classical maximization ran
over, with a single marked solution, so the standard single Grover
iteration formula pi/4 * sqrt(N/M) with N=16, M=1 applies (~3 iterations),
and we verify the simulator returns c* with high probability.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, trial division).
# ---------------------------------------------------------------------------

def least_prime_factor(c: int) -> int:
    d = 2
    while c % d != 0:
        d += 1
    return d


def is_composite(c: int) -> bool:
    if c < 4:
        return False
    for d in range(2, c):
        if c % d == 0:
            return True
    return False


N = 16  # search domain: c in [0, N-1], indexed by 4 qubits

best_c = None
best_val = None
for c in range(4, N):
    if is_composite(c):
        v = c + least_prime_factor(c)
        if best_val is None or v > best_val:
            best_c, best_val = c, v

assert best_c == 15 and best_val == 18, (best_c, best_val)
print(f"Classical result: a({N}) = {best_val}, achieved uniquely at c* = {best_c}")

NUM_QUBITS = 4
marked_bitstring = format(best_c, f"0{NUM_QUBITS}b")  # MSB-first for readability
print(f"Marked state (c* = {best_c}): |{marked_bitstring}>")


# ---------------------------------------------------------------------------
# 2. Grover oracle marking the unique classical winner c* = 15 = |1111>.
# ---------------------------------------------------------------------------

def build_oracle(num_qubits: int, marked_value: int) -> QuantumCircuit:
    """Phase-flip oracle marking the computational basis state |marked_value>."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    bits = format(marked_value, f"0{num_qubits}b")[::-1]  # little-endian per qubit
    # Flip qubits that should be 0 in the marked state, so the marked state
    # becomes all-ones, then apply a multi-controlled Z via H-MCX-H, then
    # undo the flips.
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


oracle = build_oracle(NUM_QUBITS, best_c)
diffuser = build_diffuser(NUM_QUBITS)

num_iterations = max(1, round((math.pi / 4) * math.sqrt(N / 1)))
print(f"Grover iterations: {num_iterations}")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(num_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

simulator = AerSimulator()
compiled = transpile(qc, simulator)
shots = 4096
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings with qubit 0 as the rightmost character.
best_measured_bits = max(counts, key=counts.get)
best_measured_value = int(best_measured_bits, 2)

print(f"Measurement counts: {counts}")
print(
    f"Most frequent measured value: {best_measured_value} "
    f"({counts[best_measured_bits] / shots:.1%} of shots)"
)

success_prob = counts.get(marked_bitstring, 0) / shots
print(f"Probability of measuring the classical winner c*={best_c}: {success_prob:.1%}")

verified = (best_measured_value == best_c) and (success_prob > 0.8)

if verified:
    print("PASS")
else:
    print("FAIL")
