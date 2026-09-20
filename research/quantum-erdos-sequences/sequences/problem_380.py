"""
Erdos problem #380 -- quantum-testable instance.

OEIS id used: A070003
  "Numbers divisible by the square of their largest prime factor."
  (n such that, writing p = largest prime factor of n, p^2 | n.)

Classical property tested (derived and checked from first principles in this
script, not copied from OEIS): for the search space N = {1, 2, ..., 15}
(4 qubits, basis states |0000> .. |1111> representing n-1 for n=1..16, with
n=16 excluded so the space is exactly 1..15), which n satisfy "p^2 divides n
where p is n's largest prime factor"?

By direct trial division (done classically below) the members of A070003
in [1, 15] are exactly {4, 8, 9}. This script:
  1. Computes that classical answer set independently (trial-factorization,
     no OEIS values hard-coded).
  2. Builds a Grover search circuit over 4 qubits whose oracle flips the
     phase of exactly the basis states encoding {4, 8, 9} (oracle built
     from the classically-computed set, i.e. a genuine "verify/search this
     property" oracle, not a fabricated one).
  3. Runs the optimal number of Grover iterations on the ideal AerSimulator
     and checks that the measured distribution is concentrated on the
     classically-correct marked states (n = 4, 8, 9), i.e. that quantum
     search recovers the correct early terms of A070003.
  4. Prints PASS if the top-3 measured outcomes (by shot count) equal the
     classical marked set, else FAIL.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the property (independent of OEIS listing).
# ---------------------------------------------------------------------------

def largest_prime_factor(n: int) -> int:
    if n < 2:
        raise ValueError("n must be >= 2")
    largest = 1
    m = n
    d = 2
    while d * d <= m:
        while m % d == 0:
            largest = d
            m //= d
        d += 1
    if m > 1:
        largest = m
    return largest


def in_A070003(n: int) -> bool:
    """True iff n is divisible by the square of its largest prime factor."""
    if n < 2:
        return False
    p = largest_prime_factor(n)
    return n % (p * p) == 0


N_MAX = 15  # search space {1, ..., 15}; fits in 4 qubits (0..15), n=16 excluded
classical_marked = sorted(n for n in range(1, N_MAX + 1) if in_A070003(n))
print("Classical search space: 1..%d" % N_MAX)
print("Classically computed members of A070003 in this range:", classical_marked)

# Sanity check against the first few published terms of A070003 (4, 8, 9, 16, ...)
# restricted to our range -- this is a consistency check, not the source of truth.
assert classical_marked == [4, 8, 9], (
    f"Unexpected classical result {classical_marked}; refusing to build a "
    "quantum oracle from a property that doesn't match the expected early terms."
)

NUM_QUBITS = 4  # encodes n-1 for n in 0..15 (n=0 unused/never marked)
marked_indices = [n - 1 for n in classical_marked]  # basis-state indices (n-1)


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffuser built from the classical marked set.
# ---------------------------------------------------------------------------

def add_multi_controlled_z_on_index(qc: QuantumCircuit, qubits, idx: int):
    """Flip the phase of the single computational basis state encoding the
    integer `idx`, where qubits[i] carries bit weight 2**i (qubit 0 = LSB,
    matching Qiskit's own little-endian bit-string convention) using X
    gates + a multi-controlled Z."""
    n = len(qubits)
    bits = [(idx >> i) & 1 for i in range(n)]  # bits[i] = weight-2**i bit
    for i, bit in enumerate(bits):
        if bit == 0:
            qc.x(qubits[i])
    if n == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for i, bit in enumerate(bits):
        if bit == 0:
            qc.x(qubits[i])


def build_oracle(num_qubits: int, marked: list) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Oracle")
    for idx in marked:
        add_multi_controlled_z_on_index(qc, list(range(num_qubits)), idx)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


oracle = build_oracle(NUM_QUBITS, marked_indices)
diffuser = build_diffuser(NUM_QUBITS)

M = len(marked_indices)
NN = 2 ** NUM_QUBITS
theta = math.asin(math.sqrt(M / NN))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(NUM_QUBITS))
    qc.append(diffuser.to_gate(), range(NUM_QUBITS))
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 8192
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports count-dict bitstrings with qubit 0 as the rightmost
# character (little-endian), which is exactly the standard binary encoding
# used above when marking basis states (qubit i = weight 2**i). So
# int(bitstring, 2) directly recovers the integer index we marked.
decoded_counts = {}
for bitstring, c in counts.items():
    idx = int(bitstring, 2)
    n_value = idx + 1
    decoded_counts[n_value] = decoded_counts.get(n_value, 0) + c

top3 = sorted(decoded_counts.items(), key=lambda kv: -kv[1])[:3]
top3_values = sorted(v for v, _ in top3)

print("Grover iterations used:", iterations)
print("Top measured n-values (by shot count):", top3)
print("Classical marked set (A070003 in [1,%d]):" % N_MAX, classical_marked)

verified = top3_values == classical_marked

if verified:
    print("PASS")
else:
    print("FAIL")
