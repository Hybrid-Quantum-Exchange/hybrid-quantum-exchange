"""
Erdos problem #5 -- quantum-testable instance.

Source: erdosproblems.com problem 5 (data/problems.yaml entry `number: "5"`),
tags ["number theory", "primes"], OEIS id A001223.

OEIS A001223 is the prime gap sequence: a(n) = p(n+1) - p(n), the difference
between consecutive primes, indexed from n = 1 (a(1) = 3 - 2 = 1).

Classical property tested
--------------------------
Take the first 8 gaps of A001223, i.e. the gaps between the first 9 primes
2, 3, 5, 7, 11, 13, 17, 19, 23 (a small, finite, fully computable instance,
N = 8 = 2^3, fitting a 3-qubit search register). Define the search problem:

    find every index n in {0, ..., 7} (0-based) for which a(n) == 4

This is computed classically first, from first principles (trial-division
primality test + direct subtraction -- no OEIS values are copied in), giving
the ground truth marked set. A Grover search circuit is then built whose
oracle marks exactly those basis states (wired from the classically computed
set, not hand-picked), and run on the ideal AerSimulator. The number of
Grover iterations is chosen from the classically known count of marked items
(3 out of 8), which is the standard way to size a multi-solution Grover
search. The test passes if the quantum circuit's high-probability measurement
outcomes are exactly the classically computed marked index set.

This is a genuine amplitude-amplification search (Grover), not just an
arithmetic circuit -- it searches an unstructured 3-bit index space for the
positions where the classical predicate "prime gap equals 4" holds, and its
oracle is derived mechanically from the classical computation rather than
copied from a table.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def first_n_primes(n):
    """Return the first n primes via classical trial division."""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def prime_gaps(n_gaps):
    """Classically compute the first n_gaps terms of OEIS A001223."""
    primes = first_n_primes(n_gaps + 1)
    return [primes[i + 1] - primes[i] for i in range(n_gaps)]


# ---- Classical ground truth -------------------------------------------------

NUM_QUBITS = 3
N = 2 ** NUM_QUBITS  # 8 indices, 3-qubit search register

gaps = prime_gaps(N)
TARGET_GAP = 4
marked_indices = sorted(i for i, g in enumerate(gaps) if g == TARGET_GAP)

print(f"First {N} terms of OEIS A001223 (prime gaps): {gaps}")
print(f"Classically marked indices where gap == {TARGET_GAP}: {marked_indices}")

M = len(marked_indices)
assert 0 < M < N, "Grover search needs a nontrivial, proper subset marked"


# ---- Grover oracle built mechanically from the classical marked set --------

def apply_multi_controlled_z_on_bits(qc, qubits):
    """Apply a Z controlled on all qubits being |1> (multi-controlled Z)."""
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def oracle(qc, index, qubits):
    """Flip the phase of basis state |index> (little-endian bit order)."""
    bits = format(index, f"0{len(qubits)}b")[::-1]  # bit i -> qubits[i]
    flip_qubits = [q for q, b in zip(qubits, bits) if b == "0"]
    if flip_qubits:
        qc.x(flip_qubits)
    apply_multi_controlled_z_on_bits(qc, qubits)
    if flip_qubits:
        qc.x(flip_qubits)


def diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    apply_multi_controlled_z_on_bits(qc, qubits)
    qc.x(qubits)
    qc.h(qubits)


qubits = list(range(NUM_QUBITS))
qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(qubits)

# Standard optimal iteration count for M marked items out of N.
iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))
print(f"Grover iterations used: {iterations} (N={N}, M={M})")

for _ in range(iterations):
    for idx in marked_indices:
        oracle(qc, idx, qubits)
    diffuser(qc, qubits)

qc.measure(qubits, qubits)

# ---- Run on the ideal AerSimulator ------------------------------------------

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-bit string already has bit 0 as its rightmost
# character, i.e. the same convention as int(bitstring, 2).
counted = {}
for bitstring, c in counts.items():
    idx = int(bitstring, 2)
    counted[idx] = counted.get(idx, 0) + c

sorted_counts = sorted(counted.items(), key=lambda kv: -kv[1])
print("Measurement counts by index (most frequent first):")
for idx, c in sorted_counts:
    marker = " <- classically marked" if idx in marked_indices else ""
    print(f"  index {idx}: {c}/{shots}{marker}")

# The top-M most measured outcomes should be exactly the classically marked set.
top_m_indices = set(idx for idx, _ in sorted_counts[:M])
quantum_marked = top_m_indices

verified = quantum_marked == set(marked_indices)

print()
print(f"Classical marked set : {sorted(marked_indices)}")
print(f"Quantum-found set    : {sorted(quantum_marked)}")

if verified:
    print("PASS")
else:
    print("FAIL")
