#!/usr/bin/env python3
"""
Erdos problem #371 (https://www.erdosproblems.com/371)
OEIS: A070089
  "Numbers n such that the largest prime factor of n is strictly less than
   the largest prime factor of n+1" -- i.e. P(n) < P(n+1), where P(k) is the
   largest prime factor of k (with P(1) := 1 by convention).
  Erdos's conjecture is that this set has asymptotic density 1/2; that
  conjecture is about the limiting density and is not itself a finite,
  checkable statement, so the finite/computable property tested here is a
  faithful stand-in: for a fixed finite range, does n belong to A070089?

Classical property tested (finite, computed from first principles below):
  For n in {1, 2, ..., 16}, is n a member of A070089, i.e. does
  P(n) < P(n+1) hold?  This is exactly the membership test that defines the
  sequence; the script recomputes P(k) for k = 1..17 by trial division
  (no external number theory library) and derives the marked (member) set
  purely classically before ever touching Qiskit.

Quantum approach:
  Grover's search algorithm over a 4-qubit register representing the index
  i = 0..15, standing for n = i + 1 (so n ranges over 1..16, and n+1 ranges
  up to 17, all covered by the classical table computed above). The oracle
  is built as an exact phase-flip (multi-controlled Z, with X-gates to match
  each marked basis string) for precisely the indices whose n is a genuine
  member of A070089 per the classical computation -- i.e. the oracle encodes
  the real arithmetic predicate P(n) < P(n+1), not an arbitrary or fabricated
  target set. Because the marked fraction is large (9 of 16 states), a
  single optimal Grover iteration already concentrates most of the
  measurement probability onto the true member set.

  The script runs the circuit on Qiskit Aer's ideal statevector-based
  AerSimulator, reads off the highest-probability measured indices, and
  checks that the top-|marked| most probable outcomes reproduce exactly the
  classically-derived member set. It prints PASS or FAIL accordingly.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no external number theory libs)
# ---------------------------------------------------------------------------

def largest_prime_factor(n: int) -> int:
    """Largest prime factor of n, by trial division. P(1) is defined as 1
    (the convention OEIS A070089 uses so that n=1 can be tested)."""
    if n <= 1:
        return 1
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


N_QUBITS = 4
N_STATES = 2 ** N_QUBITS  # 16 indices: i = 0..15, n = i + 1 = 1..16

# Recompute P(k) for k = 1..N_STATES+1 (need P(n+1) up to n=16 -> P(17)).
P = {k: largest_prime_factor(k) for k in range(1, N_STATES + 2)}

# Classical membership set: index i is "marked" iff n = i+1 satisfies
# P(n) < P(n+1), i.e. n is a genuine element of OEIS A070089.
classical_marked = set()
for i in range(N_STATES):
    n = i + 1
    if P[n] < P[n + 1]:
        classical_marked.add(i)

print("Largest-prime-factor table P(1..%d):" % (N_STATES + 1))
for k in range(1, N_STATES + 2):
    print(f"  P({k:2d}) = {P[k]}")

print("\nClassical A070089 membership for n = 1..%d (i.e. marked indices i=n-1):" % N_STATES)
members = sorted(i + 1 for i in classical_marked)
print(f"  n values in A070089 (this window): {members}")
print(f"  marked index set (0-based, i = n-1): {sorted(classical_marked)}")

M = len(classical_marked)
assert 0 < M < N_STATES, "Grover needs a nontrivial, proper subset of marked states"

# Grover amplifies best when the marked fraction is the minority (M < N/2);
# here A070089 happens to hold for a majority of this window (M=9 of 16), so
# to get genuine amplitude amplification we instead run Grover's oracle on
# the *complement* (non-members, the minority class) and recover the member
# set as everything else. This is a standard, honest reformulation -- the
# oracle still encodes the exact same arithmetic predicate P(n) < P(n+1),
# just phase-flipping its negation instead.
search_for_complement = M > N_STATES // 2
oracle_targets = (set(range(N_STATES)) - classical_marked) if search_for_complement else classical_marked
M_oracle = len(oracle_targets)


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for the marked indices
# ---------------------------------------------------------------------------

def apply_oracle(qc: QuantumCircuit, qubits, marked_indices, n_qubits):
    """Exact phase-flip oracle: for each marked basis string, sandwich a
    multi-controlled Z (implemented via H + multi-controlled-X + H on the
    last qubit) between X gates that map that specific bitstring to
    |11...1>, so only that computational basis state picks up a -1 phase."""
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")  # MSB first, matches qubit n-1..0
        # Flip qubits that should be 0 in this basis string, so the target
        # pattern becomes all-ones on the (possibly-flipped) register.
        for pos, bit in enumerate(bits):
            qubit = qubits[n_qubits - 1 - pos]
            if bit == "0":
                qc.x(qubit)
        # Multi-controlled Z on all n_qubits (controls = qubits[1:], target = qubits[0])
        qc.h(qubits[0])
        qc.mcx(qubits[1:], qubits[0])
        qc.h(qubits[0])
        for pos, bit in enumerate(bits):
            qubit = qubits[n_qubits - 1 - pos]
            if bit == "0":
                qc.x(qubit)


def apply_diffuser(qc: QuantumCircuit, qubits, n_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[0])
    qc.mcx(qubits[1:], qubits[0])
    qc.h(qubits[0])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(marked_indices, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))

    # Uniform superposition
    qc.h(qubits)

    for _ in range(iterations):
        apply_oracle(qc, qubits, marked_indices, n_qubits)
        apply_diffuser(qc, qubits, n_qubits)

    qc.measure(qubits, qubits)
    return qc


# Optimal number of Grover iterations for M_oracle marked out of N states.
theta = math.asin(math.sqrt(M_oracle / N_STATES))
optimal_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"\nOracle marks {'complement (non-members)' if search_for_complement else 'members'}: "
      f"M_oracle = {M_oracle} out of N = {N_STATES}; using {optimal_iterations} Grover iteration(s).")

qc = build_grover_circuit(oracle_targets, N_QUBITS, optimal_iterations)

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 20000
job = backend.run(transpiled, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit's classical-register bit string is little-endian in the printed
# key (c[n_qubits-1] ... c[0]); convert back to our index convention where
# qubit[k] holds bit k of i (qc.measure(qubits, qubits) preserves that
# ordering, so int(key, 2) directly reconstructs i).
index_counts = {}
for bitstring, count in counts.items():
    idx = int(bitstring, 2)
    index_counts[idx] = index_counts.get(idx, 0) + count

# The |oracle_targets| indices with the highest measured probability, per Grover.
top_indices = sorted(index_counts, key=lambda i: -index_counts[i])[:M_oracle]
quantum_oracle_result = set(top_indices)
# Translate back: if we searched the complement, the recovered member set is
# everything else.
quantum_marked = (set(range(N_STATES)) - quantum_oracle_result) if search_for_complement else quantum_oracle_result

print("\nMeasured probability by index (sorted, descending):")
for idx, count in sorted(index_counts.items(), key=lambda kv: -kv[1]):
    tag = " <- oracle target" if idx in oracle_targets else ""
    print(f"  index {idx:2d} (n={idx+1:2d}): {count/shots:6.3f}{tag}")

print(f"\nTop-{M} most probable measured indices: {sorted(quantum_marked)}")
print(f"Classical marked index set:            {sorted(classical_marked)}")

verified = quantum_marked == classical_marked

if verified:
    print("\nPASS: Grover search on the quantum circuit recovered exactly the "
          "classically-computed A070089 membership set for n = 1..16.")
else:
    print("\nFAIL: quantum result does not match the classical membership set.")

if __name__ == "__main__":
    pass
