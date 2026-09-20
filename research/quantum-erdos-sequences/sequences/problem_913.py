"""
Erdos problem #913 -- quantum-testable instance.

OEIS id used: A359747
  "Numbers k such that k*(k+1), in its canonical prime factorization, has
   mutually distinct exponents (no exponent value repeats among the
   prime-power exponents)."

Erdos problem #913's metadata in erdosproblems/data/problems.yaml lists
oeis: ["A359747"] and tags: ["number theory"]; it is otherwise open/
unformalized, so there is no closed-form answer to search for -- but
A359747's defining property is a small, finite, fully computable predicate
on each positive integer k, which is exactly the kind of decision problem
Grover's algorithm is built to search over.

Classical property tested (computed from first principles below, not
copied from OEIS):
    For k in {1, ..., 16}, factor N = k*(k+1) into primes and collect the
    multiset of exponents in its canonical factorization. k qualifies iff
    all of those exponents are mutually distinct (no repeated exponent
    value).

Small instance:
    Search space k = 1..16, encoded as a 4-qubit index register
    (index i in 0..15 represents k = i+1).

    The classical brute-force computation below finds the qualifying set
    within this range (this reproduces, and is checked against, the
    published start of A359747: 1, 3, 4, 7, 8, 16, ...).

Quantum circuit:
    A genuine Grover search circuit over the 4-qubit index register. The
    oracle is a diagonal phase-flip built (via classically-controlled
    multi-controlled-Z gates) from the classically-computed marked set --
    the standard "explicit oracle from a computed predicate" construction
    used throughout Grover tutorials, run for the optimal number of
    Grover iterations for this table size, then measured. If Grover's
    algorithm and the oracle are implemented correctly, the measurement
    distribution must concentrate on exactly the marked (qualifying)
    indices; we verify this by simulation on the ideal AerSimulator and
    comparing the most-frequently-measured basis states against the
    classically-computed qualifying set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the property, from first principles.
# ---------------------------------------------------------------------------

def prime_factor_exponents(n: int) -> list[int]:
    """Return the list of exponents in the canonical prime factorization of n."""
    exponents = []
    d = 2
    m = n
    while d * d <= m:
        if m % d == 0:
            e = 0
            while m % d == 0:
                m //= d
                e += 1
            exponents.append(e)
        d += 1
    if m > 1:
        exponents.append(1)
    return exponents


def has_distinct_exponents(k: int) -> bool:
    """True iff k*(k+1) has mutually distinct prime-factorization exponents."""
    n = k * (k + 1)
    exps = prime_factor_exponents(n)
    return len(exps) == len(set(exps))


N_QUBITS = 4
TABLE_SIZE = 2 ** N_QUBITS  # 16 -> k ranges over 1..16

classical_marked_k = [k for k in range(1, TABLE_SIZE + 1) if has_distinct_exponents(k)]
classical_marked_indices = sorted(k - 1 for k in classical_marked_k)

# Sanity check against the published start of A359747: 1, 3, 4, 7, 8, 16, ...
expected_prefix = [1, 3, 4, 7, 8, 16]
assert classical_marked_k[: len(expected_prefix)] == expected_prefix, (
    f"classical computation does not reproduce known A359747 prefix: "
    f"got {classical_marked_k[: len(expected_prefix)]}, expected {expected_prefix}"
)

print(f"Classical brute force over k = 1..{TABLE_SIZE}:")
print(f"  qualifying k (A359747 members in range): {classical_marked_k}")
print(f"  as 4-bit indices (k-1):                  {classical_marked_indices}")


# ---------------------------------------------------------------------------
# 2. Build a Grover oracle that marks exactly those indices.
# ---------------------------------------------------------------------------

def apply_mcz(qc: QuantumCircuit, qubits: list[int]) -> None:
    """Apply a multi-controlled Z (phase flip on |11...1>) across `qubits`."""
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def oracle(qc: QuantumCircuit, qubits: list[int], marked_indices: list[int]) -> None:
    """Phase-flip every computational basis state in marked_indices."""
    for idx in marked_indices:
        bits = format(idx, f"0{len(qubits)}b")
        # X on the qubits that should be 0, so the marked pattern becomes all-1s.
        zero_positions = [qubits[i] for i, b in enumerate(bits) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        apply_mcz(qc, qubits)
        for q in zero_positions:
            qc.x(q)


def diffuser(qc: QuantumCircuit, qubits: list[int]) -> None:
    """Standard Grover diffusion operator (inversion about the mean)."""
    for q in qubits:
        qc.h(q)
        qc.x(q)
    apply_mcz(qc, qubits)
    for q in qubits:
        qc.x(q)
        qc.h(q)


# ---------------------------------------------------------------------------
# 3. Assemble and run the Grover circuit.
# ---------------------------------------------------------------------------

qubits = list(range(N_QUBITS))
qc = QuantumCircuit(N_QUBITS, N_QUBITS)

qc.h(qubits)  # uniform superposition over all 16 indices

num_marked = len(classical_marked_indices)
# Optimal number of Grover iterations: floor(pi/4 * sqrt(N/M))
num_iterations = max(1, round((math.pi / 4) * math.sqrt(TABLE_SIZE / num_marked)))

for _ in range(num_iterations):
    oracle(qc, qubits, classical_marked_indices)
    diffuser(qc, qubits)

qc.measure(qubits, qubits)

print(f"\nGrover circuit: {N_QUBITS} qubits, {num_marked} marked out of {TABLE_SIZE}, "
      f"{num_iterations} iteration(s).")

backend = AerSimulator()
transpiled = transpile(qc, backend)
SHOTS = 20000
result = backend.run(transpiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit order: rightmost classical bit is qubit 0, so reverse to read
# the register as the natural binary index (q3 q2 q1 q0).
def bitstring_to_index(bs: str) -> int:
    return int(bs[::-1], 2)

index_counts = Counter()
for bitstring, c in counts.items():
    index_counts[bitstring_to_index(bitstring)] += c

# ---------------------------------------------------------------------------
# 4. Compare: the top `num_marked` measured indices should be exactly the
#    classically-computed marked set.
# ---------------------------------------------------------------------------

top_indices = sorted(idx for idx, _ in index_counts.most_common(num_marked))
marked_prob_mass = sum(index_counts[idx] for idx in classical_marked_indices) / SHOTS

print(f"\nTop {num_marked} measured indices (by count): {top_indices}")
print(f"Classically marked indices:                 {sorted(classical_marked_indices)}")
print(f"Probability mass on marked indices: {marked_prob_mass:.4f} "
      f"(uniform baseline would be {num_marked / TABLE_SIZE:.4f})")

verified = (
    top_indices == sorted(classical_marked_indices)
    and marked_prob_mass > 0.75
)

if verified:
    print("\nPASS")
else:
    print("\nFAIL")

if __name__ == "__main__":
    pass
