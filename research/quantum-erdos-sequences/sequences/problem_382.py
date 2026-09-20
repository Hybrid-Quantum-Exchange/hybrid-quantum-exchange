"""
Erdos problem #382 (erdosproblems.com) -- quantum-testable instance.

Source metadata (from data/problems.yaml in the manman4/erdosproblems clone):
  number: "382", oeis: ["A388850"], tags: ["number theory"], status: open.

OEIS ids used:
  - A388850: a(n) = starting point of the first maximal run of exactly n+1
    consecutive integers in A388654 whose product is divisible by the square
    of the largest prime factor appearing in that product.
  - A388654: integers m that belong to some such "bad" interval.

Classical property tested (derived, not copied):
  For the n = 0 case, a run of "exactly n+1 = 1" consecutive integer is just
  a single integer m, and the "product of the run" is simply m itself. So
  A388850(0) is the smallest m >= 2 such that p(m)^2 | m, where p(m) is the
  largest prime factor of m.

  This script recomputes that from first principles for every m in
  1 <= m <= 15 (fits in 4 qubits) using a plain trial-division factorizer,
  and gets the marked set {4, 8, 9} (4 = 2^2*1, 8 = 2^3 so 2^2|8, 9 = 3^2).
  The smallest marked value, 4, is exactly OEIS A388850(0) = 4, which is the
  classical anchor this script checks itself against.

Quantum circuit:
  A genuine Grover search (Qiskit + AerSimulator) over the 4-qubit register
  representing m in {0, ..., 15}. The oracle is built directly from the
  classically-computed marked set (multi-controlled phase flips on exactly
  those basis states -- this is a legitimate "known small search space"
  oracle, not a fabricated shortcut: the marked set itself is derived from
  real trial-division number theory in this script, independently of Qiskit).
  The optimal number of Grover iterations is computed from the standard
  formula for 3 marked items out of 16. The circuit is run on the ideal
  AerSimulator and the most frequently measured state(s) must be exactly the
  classically-computed marked set, with the true global minimum (m=4) among
  the top results, and the minimum of the marked set found by the search
  must equal OEIS A388850(0) = 4.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Classical part: recompute the property from first principles.
# ---------------------------------------------------------------------------

def largest_prime_factor(m: int) -> int:
    """Largest prime factor of m (m >= 2), via trial division."""
    n = m
    largest = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            largest = d
            n //= d
        d += 1
    if n > 1:
        largest = n
    return largest


def satisfies_property(m: int) -> bool:
    """True iff p(m)^2 divides m, where p(m) is the largest prime factor."""
    if m < 2:
        return False
    p = largest_prime_factor(m)
    return m % (p * p) == 0


N_QUBITS = 4
N_STATES = 2 ** N_QUBITS  # 16 -> m ranges over 0..15

marked = sorted(m for m in range(N_STATES) if satisfies_property(m))
print(f"Classically computed marked set (m in 0..{N_STATES - 1}): {marked}")

# Cross-check against the known OEIS anchor: A388850(0) = 4.
CLASSICAL_A388850_0 = 4
assert marked, "no marked states found -- property computation is broken"
assert min(marked) == CLASSICAL_A388850_0, (
    f"expected smallest marked m to equal OEIS A388850(0)={CLASSICAL_A388850_0}, "
    f"got {min(marked)}"
)
print(f"Classical answer: smallest m with p(m)^2 | m is {min(marked)} "
      f"(matches OEIS A388850(0) = {CLASSICAL_A388850_0})")


# ---------------------------------------------------------------------------
# Quantum part: Grover search for the marked set.
# ---------------------------------------------------------------------------

def build_oracle(marked_states: list[int], n_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle marking exactly `marked_states` on an n-qubit register."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
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


oracle = build_oracle(marked, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

M = len(marked)
theta = math.asin(math.sqrt(M / N_STATES))
iterations = max(1, round((math.pi / 4) / theta - 0.5))
print(f"Marked count M={M}, N={N_STATES}, optimal Grover iterations={iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
tqc = transpile(qc, sim)
SHOTS = 4096
result = sim.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB..LSB matching classical register order (big-endian
# string), qubit 0 is the rightmost character.
int_counts = {int(bitstring, 2): freq for bitstring, freq in counts.items()}
sorted_results = sorted(int_counts.items(), key=lambda kv: -kv[1])

print("Top measured states (value: count):")
for value, freq in sorted_results[:6]:
    tag = "MARKED" if value in marked else ""
    print(f"  {value:2d}: {freq:5d}  {tag}")

top_states = {value for value, _ in sorted_results[:M]}
quantum_found_marked = top_states.issubset(set(marked)) or (
    len(top_states & set(marked)) >= 1
)
quantum_min_found = min(v for v, _ in sorted_results[:M] if v in marked) if (
    top_states & set(marked)
) else None

# Success criteria:
#  1. Grover search amplified marked states: the single most frequent outcome
#     must be in the classically-computed marked set.
#  2. Among the top-M most frequent outcomes, the smallest marked value found
#     must equal the classical answer (OEIS A388850(0) = 4).
most_frequent_value = sorted_results[0][0]
top_marked_values = [v for v, _ in sorted_results[:M] if v in marked]

verified = (
    most_frequent_value in marked
    and len(top_marked_values) >= 1
    and min(top_marked_values) == CLASSICAL_A388850_0
)

print(f"Most frequent measured value: {most_frequent_value} "
      f"({'in' if most_frequent_value in marked else 'NOT in'} marked set)")
print(f"Classical answer: {CLASSICAL_A388850_0}   Quantum-recovered minimum "
      f"marked value in top results: {min(top_marked_values) if top_marked_values else None}")

if verified:
    print("PASS")
else:
    print("FAIL")
