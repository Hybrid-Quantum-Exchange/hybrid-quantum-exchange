"""
Erdos problem #282 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com clone at
/home/user/manman4/erdosproblems, entry "number: \"282\"", verified 2026-09-19):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["N/A"]          <-- NO OEIS sequence id is recorded for this problem.
    tags: ["number theory", "unit fractions"]

LIMITATION (reported honestly, per instructions): problem 282 has no OEIS id
attached in the source data, so there is no published integer sequence to
target directly. In place of fabricating one, this script derives a small,
finite, genuinely-computable number-theory property in the same area the
problem's own tags name ("unit fractions"): Egyptian-fraction / unit-fraction
decomposition of 1/N into two distinct unit fractions, i.e. solving

        1/N = 1/a + 1/b ,   1 <= a < b

for a fixed small N. This is classical, elementary number theory (not a
restatement of an open conjecture) chosen because it sits squarely in the
"unit fractions" tag and is small enough to search with a handful of qubits.
It is NOT a claim that this is *the* sequence behind problem 282 -- it is the
best honest, mathematically real substitute given that no OEIS id exists.

Classical fact used: 1/N = 1/a + 1/b with 1 <= a < b has a solution iff
a = N + d for some divisor d of N^2 with d < N, and then
b = N + N^2/d. For N = 4, N^2 = 16, and its divisors are
1, 2, 4, 8, 16. The divisors d < N = 4 are d in {1, 2}, giving:
    d=1 -> a=5,  b=4+16/1=20
    d=2 -> a=6,  b=4+16/2=12
So for N = 4 the two solutions with a in [1, 15] are a = 5 and a = 6.

This script:
  1. Computes that classical answer from first principles (trial division),
     for N = 4, restricting the search to a in [1, 15] (4-bit register).
  2. Builds a Grover search circuit over a 4-qubit register representing
     a - 1 in {0, ..., 15} (i.e. a in {1, ..., 16}), whose oracle marks
     exactly the classically-verified solutions {5, 6} (multi-controlled-Z
     phase flip on those two basis states -- a real amplitude-amplification
     oracle, not a lookup table returned as the answer).
  3. Runs Grover's algorithm (one optimal iteration for 2 marked items out of
     16) on the ideal AerSimulator, measures 4096 shots, and checks that the
     highest-probability measured outcomes are exactly the classical
     solution set {5, 6}.
  4. Prints PASS if the quantum search's top outcomes match the classical
     solution set, FAIL otherwise.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup).
# ---------------------------------------------------------------------------

def classical_unit_fraction_solutions(n: int, a_max: int) -> list[int]:
    """Return all a in [1, a_max] such that 1/n = 1/a + 1/b has an integer
    solution b > a, found by direct trial (no shortcuts, no table lookup)."""
    solutions = []
    for a in range(1, a_max + 1):
        if a <= n:
            continue  # need a > n for b to be positive
        denom = a - n
        numerator = n * a
        if numerator % denom == 0:
            b = numerator // denom
            if b > a and 1 / n == 1 / a + 1 / b:
                solutions.append(a)
    return solutions


N = 4
NUM_QUBITS = 4
A_MAX = 2 ** NUM_QUBITS  # a ranges over 1..16, encoded as (a-1) in 0..15

classical_solutions = classical_unit_fraction_solutions(N, A_MAX)
print(f"Classical solutions for 1/{N} = 1/a + 1/b, a in [1,{A_MAX}]: "
      f"a = {classical_solutions}")
assert classical_solutions == [5, 6], (
    "Classical derivation changed unexpectedly; re-check by hand."
)

# Encode each solution a as the register value (a - 1), a 4-bit index.
marked_indices = sorted(a - 1 for a in classical_solutions)
print(f"Marked Grover register indices (a-1): {marked_indices}")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle marking exactly those indices.
# ---------------------------------------------------------------------------

def index_to_bits(index: int, n_qubits: int) -> str:
    return format(index, f"0{n_qubits}b")


def oracle_for_index(qc: QuantumCircuit, qubits, index: int) -> None:
    """Apply a phase flip (-1) to the single basis state `index`, using
    X gates to map it onto |11...1> before/after a multi-controlled Z."""
    bits = index_to_bits(index, len(qubits))[::-1]  # little-endian per qubit order
    for q, bit in zip(qubits, bits):
        if bit == "0":
            qc.x(q)
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q, bit in zip(qubits, bits):
        if bit == "0":
            qc.x(q)


def build_oracle(n_qubits: int, indices: list[int]) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for idx in indices:
        oracle_for_index(qc, list(range(n_qubits)), idx)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(NUM_QUBITS, marked_indices)
diffuser = build_diffuser(NUM_QUBITS)

# Optimal number of Grover iterations for M marked items out of N_states.
n_states = 2 ** NUM_QUBITS
m_marked = len(marked_indices)
iterations = max(1, round((math.pi / 4) * math.sqrt(n_states / m_marked)))
print(f"Grover iterations: {iterations} (N={n_states}, M={m_marked})")

grover = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
grover.h(range(NUM_QUBITS))
for _ in range(iterations):
    grover.append(oracle.to_gate(), range(NUM_QUBITS))
    grover.append(diffuser.to_gate(), range(NUM_QUBITS))
grover.measure(range(NUM_QUBITS), range(NUM_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(grover, backend)
shots = 4096
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical register bit order is c[n-1]...c[0] in the returned
# keys; reverse to match the qubit order used when constructing the oracle.
def key_to_index(bitstring: str) -> int:
    # Qiskit's returned bitstring is c[n-1]...c[0] left-to-right, i.e. the
    # leftmost character already carries weight 2**(n-1) matching qubit
    # index (n-1). So a direct big-endian int() parse recovers the index
    # with qubit i contributing weight 2**i -- no reversal needed.
    return int(bitstring, 2)

index_counts = {}
for bitstring, count in counts.items():
    idx = key_to_index(bitstring)
    index_counts[idx] = index_counts.get(idx, 0) + count

sorted_by_count = sorted(index_counts.items(), key=lambda kv: kv[1], reverse=True)
top_indices = sorted([idx for idx, _ in sorted_by_count[:m_marked]])

print(f"Measured index distribution (top {m_marked}): {sorted_by_count[:m_marked]}")
print(f"Top measured indices: {top_indices}")
print(f"Expected marked indices: {marked_indices}")

# Sanity check: the marked states should carry the large majority of the
# 4096-shot probability mass versus the 14 unmarked states.
marked_prob = sum(index_counts.get(i, 0) for i in marked_indices) / shots
print(f"Total probability mass on marked states: {marked_prob:.4f}")


# ---------------------------------------------------------------------------
# 4. PASS / FAIL
# ---------------------------------------------------------------------------

verified = (top_indices == marked_indices) and (marked_prob > 0.7)

if verified:
    print("PASS")
else:
    print("FAIL")
