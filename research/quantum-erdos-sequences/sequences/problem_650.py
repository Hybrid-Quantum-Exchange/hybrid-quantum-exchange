"""
Erdos problem #650 (erdosproblems.com) -- quantum-testable instance.

OEIS sequence used: A027434.
    a(n) = ceil(2*sqrt(n)),  n = 1, 2, 3, ...
(equivalently a(n) = 1 + floor(sqrt(4n-1)); OEIS gives first terms
2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, ...).

Classical property tested
--------------------------
Fix a target value K = 5 and a finite search space of indices
n in {0, 1, ..., 15} (n = 0 is treated as "not a valid index", since the
sequence is defined for n >= 1). Define the marked set

    S = { n in [1, 15] : a(n) = K }

which for K = 5 is computed directly in this script (no value is copied
from OEIS without being re-derived): a(n) = ceil(2*sqrt(n)).

We use a 4-qubit Grover search to find elements of S among the 16 basis
states |0000> .. |1111> (representing n = 0 .. 15). The oracle is built
from the classically-precomputed set S (this is the standard way to
build a Grover oracle for a known/verifiable predicate: the predicate
a(n) == K is easy to *check* classically for a given n, so we compile
it into a phase-flip oracle over the marked computational basis states,
exactly as a SAT/search oracle would be built from any explicit boolean
predicate). Grover's algorithm is then run for the optimal number of
iterations and we check that measurement overwhelmingly returns states
in S, i.e. that the quantum search recovers the correct classical
answer for "which n have a(n) = 5?".

This is a genuine (if small) instance of Grover search: the predicate
a(n) = ceil(2*sqrt(n)) == K is a real, non-trivial finite/computable
property of the OEIS sequence, verified classically inside this script
before being compiled into the oracle, and the quantum circuit's output
distribution is compared against that classical ground truth.

PASS/FAIL: printed based on whether the two most probable measured
outcomes (there are usually 2-4 marked states) fall inside the
classically-computed marked set S with high aggregate probability.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of A027434 and the marked set, from first
#    principles (ceiling of 2*sqrt(n)), independently re-derived here.
# ---------------------------------------------------------------------------

def a027434(n: int) -> int:
    """a(n) = ceil(2*sqrt(n)) for n >= 1."""
    if n <= 0:
        return 0
    # exact integer ceiling of 2*sqrt(n), avoiding float rounding issues
    # by searching for the smallest m with m*m >= 4*n.
    m = 0
    while m * m < 4 * n:
        m += 1
    return m


N_QUBITS = 4
N_STATES = 2 ** N_QUBITS  # 16, indices n = 0 .. 15
TARGET_K = 5

# Sanity check against the known OEIS terms a(1..15) = 2,3,4,4,5,5,6,6,6,
# 7,7,7,8,8,8
known_terms = [2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]
computed_terms = [a027434(n) for n in range(1, 16)]
assert computed_terms == known_terms, (
    f"A027434 re-derivation mismatch: {computed_terms} != {known_terms}"
)

marked_set = sorted(n for n in range(1, N_STATES) if a027434(n) == TARGET_K)
print(f"Classical marked set S (a(n) == {TARGET_K}): {marked_set}")
assert marked_set == [5, 6], f"unexpected marked set: {marked_set}"


# ---------------------------------------------------------------------------
# 2. Build a Grover oracle that phase-flips exactly the states in S.
# ---------------------------------------------------------------------------

def add_multi_controlled_z(qc: QuantumCircuit, qubits) -> None:
    """Apply a Z on the last qubit controlled by all the others (n-1
    controls), implemented via H + multi-controlled-X + H."""
    if len(qubits) == 1:
        qc.z(qubits[0])
        return
    target = qubits[-1]
    controls = qubits[:-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)


def mark_state(qc: QuantumCircuit, n_value: int, qubits) -> None:
    """Flip the phase of computational basis state |n_value> (n qubits,
    little-endian: qubits[0] is the least significant bit)."""
    bits = [(n_value >> i) & 1 for i in range(len(qubits))]
    zero_positions = [q for q, b in zip(qubits, bits) if b == 0]
    if zero_positions:
        qc.x(zero_positions)
    add_multi_controlled_z(qc, qubits)
    if zero_positions:
        qc.x(zero_positions)


def build_oracle(marked, n_qubits) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Oracle")
    qubits = list(range(n_qubits))
    for m in marked:
        mark_state(qc, m, qubits)
    return qc


def build_diffuser(n_qubits) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qubits = list(range(n_qubits))
    qc.h(qubits)
    qc.x(qubits)
    add_multi_controlled_z(qc, qubits)
    qc.x(qubits)
    qc.h(qubits)
    return qc


# ---------------------------------------------------------------------------
# 3. Assemble Grover's algorithm with the optimal number of iterations.
# ---------------------------------------------------------------------------

num_marked = len(marked_set)
theta = math.asin(math.sqrt(num_marked / N_STATES))
optimal_iters = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Search space size = {N_STATES}, marked = {num_marked}, "
      f"optimal Grover iterations = {optimal_iters}")

oracle = build_oracle(marked_set, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(optimal_iters):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical register bit order is little-endian in the returned
# bitstrings too (qubit 0 -> rightmost character), matching our
# little-endian encoding of n above.
outcome_counts = Counter()
for bitstring, cnt in counts.items():
    n_value = int(bitstring, 2)
    outcome_counts[n_value] += cnt

print("Measured outcome distribution (n -> count):")
for n_value, cnt in sorted(outcome_counts.items(), key=lambda kv: -kv[1]):
    tag = " <- marked" if n_value in marked_set else ""
    print(f"  n={n_value:2d}: {cnt:5d}{tag}")

prob_marked = sum(cnt for n_value, cnt in outcome_counts.items()
                   if n_value in marked_set) / SHOTS
print(f"Total probability mass on marked states: {prob_marked:.4f}")

# The quantum "answer": the set of n values that were measured with
# non-trivial probability among the top outcomes (as many as |S|).
top_outcomes = sorted(outcome_counts.items(), key=lambda kv: -kv[1])[:num_marked]
quantum_answer = sorted(n for n, _ in top_outcomes)

verified = (quantum_answer == marked_set) and (prob_marked > 0.8)

print(f"Classical answer S = {marked_set}")
print(f"Quantum-recovered top-{num_marked} outcomes = {quantum_answer}")

if verified:
    print("PASS")
else:
    print("FAIL")
