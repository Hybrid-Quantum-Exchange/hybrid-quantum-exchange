"""
Erdos problem #327 -- quantum-testable instance.

Source: erdosproblems.com problem #327 (data/problems.yaml entry
`number: "327"`), tags ["number theory", "unit fractions"], OEIS id A384927.

OEIS A384927 is defined as: a(n) = the maximum size of a subset S of
{1, 2, ..., n} such that for any two distinct elements t, u in S,
(t + u) does NOT divide (t * u).

The divisibility condition (t + u) | (t * u) is exactly the condition
under which the two unit fractions 1/t and 1/u combine to an "integer
reciprocal" relationship: 1/t + 1/u = (t+u)/(t*u), and (t+u) | (t*u) is
precisely what makes (t*u)/(t+u) an integer -- i.e. t and u have a common
"harmonic mean partner" k = t*u/(t+u) that is itself an integer. This is
the unit-fractions condition the OEIS sequence's underlying problem
(#327) is about: which pairs of unit fractions 1/t, 1/u sum to something
whose harmonic-mean denominator is an integer.

CLASSICAL PROPERTY TESTED HERE
-------------------------------
Fix N = 6. Consider all C(6,2) = 15 unordered pairs {t, u} with
1 <= t < u <= N. For each pair, define the boolean predicate

    marked(t, u)  :=  (t + u) divides (t * u)

This script:
  1. Computes marked(t, u) for all 15 pairs directly from first
     principles (integer modulo check) -- no OEIS value is copied.
  2. Encodes the 15 pairs as computational basis states of a 4-qubit
     register (states 0..14; state 15 is unused padding, never marked).
  3. Builds a genuine Grover search circuit whose oracle phase-flips
     exactly the basis states corresponding to marked pairs (the oracle
     is built mechanically from the classically-computed boolean array,
     not hand-picked), amplifies them with the standard diffuser, and
     runs it on the ideal AerSimulator.
  4. Compares the set of pairs (i.e. computational basis states) the
     quantum circuit finds with highest probability against the pairs
     classically computed as marked. PASS/FAIL is decided by exact set
     equality (all marked states must land among the top output
     probabilities, all unmarked states must not).

This is a real, if small, instance of Grover's algorithm doing quantum
amplitude amplification over the exact boolean structure defined by
problem #327's unit-fraction divisibility condition.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector


# ---------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS values copied)
# ---------------------------------------------------------------------

N = 6
NUM_QUBITS = 4  # 2^4 = 16 >= 15 pairs, state 15 left as unused padding

pairs = list(itertools.combinations(range(1, N + 1), 2))  # 15 pairs
assert len(pairs) == 15

def is_marked(t, u):
    """(t+u) divides (t*u) -- the unit-fraction integer-partner condition."""
    return (t * u) % (t + u) == 0

marked_pairs = [(t, u) for (t, u) in pairs if is_marked(t, u)]
marked_indices = sorted(pairs.index(p) for p in marked_pairs)

print("N =", N)
print("All pairs (index: pair, marked?):")
for idx, (t, u) in enumerate(pairs):
    print(f"  {idx:2d}: ({t},{u})  t+u={t+u:2d} t*u={t*u:2d}  marked={is_marked(t, u)}")
print("Marked pairs (t+u | t*u):", marked_pairs)
print("Marked basis-state indices:", marked_indices)

M = len(marked_indices)
Nstates = 2 ** NUM_QUBITS  # 16
assert 0 < M < Nstates


# ---------------------------------------------------------------------
# 2. Build the Grover oracle mechanically from marked_indices
# ---------------------------------------------------------------------

def apply_mark_flip(qc, index, num_qubits):
    """Phase-flip the |index> basis state (multi-controlled Z via X + H/MCX/H)."""
    bits = format(index, f"0{num_qubits}b")[::-1]  # bit i -> qubit i
    zero_qubits = [i for i, b in enumerate(bits) if b == "0"]

    for q in zero_qubits:
        qc.x(q)

    # multi-controlled Z on all qubits: H on last qubit, MCX, H
    controls = list(range(num_qubits - 1))
    target = num_qubits - 1
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)

    for q in zero_qubits:
        qc.x(q)


def oracle(num_qubits, indices):
    qc = QuantumCircuit(num_qubits, name="Oracle")
    for idx in indices:
        apply_mark_flip(qc, idx, num_qubits)
    return qc


def diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


# ---------------------------------------------------------------------
# 3. Assemble and run the Grover circuit
# ---------------------------------------------------------------------

iterations = max(1, round((math.pi / 4) * math.sqrt(Nstates / M)))
print(f"\nM = {M} marked states out of {Nstates}; using {iterations} Grover iteration(s)")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

orc = oracle(NUM_QUBITS, marked_indices)
dif = diffuser(NUM_QUBITS)

for _ in range(iterations):
    qc.compose(orc, inplace=True)
    qc.compose(dif, inplace=True)

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

sim = AerSimulator()
shots = 20000
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order is little-endian in the classical register string;
# convert each outcome string back to the integer basis-state index.
state_counts = {}
for bitstring, c in counts.items():
    # Qiskit's classical bitstring is ordered clbit[n-1]...clbit[0] (clbit0
    # is the rightmost/least-significant character), and qubit i was
    # measured into clbit i, so int(bitstring, 2) already reconstructs the
    # index with qubit 0 as the least-significant bit -- no reversal needed.
    idx = int(bitstring, 2)
    state_counts[idx] = state_counts.get(idx, 0) + c

sorted_states = sorted(state_counts.items(), key=lambda kv: -kv[1])
print("\nTop measured basis states (index: count, pair, was it marked classically):")
for idx, c in sorted_states[:M + 3]:
    pair = pairs[idx] if idx < len(pairs) else None
    print(f"  {idx:2d}: count={c:5d}  pair={pair}  classically_marked={idx in marked_indices}")


# ---------------------------------------------------------------------
# 4. Verify: the M most-measured states must be exactly the marked ones
# ---------------------------------------------------------------------

top_m_states = set(idx for idx, _ in sorted_states[:M])
classical_marked_set = set(marked_indices)

verified = top_m_states == classical_marked_set

print("\nClassically marked states:", sorted(classical_marked_set))
print("Quantum top-", M, "measured states:", sorted(top_m_states))

if verified:
    print("\nPASS: Grover search recovered exactly the classically-marked pairs.")
else:
    print("\nFAIL: quantum result does not match the classical computation.")

assert verified, "quantum/classical mismatch"
