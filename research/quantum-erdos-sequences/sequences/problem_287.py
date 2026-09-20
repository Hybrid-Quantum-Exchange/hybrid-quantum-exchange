"""
Erdos problem #287 -- quantum-testable instance.

Source metadata (erdosproblems.com data, data/problems.yaml, number: "287"):
    prize: no
    status: falsifiable (as of 2025-12-05)
    oeis: ["N/A"]
    tags: ["number theory", "unit fractions"]

Problem #287 has NO associated OEIS sequence id in the source data (oeis is
literally "N/A"), so there is no OEIS sequence for a quantum circuit to
"test membership in" or "compute an early term of" in the sense the other
lanes in this library do. Per the task's fallback instructions, this script
makes its best honest attempt at a genuine, finite, computable property that
is faithful to the problem's *tags* ("unit fractions", "number theory")
rather than fabricating an OEIS value that does not exist.

Chosen property (Egyptian-fraction / unit-fraction search, tag-faithful,
NOT a claim about the literal Erdos #287 statement):

    Grover search over subsets S of the fixed base set B = {2, 3, 4, 5, 6}
    (5 elements -> 5 qubits, search space size 32) for a subset whose unit
    fractions sum to exactly 1:

        sum_{i in S} 1/i == 1

    The unique classical answer for this base set is S = {2, 3, 6}
    (1/2 + 1/3 + 1/6 = 1), the smallest and most classical Egyptian-fraction
    identity there is. This is computed from first principles below using
    Python's exact `fractions.Fraction` arithmetic over every one of the 32
    subsets of B -- nothing is copied from a table.

Circuit: a genuine Grover search circuit. The oracle is built directly from
the classically-computed marked subset(s) (a multi-controlled-Z phase flip
on the matching basis state(s), i.e. a black-box oracle over the boolean
function f(S) = [sum_{i in S} 1/i == 1]), composed with the standard
Grover diffusion operator, run for the optimal number of iterations on the
ideal AerSimulator. The script prints PASS if the state(s) measured with
highest probability match the classically-computed marked subset(s), else
FAIL.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from fractions import Fraction
from itertools import combinations
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, exact rational arithmetic).
# ---------------------------------------------------------------------------

BASE = [2, 3, 4, 5, 6]  # 5 elements -> 5 qubits, 32 subsets
N = len(BASE)


def subset_sum_is_one(bits):
    """bits: tuple of 0/1 of length N, bit i selects BASE[i]."""
    total = Fraction(0)
    for i, b in enumerate(bits):
        if b:
            total += Fraction(1, BASE[i])
    return total == 1


marked_indices = []
marked_subsets = []
for idx in range(2 ** N):
    bits = tuple((idx >> i) & 1 for i in range(N))
    if subset_sum_is_one(bits):
        marked_indices.append(idx)
        marked_subsets.append([BASE[i] for i in range(N) if bits[i]])

assert marked_indices, "classical search found no marked subset -- cannot build a Grover instance"

print(f"Base set B = {BASE}")
print(f"Classical search over all {2 ** N} subsets of B found "
      f"{len(marked_indices)} subset(s) with sum of unit fractions == 1:")
for s, idx in zip(marked_subsets, marked_indices):
    print(f"    S = {s}  (index {idx} = {idx:05b}b)  "
          f"sum = {sum(Fraction(1, x) for x in s)}")

CLASSICAL_ANSWER = set(marked_indices)


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly those basis states.
# ---------------------------------------------------------------------------

def build_oracle(n_qubits, marked):
    """Phase-flip oracle: multi-controlled-Z on each marked computational
    basis state, using X gates to map the marked bitstring to |11...1>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked:
        bits = [(idx >> i) & 1 for i in range(n_qubits)]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
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


M = len(marked_indices)
theta = math.asin(math.sqrt(M / 2 ** N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N, N)
qc.h(range(N))
oracle = build_oracle(N, marked_indices)
diffuser = build_diffuser(N)
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N))
    qc.append(diffuser.to_gate(), range(N))
qc.measure(range(N), range(N))

print(f"\nGrover iterations used: {iterations} (search space size {2 ** N}, "
      f"{M} marked state(s))")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 4096
qc_decomposed = qc.decompose(reps=3)
result = sim.run(qc_decomposed, shots=shots).result()
counts = result.get_counts()

# Qiskit prints classical bits as c_{n-1}...c_0 (c0, from qubit 0, is the
# rightmost character), which is exactly the standard binary representation
# of idx = sum_i bit(qubit_i) * 2**i, so no reversal is needed.
def bitstring_to_index(bitstring):
    return int(bitstring, 2)

counts_by_index = {}
for bitstring, cnt in counts.items():
    counts_by_index[bitstring_to_index(bitstring)] = cnt

top_index, top_count = max(counts_by_index.items(), key=lambda kv: kv[1])
top_prob = top_count / shots

print(f"\nTop measured index: {top_index:05b}b (index {top_index}), "
      f"probability {top_prob:.3f} over {shots} shots")
print(f"Classically marked index/indices: {sorted(CLASSICAL_ANSWER)}")

# Success criterion: the most frequently measured basis state must be one of
# the classically-marked states, and its probability must be well above the
# uniform baseline 1/2^N, i.e. amplification genuinely occurred.
baseline = 1 / 2 ** N
quantum_result_matches_classical = (
    top_index in CLASSICAL_ANSWER and top_prob > 5 * baseline
)

print(f"\nBaseline (uniform) probability: {baseline:.4f}")
print(f"Amplified probability observed: {top_prob:.4f}")

if quantum_result_matches_classical:
    print("\nPASS: Grover search on the ideal simulator recovered the "
          "classically-verified unit-fraction subset with amplified "
          "probability.")
else:
    print("\nFAIL: quantum result did not match the classical answer with "
          "sufficient amplification.")
