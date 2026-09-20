"""
Erdos problem #874 -- quantum-testable instance.

Source data: /home/user/manman4/erdosproblems/data/problems.yaml, entry
`number: "874"` (tags: ["number theory", "additive combinatorics"],
oeis: ["possible"]).

IMPORTANT LIMITATION, stated honestly up front: the erdosproblems.com data
clone used here does not record a real OEIS sequence id for problem 874 --
its `oeis` field holds the placeholder string "possible", not an actual
A-number, and the clone contains no problem-statement text for #874 beyond
the YAML metadata (prize/status/tags). So this script is NOT verifying any
specific OEIS sequence tied to problem 874; that would require fabricating
a sequence link that the source data does not support, which the task
instructions explicitly forbid. Faking that connection would not be honest.

Instead, this script honors the *tags* of problem 874 ("number theory",
"additive combinatorics") with a real, small, finite, classically-checkable
property from that area of mathematics, and verifies a genuine quantum
circuit (Grover search) against the classical ground truth for that
property. The property chosen is:

    Sidon set membership (a.k.a. B_2 set): a finite set S of integers is a
    Sidon set iff all pairwise sums a+b (a,b in S, a<=b) are distinct.
    Sidon sets are a classical object of additive combinatorics -- exactly
    the area problem 874 is tagged with -- and "is this specific finite
    set a Sidon set" is a small, decidable, easily verified property.

Concrete finite instance:
    We fix 8 candidate 3-element subsets of {1,...,9} (indexed 0..7, i.e.
    a 3-qubit search space). For each candidate we classically decide
    whether it is a Sidon set (all C(3,2)+3 = 6 pairwise sums a+b, a<=b,
    distinct). We then run Grover's algorithm on a 3-qubit register whose
    oracle marks exactly the indices of the Sidon-set candidates, and check
    that measurement recovers precisely the classically-computed marked set
    (as the highest-probability outcomes).

The classical answer (which indices are Sidon sets) is computed here from
first principles, in this script, before the quantum circuit is built.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile


# ---------------------------------------------------------------------------
# 1. Fixed finite instance: 8 candidate 3-element subsets of {1,...,9}.
# ---------------------------------------------------------------------------

CANDIDATES = [
    (1, 2, 3),
    (2, 3, 4),
    (3, 4, 5),
    (4, 5, 6),
    (5, 6, 7),
    (6, 7, 8),
    (1, 2, 4),
    (1, 3, 7),
]
N = len(CANDIDATES)
NUM_QUBITS = int(math.log2(N))
assert 2 ** NUM_QUBITS == N


def is_sidon_set(subset):
    """A finite set of integers is Sidon iff all pairwise sums a+b (a<=b,
    both in subset) are distinct. Computed here from first principles."""
    sums = []
    for a, b in itertools.combinations_with_replacement(subset, 2):
        sums.append(a + b)
    return len(sums) == len(set(sums))


# ---------------------------------------------------------------------------
# 2. Classical ground truth.
# ---------------------------------------------------------------------------

classical_marked = [i for i, s in enumerate(CANDIDATES) if is_sidon_set(s)]

print("Candidates (index: subset -> is Sidon set?):")
for i, s in enumerate(CANDIDATES):
    print(f"  {i} ({i:0{NUM_QUBITS}b}): {s} -> {is_sidon_set(s)}")
print(f"Classical marked indices (Sidon sets): {classical_marked}")

if not (1 <= len(classical_marked) <= N - 1):
    raise RuntimeError(
        "Instance is degenerate for Grover (need at least one marked and "
        "one unmarked item); adjust CANDIDATES."
    )


# ---------------------------------------------------------------------------
# 3. Grover oracle + diffuser, built directly from the classical answer.
# ---------------------------------------------------------------------------

def build_oracle(marked_indices, num_qubits):
    qc = QuantumCircuit(num_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{num_qubits}b")
        # Flip qubits that should be 0 in this index, so a multi-controlled
        # Z fires only on |idx>, then flip back.
        for q, bit in enumerate(reversed(bits)):
            if bit == "0":
                qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for q, bit in enumerate(reversed(bits)):
            if bit == "0":
                qc.x(q)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


num_marked = len(classical_marked)
# Optimal number of Grover iterations for this N and number of marked items.
theta = math.asin(math.sqrt(num_marked / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

oracle = build_oracle(classical_marked, NUM_QUBITS)
diffuser = build_diffuser(NUM_QUBITS)
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(NUM_QUBITS))
    qc.append(diffuser.to_gate(), range(NUM_QUBITS))

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

print(f"\nGrover iterations used: {iterations}")


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

print("\nMeasurement counts (bitstring -> count):")
for bitstring, count in sorted(counts.items(), key=lambda kv: -kv[1]):
    idx = int(bitstring, 2)
    print(f"  {bitstring} (index {idx}): {count}")

# The quantum-predicted marked set: indices whose measured probability is
# well above the uniform baseline 1/N, i.e. Grover actually amplified them.
baseline = shots / N
threshold = baseline * 1.5
quantum_marked = sorted(
    int(bs, 2) for bs, c in counts.items() if c > threshold
)

print(f"\nQuantum-recovered marked indices (amplified above baseline): {quantum_marked}")
print(f"Classical marked indices:                                    {sorted(classical_marked)}")

passed = quantum_marked == sorted(classical_marked)

print("\nPASS" if passed else "\nFAIL")
