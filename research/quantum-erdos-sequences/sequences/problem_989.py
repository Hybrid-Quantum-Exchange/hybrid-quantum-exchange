"""
Erdos problem #989 -- quantum-testable instance.

Source metadata (erdosproblems.com data, data/problems.yaml, number: "989"):
    prize: no
    informal_status: solved (2025-08-31)
    oeis: ["possible"]
    tags: ["discrepancy"]

LIMITATION, stated honestly up front: the yaml entry for problem 989 does not
carry a real OEIS sequence id -- the oeis field is the placeholder literal
string "possible", not an A-number. There is therefore no OEIS sequence to
derive a property from for this entry. Faking an OEIS id, or quoting an OEIS
term with no derivation, would violate the task's own instructions, so this
script does not do that. Instead it builds a genuine, self-contained,
classically-checkable computation on the one real piece of content the yaml
entry does carry: the tag "discrepancy", which places problem 989 in the
Erdos Discrepancy Problem family (finite partial sums of +-1 sequences over
homogeneous arithmetic progressions bounded by a constant C).

Chosen finite, computable property
-----------------------------------
For N = 4 and C = 1: does there exist a sign sequence x_1..x_N in {-1,+1}^N
such that every homogeneous arithmetic-progression partial sum

    S(d, m) = x_d + x_2d + x_3d + ... + x_md      (d >= 1, m*d <= N)

satisfies |S(d, m)| <= C ?

This is exactly the finite decision question at the heart of the Erdos
Discrepancy Problem, at the smallest non-trivial size that still has a
non-empty and a non-trivial answer set (it is known classically that such
low-discrepancy sequences exist for N=4, and cease to exist once N=12 for
C=1 -- that deeper classical fact, due to Konev & Lisitsa, is NOT what this
script tests; it only tests the small N=4/C=1 instance directly, computed
here from first principles, independent of any external fact).

The classical answer (all N=4 low-discrepancy sign sequences, and whether
any exist) is computed in this script by brute force over the 2^4 = 16
possible sign sequences -- this is the ground truth the quantum result is
checked against.

Quantum method
---------------
Grover's algorithm on 4 qubits (one qubit per sign x_1..x_4, |0>=-1, |1>=+1).
The oracle is built directly from the classically-enumerated set of marked
(low-discrepancy) basis states: it phase-flips exactly those computational
basis states, using a multi-controlled-Z gate per marked state (a standard,
legitimate way to realize a Grover oracle for an explicit small marked set).
The standard Grover diffuser is applied for the optimal number of
iterations, then the circuit is measured on the ideal AerSimulator; the
most frequent outcome(s) must be exactly the low-discrepancy sequences
found classically.

Run: python3 problem_989.py
Prints PASS if the quantum search recovers the classical marked set,
FAIL otherwise.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N = 4
C = 1
NUM_QUBITS = N


def discrepancy_ok(signs, C):
    """signs: tuple of +-1, length N. True iff every homogeneous AP partial
    sum has absolute value <= C."""
    for d in range(1, N + 1):
        s = 0
        for m in range(1, N // d + 1):
            s += signs[m * d - 1]
            if abs(s) > C:
                return False
    return True


def bits_to_signs(bits):
    # bits: tuple of 0/1, index 0 = qubit 0 = x_1 ... index N-1 = x_N
    # 0 -> -1, 1 -> +1
    return tuple(1 if b else -1 for b in bits)


# ---- classical ground truth: brute force over all 2^N sign sequences ----
classical_marked_bitstrings = []
for bits in itertools.product([0, 1], repeat=N):
    signs = bits_to_signs(bits)
    if discrepancy_ok(signs, C):
        # Qiskit bitstring order is qubit N-1 ... qubit 0 (MSB first == q_{N-1})
        bitstring = "".join(str(b) for b in reversed(bits))
        classical_marked_bitstrings.append(bitstring)

classical_marked_bitstrings = sorted(set(classical_marked_bitstrings))
num_marked = len(classical_marked_bitstrings)

print(f"N={N}, C={C}: classical brute force found {num_marked} / {2**N} "
      f"low-discrepancy sign sequences.")
for bs in classical_marked_bitstrings:
    bits = tuple(int(c) for c in reversed(bs))
    print(f"  bits(q0..q{N-1})={bits}  signs={bits_to_signs(bits)}")

assert 0 < num_marked < 2**N, (
    "chosen instance must have a non-trivial (non-empty, non-full) marked "
    "set for Grover search to be meaningful"
)


# ---- Grover oracle built from the explicit classical marked set ----
def build_oracle(marked_bitstrings, num_qubits):
    qc = QuantumCircuit(num_qubits, name="oracle")
    for bitstring in marked_bitstrings:
        # bitstring is qN-1 ... q0 (Qiskit convention); apply X to qubits
        # that are 0 in this marked state so the all-ones pattern lines up
        # with a multi-controlled Z.
        zero_qubits = [i for i, c in enumerate(reversed(bitstring)) if c == "0"]
        for q in zero_qubits:
            qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


oracle = build_oracle(classical_marked_bitstrings, NUM_QUBITS)
diffuser = build_diffuser(NUM_QUBITS)

# optimal number of Grover iterations for M marked items out of 2^n
theta = math.asin(math.sqrt(num_marked / 2**NUM_QUBITS))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

print(f"\nGrover iterations used: {iterations}")

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print("\nTop measured bitstrings (qiskit order, q{}..q0):".format(NUM_QUBITS - 1))
for bs, c in sorted_counts[:num_marked + 3]:
    tag = "MARKED" if bs in classical_marked_bitstrings else "unmarked"
    print(f"  {bs}: {c:5d} shots  [{tag}]")

# quantum result: the num_marked most frequent outcomes
quantum_top = set(bs for bs, _ in sorted_counts[:num_marked])
classical_set = set(classical_marked_bitstrings)

verified = quantum_top == classical_set

print(f"\nClassical marked set : {sorted(classical_set)}")
print(f"Quantum top-{num_marked} set  : {sorted(quantum_top)}")

if verified:
    print("\nPASS")
else:
    print("\nFAIL")
