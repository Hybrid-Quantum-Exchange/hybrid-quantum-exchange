"""
Erdos problem #611 (https://www.erdosproblems.com/611) — quantum-testable lane.

Source metadata (data/problems.yaml in manman4/erdosproblems, verified 2026-09-19):
    number: "611"
    tags: ["graph theory"]
    oeis: ["N/A"]
    formal_status: unformalized
    status: open

LIMITATION, stated up front: problem #611 carries no OEIS sequence id in the
source data ("N/A") and is not formalized. There is therefore no OEIS-derived
integer sequence to test membership/terms of, and this script cannot honestly
claim to verify anything about Erdos problem #611 itself. What follows is the
best-effort honest fallback the task instructions ask for in that case: a
small, genuinely computable property drawn from the problem's own tag
("graph theory") — independent sets of a fixed small graph — verified both
classically and with a real Grover search circuit on AerSimulator. This is a
generic graph-theory instance chosen because the problem is *about* graph
theory, not a property of any specific sequence attached to #611 (none
exists).

Classical property under test
------------------------------
Graph: the 4-cycle C4 on vertices {0,1,2,3} with edges (0,1),(1,2),(2,3),(3,0).
Subsets of vertices are encoded as 4-bit strings (bit i = 1 iff vertex i is
selected), so the search space has N = 2^4 = 16 candidates.

Property being searched for: subsets S of size exactly 2 that are
*independent sets* of C4 (no edge of the graph has both endpoints in S).

By direct enumeration (done classically in this script, not copied from
anywhere) the independent sets of size 2 in C4 are exactly:
    {0, 2}  and  {1, 3}
i.e. bitstrings 0101 and 1010 (bit0..bit3 left-to-right as vertex 0..3),
2 marked states out of 16.

Quantum approach
-----------------
A genuine Grover search circuit (4 qubits + oracle + diffuser) is built,
using the standard optimal iteration count for M=2 marked states out of
N=16 (~2 iterations), and run on the ideal AerSimulator. PASS is declared
if the two classically-known marked bitstrings together receive the
overwhelming majority of measurement probability (amplified far above the
uniform 2/16 = 12.5% baseline), matching the classically precomputed answer.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: enumerate independent sets of size 2 in C4.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def is_independent_set(subset, edges):
    subset = set(subset)
    for (u, v) in edges:
        if u in subset and v in subset:
            return False
    return True


classical_marked = []
for subset in itertools.combinations(VERTICES, 2):
    if is_independent_set(subset, EDGES):
        classical_marked.append(subset)

assert classical_marked == [(0, 2), (1, 3)], (
    f"unexpected classical result: {classical_marked}"
)

# bitstring convention: qubit i (i=0..3) represents vertex i; bit string
# printed by Qiskit is q3 q2 q1 q0 (little-endian display), so build the
# target integers directly from the vertex-set membership using bit i for
# vertex i.
def subset_to_int(subset):
    val = 0
    for v in subset:
        val |= (1 << v)
    return val


marked_ints = sorted(subset_to_int(s) for s in classical_marked)
N = 16  # 2^4 candidates
M = len(marked_ints)  # number of marked states
print(f"Classical answer: independent sets of size 2 in C4 = {classical_marked}")
print(f"Marked bitstring integers (vertex-bit encoding): {marked_ints}, M={M}, N={N}")

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle that flips the phase of exactly marked_ints.
# ---------------------------------------------------------------------------

NUM_QUBITS = 4


def build_oracle(marked_ints, num_qubits):
    qc = QuantumCircuit(num_qubits, name="oracle")
    for target in marked_ints:
        bits = [(target >> i) & 1 for i in range(num_qubits)]
        # flip qubits that should be 0 in the target so an all-ones pattern
        # corresponds to "this is the target state"
        flip_qubits = [i for i, b in enumerate(bits) if b == 0]
        for q in flip_qubits:
            qc.x(q)
        # multi-controlled Z on all qubits (phase flip when all are |1>)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
            qc.h(num_qubits - 1)
        for q in flip_qubits:
            qc.x(q)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


oracle = build_oracle(marked_ints, NUM_QUBITS)
diffuser = build_diffuser(NUM_QUBITS)

# optimal number of Grover iterations for M marked out of N states
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations}")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(NUM_QUBITS))
    qc.append(diffuser.to_gate(), range(NUM_QUBITS))
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 8192
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-register bitstrings are printed MSB..LSB i.e. c3c2c1c0,
# which matches our little-endian vertex-bit encoding directly when read as
# an integer via int(bitstring, 2).
marked_prob = 0
for bitstring, cnt in counts.items():
    val = int(bitstring, 2)
    if val in marked_ints:
        marked_prob += cnt
marked_prob /= shots

print(f"Measured probability mass on the two classically-marked states: {marked_prob:.4f}")
print(f"Raw counts: {counts}")

# ---------------------------------------------------------------------------
# 4. Compare to classical answer and report PASS/FAIL.
# ---------------------------------------------------------------------------

# Uniform baseline (no amplification) would give M/N = 2/16 = 0.125.
# A working Grover search on N=16, M=2 should push this well above baseline
# (theoretically close to 1 after the optimal number of iterations).
THRESHOLD = 0.7

if marked_prob >= THRESHOLD:
    print(f"PASS: quantum Grover search concentrated {marked_prob:.4f} probability "
          f"(>= {THRESHOLD}) on the classically verified independent sets "
          f"{classical_marked} of C4, versus a {M/N:.4f} uniform baseline.")
else:
    print(f"FAIL: quantum result ({marked_prob:.4f}) did not concentrate on the "
          f"classical answer {classical_marked} as expected.")
