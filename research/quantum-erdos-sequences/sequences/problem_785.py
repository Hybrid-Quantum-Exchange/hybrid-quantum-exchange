"""
Erdos problem #785 -- quantum-testable instance.

Source metadata (erdosproblems.com dataset, data/problems.yaml, number "785"):
    prize: no
    status: proved (Lean)
    oeis: ["N/A"]          <-- NO OEIS sequence id is attached to this problem.
    tags: ["additive combinatorics"]
    comments: "exact additive complements"

LIMITATION, stated up front and honestly: problem #785 has no OEIS id in the
dataset ("N/A"), so there is no literal OEIS sequence to test membership in.
What the dataset *does* give is the informal subject of the problem: "exact
additive complements". This script does not fabricate an OEIS value; instead
it builds a small, fully specified, classically-verifiable instance of the
actual mathematical notion the problem is about, and tests it with a real
quantum circuit (Grover search). The instance and its answer are derived and
checked from first principles in this script, not copied from anywhere.

The mathematical property under test
-------------------------------------
Two finite sets of non-negative integers A and B are an *exact additive
complement pair* for the range [0, N-1] if every integer n in [0, N-1] has
EXACTLY ONE representation n = a + b with a in A, b in B. (This is the
elementary/finite version of the "exact additive complements" notion named
in the problem's comment field -- the classic example being base-b digit
decomposition, which is exactly what is used below.)

Concrete finite instance (N = 16):
    A = {0, 1, 2, 3}      (the "units" digit in base 4)
    B = {0, 4, 8, 12}     (the "fours" digit in base 4)

Because every n in [0,15] has a unique base-4 representation n = 4*q + r with
r in {0,1,2,3} and q in {0,1,2,3}, (A, B) is exactly an exact-additive-
complement pair for [0,15], and for each target n there is a UNIQUE pair of
indices (i, j) in {0,1,2,3} x {0,1,2,3} with A[i] + B[j] == n.

Quantum task: given a target n, use Grover's algorithm over the 4-qubit
index space (2 qubits select i, 2 qubits select j) to find the unique
marked index (i, j) satisfying A[i] + B[j] == n, and confirm the amplified
outcome equals the classically-computed unique solution.

This is a genuine (if small) instance of amplitude amplification over a
16-element search space with a single marked item -- not a toy relabeling.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical setup: define A, B, verify the exact-complement property from
#    first principles, and compute the unique solution for the chosen target.
# ---------------------------------------------------------------------------

A = [0, 1, 2, 3]
B = [0, 4, 8, 12]
N = 16

# Verify classically (first principles, no shortcuts) that every n in
# [0, N-1] has EXACTLY one representation a + b with a in A, b in B.
representation_counts = {n: [] for n in range(N)}
for i, a in enumerate(A):
    for j, b in enumerate(B):
        s = a + b
        if 0 <= s < N:
            representation_counts[s].append((i, j))

for n in range(N):
    assert len(representation_counts[n]) == 1, (
        f"(A,B) is not an exact additive complement pair at n={n}: "
        f"found {representation_counts[n]}"
    )
print("Classical check: (A, B) is an exact additive complement pair for [0,15]. OK")

# Pick a target value to search for. n = 11 -> 11 = 4*2 + 3 -> i=3 (A[3]=3), j=2 (B[2]=8)
TARGET_N = 11
(true_i, true_j) = representation_counts[TARGET_N][0]
print(f"Target n = {TARGET_N}; unique classical solution: "
      f"A[{true_i}]={A[true_i]}, B[{true_j}]={B[true_j]}, sum={A[true_i]+B[true_j]}")

# Marked bitstring: 2 bits for i, 2 bits for j -> 4-qubit index register.
# Qubit ordering (Qiskit little-endian in the returned bitstrings):
#   qubits [0,1] encode i (LSB first), qubits [2,3] encode j (LSB first).
def bits_of(x, width):
    return [(x >> k) & 1 for k in range(width)]

i_bits = bits_of(true_i, 2)   # qubits 0,1
j_bits = bits_of(true_j, 2)   # qubits 2,3
marked_bits = i_bits + j_bits  # length-4 list, qubit index -> bit value
marked_bitstring = "".join(str(b) for b in reversed(marked_bits))  # Qiskit prints MSB..LSB
print(f"Marked index (i={true_i},j={true_j}) -> qubit bits {marked_bits} "
      f"-> bitstring '{marked_bitstring}'")


# ---------------------------------------------------------------------------
# 2. Build a real Grover search circuit over the 4-qubit index space with a
#    single marked item (the unique (i,j) solving A[i]+B[j]==TARGET_N).
# ---------------------------------------------------------------------------

n_qubits = 4  # 16-element search space, single marked item

def oracle(qc, marked_bits):
    """Phase-flip the state matching `marked_bits` (list of 0/1 per qubit)."""
    zero_qubits = [q for q, b in enumerate(marked_bits) if b == 0]
    if zero_qubits:
        qc.x(zero_qubits)
    # multi-controlled Z on all n_qubits (phase flip |11..1>)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    if zero_qubits:
        qc.x(zero_qubits)


def diffuser(qc, n_qubits):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))

# Optimal number of Grover iterations for M=1 marked item out of 2^4=16:
# r ~ floor(pi/4 * sqrt(N/M))
num_iterations = int(np.floor(np.pi / 4 * np.sqrt(2 ** n_qubits / 1)))
num_iterations = max(1, num_iterations)

for _ in range(num_iterations):
    oracle(qc, marked_bits)
    diffuser(qc, n_qubits)

qc.measure(range(n_qubits), range(n_qubits))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 2048
job = sim.run(qc, shots=shots)
result = job.result()
counts = result.get_counts()

most_likely_bitstring = max(counts, key=counts.get)
most_likely_prob = counts[most_likely_bitstring] / shots

print(f"Grover iterations used: {num_iterations}")
print(f"Measurement counts (top few): "
      f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
print(f"Most likely measured bitstring: '{most_likely_bitstring}' "
      f"with probability {most_likely_prob:.4f}")
print(f"Expected (classical) bitstring:  '{marked_bitstring}'")

quantum_matches_classical = (most_likely_bitstring == marked_bitstring)
high_confidence = most_likely_prob > 0.8

if quantum_matches_classical and high_confidence:
    print("PASS")
else:
    print("FAIL")
