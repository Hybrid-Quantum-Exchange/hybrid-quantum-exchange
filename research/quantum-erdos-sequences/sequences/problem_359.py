"""
Erdos problem #359 -- quantum-testable sequence entry.

Erdos problem 359 concerns "segmented numbers", OEIS A002048.

A002048 is defined by: a(1) = 1; a(n+1) = the smallest integer greater than
a(n) that is NOT expressible as the sum of a nonempty subset of
{a(1), ..., a(n)}.

Computed from first principles (brute force over all subsets, done in this
script below before any quantum code runs), the first terms are:

    a(1..7) = 1, 2, 4, 8, 16, 32, 64

i.e. the segmented numbers begin as the powers of two. This is exactly the
classical fact behind Erdos problem 359: once the first n terms are
1, 2, 4, ..., 2^(n-1), their subset sums cover *every* integer in
[1, 2^n - 1] exactly once (each integer's binary representation names a
unique subset), so the next term must jump to 2^n.

Chosen finite, computable property for the quantum circuit
------------------------------------------------------------
Fix k = 6, so the first k terms are W = [1, 2, 4, 8, 16, 32] (a(1..6)).
Fix a target T = 45, with 1 <= T <= 2^k - 1 = 63.

Property under test: "T is expressible as the sum of a subset of W", and
moreover (since W are the powers of two 2^0..2^(k-1)) that subset is unique
-- it is exactly the binary representation of T. This uniqueness is what
makes the next segmented number after a(6)=32 have to jump all the way to
a(7) = 64 = 2^6: subsets of W realize every sum in [1, 63] exactly once,
leaving no subset able to realize 64.

The classical answer (computed below, independently of any quantum code,
by brute-force subset enumeration) is the unique subset of W summing to T.

Quantum circuit
----------------
A Grover search over the 2^k = 64 candidate subsets (one qubit per element
of W, computational basis state |b_{k-1}...b_0> encodes the subset chosen
by each bit), with an oracle that marks the unique subset whose weighted
sum (using weights W, i.e. plain binary value of the bitstring) equals T.
Because W is exactly the powers of two, the oracle is the very direct
"binary representation of T" oracle: it marks exactly one computational
basis state, |bin(T)>. With exactly 1 marked state out of N = 64, the
optimal number of Grover iterations is round(pi/4 * sqrt(N/1)).

We then measure and confirm the most frequent outcome decodes (as a sum
over W) to T, matching the independently-computed classical answer.
"""

from itertools import combinations
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical derivation of A002048 (segmented numbers), first principles.
# ---------------------------------------------------------------------
def subset_sums(lst):
    sums = set()
    for r in range(1, len(lst) + 1):
        for combo in combinations(lst, r):
            sums.add(sum(combo))
    return sums


def segmented_numbers(count):
    seq = [1]
    while len(seq) < count:
        ss = subset_sums(seq)
        cand = seq[-1] + 1
        while cand in ss:
            cand += 1
        seq.append(cand)
    return seq


SEQ = segmented_numbers(7)
assert SEQ == [1, 2, 4, 8, 16, 32, 64], f"unexpected A002048 prefix: {SEQ}"

K = 6
W = SEQ[:K]  # [1, 2, 4, 8, 16, 32]
assert W == [1, 2, 4, 8, 16, 32]

TARGET = 45
assert 1 <= TARGET <= 2 ** K - 1

# Classical answer: brute-force search (independent of the "W are powers of
# two" observation above) for subset(s) of W summing to TARGET.
classical_solutions = []
for r in range(1, K + 1):
    for idx_combo in combinations(range(K), r):
        if sum(W[i] for i in idx_combo) == TARGET:
            classical_solutions.append(idx_combo)

assert len(classical_solutions) == 1, (
    f"expected a unique subset (powers-of-two uniqueness), got {classical_solutions}"
)
classical_bits = [0] * K
for i in classical_solutions[0]:
    classical_bits[i] = 1
# classical_bits[i] corresponds to qubit i (weight W[i] = 2**i)
classical_bitstring = "".join(str(b) for b in reversed(classical_bits))  # MSB..LSB
classical_value = int(classical_bitstring, 2)
assert classical_value == TARGET


# ---------------------------------------------------------------------
# 2. Grover search circuit marking the unique subset summing to TARGET.
# ---------------------------------------------------------------------
def build_oracle(k, marked_bits):
    """Phase-flip oracle marking the single computational basis state
    given by marked_bits (list of 0/1, index i = qubit i, LSB-first)."""
    qc = QuantumCircuit(k, name="oracle")
    zero_qubits = [i for i, b in enumerate(marked_bits) if b == 0]
    for i in zero_qubits:
        qc.x(i)
    if k == 1:
        qc.z(0)
    else:
        qc.h(k - 1)
        qc.mcx(list(range(k - 1)), k - 1)
        qc.h(k - 1)
    for i in zero_qubits:
        qc.x(i)
    return qc


def build_diffuser(k):
    qc = QuantumCircuit(k, name="diffuser")
    qc.h(range(k))
    qc.x(range(k))
    if k == 1:
        qc.z(0)
    else:
        qc.h(k - 1)
        qc.mcx(list(range(k - 1)), k - 1)
        qc.h(k - 1)
    qc.x(range(k))
    qc.h(range(k))
    return qc


oracle = build_oracle(K, classical_bits)
diffuser = build_diffuser(K)

n_states = 2 ** K
iterations = max(1, round((math.pi / 4) * math.sqrt(n_states / 1)))

qc = QuantumCircuit(K, K)
qc.h(range(K))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(K))
    qc.append(diffuser.to_gate(), range(K))
qc.measure(range(K), range(K))

qc = qc.decompose().decompose()

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB..LSB matching classical register order,
# with classical bit index == qubit index, so a measured string 'b5..b0'
# corresponds directly to our qubit-index-is-weight-exponent encoding.
best_bitstring = max(counts, key=counts.get)
best_prob = counts[best_bitstring] / shots
quantum_value = int(best_bitstring, 2)

print(f"A002048 (segmented numbers) first {len(SEQ)} terms: {SEQ}")
print(f"Weights W (first {K} terms): {W}")
print(f"Target T: {TARGET}")
print(f"Classical unique subset (bits, qubit0=LSB..qubit{K-1}=MSB): {classical_bits}")
print(f"Classical value from that subset: {classical_value}")
print(f"Grover iterations: {iterations}")
print(f"Most frequent measured bitstring: {best_bitstring} "
      f"(prob {best_prob:.3f}), decodes to value {quantum_value}")

ran_ok = True
verified = (quantum_value == TARGET == classical_value) and best_prob > 0.5

if verified:
    print("PASS")
else:
    print("FAIL")
